from __future__ import annotations

import json
import logging
import re

# temporarily add this in for neuropixel course
# until figured out fix to problem on win32
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import ants
import numpy as np
import one.alf.io as alfio
import pandas
import SimpleITK as sitk
from aind_data_access_api.helpers.data_schema import get_quality_control_by_id
from aind_registration_utils.annotations import expand_compacted_image
from iblatlas import atlas
from iblatlas.regions import BrainRegions
from iblutil.util import Bunch
from numpy.typing import NDArray
from one import alf

from ephys_alignment_gui.custom_atlas import _BLESSED_DIRECTION, BrainAtlasAnatomical
from ephys_alignment_gui.docdb import docdb_api_client, query_docdb_id

ssl._create_default_https_context = ssl._create_unverified_context
logger = logging.getLogger(__name__)

ANTS_DIMENSION = 3


def _cut_slice_from_atlas_image(
    atlas_array: NDArray,
    xyz_channel_indices: NDArray,
    func: Callable[[NDArray], NDArray] | None = None,
) -> NDArray:
    """
    Extract a "wavy" slice from the atlas image given the xyz channel indices.
    xyz channel indices are indices in the atlas array, not physical coordinates.

    Parameters
    ----------
    atlas_array : NDArray
        The atlas image array from which to extract the slice.
    xyz_channel_indices : NDArray
        The xyz channel indices in the atlas array.
    func : Callable[[NDArray], NDArray] | None, optional
        An optional function to apply to the extracted slice, by default None.

    Returns
    -------
    NDArray
        The extracted slice from the atlas image.
    """
    # N x image.shape[1]
    slice = atlas_array[xyz_channel_indices[:, 0], :, xyz_channel_indices[:, 2]]
    if func is not None:
        slice = func(slice)
    slice = np.swapaxes(slice, 0, 1)  # Now it's image.shape[1] x N [x 3 for RGB]
    return slice


@dataclass(frozen=True)
class AntsTransformChainFiles:
    smartspim_template_affine_transform: Path
    smartspim_template_warp_transform: Path
    template_to_ccf_affine_transform: Path
    template_to_ccf_warp_transform: Path

    def as_list(self) -> list[str]:
        return [
            self.smartspim_template_affine_transform.as_posix(),
            self.smartspim_template_warp_transform.as_posix(),
            self.template_to_ccf_affine_transform.as_posix(),
            self.template_to_ccf_warp_transform.as_posix(),
        ]

    def which_to_invert(self) -> list[bool]:
        return [True, False, True, False]


@dataclass
class LoadDataLocal:
    brain_atlas: BrainAtlasAnatomical | None = None
    folder_path: Path | None = None
    histology_path: Path | None = None
    chn_coords: NDArray | None = None
    chn_coords_all: NDArray | None = None
    sess_path: Path | None = None
    n_shanks: int = 1
    data_root: Path | None = None
    previous_directory: Path | None = None

    histology_images: dict[str, sitk.Image] = field(default_factory=dict)
    channel_dict: dict[str, dict[str, Any]] = field(default_factory=dict)
    alignments: dict[str, list[list[float]]] = field(default_factory=dict)
    prev_align: list[str] = field(default_factory=list)

    def get_info(self, folder_path, shank_idx: int, input_path=None, skip_shanks=False):
        """
        Read in the local json file to see if any previous alignments exist
        """
        shank_list = None

        self.folder_path = input_path
        if not skip_shanks:
            shank_list = self.get_nshanks()

        prev_aligns = self.get_previous_alignments(
            shank_idx=shank_idx, folder_path=folder_path
        )
        return prev_aligns, shank_list

    def get_previous_info(self, folder_path):
        """
        Read in the local json file to see if any previous alignments exist
        """
        shank_list = self.get_nshanks()
        prev_aligns = self.get_previous_alignments(folder_path=folder_path)
        return prev_aligns, shank_list

    def get_previous_alignments(self, shank_idx=0, folder_path: Path | None = None):
        if folder_path is None:
            folder_path = self.folder_path

        logger.info("Checking docdb for existing records")

        quality_control = None
        try:
            docdb_id = query_docdb_id(folder_path.parent.stem)[0]
            quality_control = get_quality_control_by_id(docdb_api_client, docdb_id)
        except ValueError as e:
            logger.warning(
                f"Failed to get record from docdb with exception {e}. Proceeding to load from scratch"
            )

        if quality_control is not None:
            evaluations = quality_control.evaluations

            evaluation_name = (
                f"{folder_path.parent.stem}_{folder_path.stem}_{shank_idx}"
            )
            alignment_evaluations = [
                evaluation
                for evaluation in evaluations
                if evaluation.name == f"Probe Alignment for {evaluation_name}"
            ]

            if len(alignment_evaluations) > 0:
                logger.info(
                    f"Found exisitng record for {evaluation_name}. Loading alignment now"
                )
                latest_alignment_evaluation = max(
                    alignment_evaluations, key=lambda x: x.created
                )  # pull latest alignment evaluation
                curation_metric = latest_alignment_evaluation.metrics[0].value[
                    "curations"
                ]
                self.alignments = json.loads(curation_metric[0])[
                    "previous_alignments"
                ]  # load in the previous alignment
                self.prev_align = []
                if self.alignments:
                    self.prev_align = [*self.alignments.keys()]
                self.prev_align = sorted(self.prev_align, reverse=True)
                self.prev_align.append("original")
            else:
                logger.info(f"No alignment found in docdb for {evaluation_name}")
                self.alignments = {}
                self.prev_align = ["original"]
        else:
            # If previous alignment json file exists, read in previous alignments
            prev_align_filename = (
                "prev_alignments.json"
                if self.n_shanks == 1
                else f"prev_alignments_shank{shank_idx + 1}.json"
            )

            if folder_path.joinpath(prev_align_filename).exists():
                with open(folder_path.joinpath(prev_align_filename), "r") as f:
                    self.alignments = json.load(f)
                    self.prev_align = []
                    if self.alignments:
                        self.prev_align = [*self.alignments.keys()]
                    self.prev_align = sorted(self.prev_align, reverse=True)
                    self.prev_align.append("original")
            else:
                self.alignments = {}
                self.prev_align = ["original"]

        return self.prev_align

    def get_starting_alignment(
        self, idx: int, shank_idx=0, folder_path: Path | None = None
    ):
        """
        Find out the starting alignment
        """
        align = self.get_previous_alignments(
            shank_idx=shank_idx, folder_path=folder_path
        )[idx]

        if align == "original":
            feature = None
            track = None
        else:
            feature = np.array(self.alignments[align][0])
            track = np.array(self.alignments[align][1])

        return feature, track

    def get_nshanks(self):
        """
        Find out the number of shanks on the probe, either 1 or 4
        """
        self.chn_coords_all = np.load(
            self.folder_path.joinpath("channels.localCoordinates.npy")
        )

        chn_x = np.unique(self.chn_coords_all[:, 0])
        chn_x_diff = np.diff(chn_x)
        self.n_shanks = np.sum(chn_x_diff > 100) + 1

        if self.n_shanks == 1:
            shank_list = ["1/1"]
        else:
            shank_list = [
                f"{iShank + 1}/{self.n_shanks}" for iShank in range(self.n_shanks)
            ]

        return shank_list

    def get_data(self, shank_idx, reload_data: bool = True):
        if reload_data:
            if not hasattr(self, "atlas_image_path"):
                search_path: Path = self.folder_path.parent.parent

                def _glob_nonempty_or_err(pattern: str) -> tuple[Path, ...]:
                    paths = tuple(search_path.glob(pattern))
                    if not paths:
                        raise FileNotFoundError(
                            f"Could not find path to atlas image in data asset attached. Looking for pattern {pattern} in {search_path}"
                        )
                    return paths

                self.atlas_image_path = _glob_nonempty_or_err(
                    "image_space_histology/ccf_in_*.nrrd"
                )
                self.atlas_labels_path = _glob_nonempty_or_err(
                    "image_space_histology/labels_in_*.nrrd"
                )
                self.pipeline_image_path = _glob_nonempty_or_err(
                    "image_space_histology/histology_registration_pipeline.nrrd"
                )
                self.histology_image_path = _glob_nonempty_or_err(
                    "image_space_histology/histology_registration.nrrd"
                )
                self.histology_path = self.atlas_image_path[0].parent
                intensity_image = sitk.ReadImage(self.atlas_image_path[0])
                label_image_compacted = sitk.ReadImage(self.atlas_labels_path[0])
                pipeline_image = sitk.ReadImage(self.pipeline_image_path[0])
                unq_annotations = np.load(
                    "/data/allen_mouse_ccf_annotations_lateralized_compact/ccf_2017_annotation_25_lateralized_unique_vals.npz"
                )["unique_labels"]
                label_image = expand_compacted_image(
                    label_image_compacted, unq_annotations
                )
                self.brain_atlas = BrainAtlasAnatomical(
                    intensity_img=intensity_image,
                    label_img=label_image,
                    pipeline_img=pipeline_image,
                )

                histology_image = sitk.ReadImage(self.histology_image_path[0])
                dicom_orient_str = (
                    sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                        histology_image.GetDirection()
                    )
                )
                if dicom_orient_str == _BLESSED_DIRECTION:
                    reorient = False
                else:
                    reorient = True
                    histology_image = sitk.DICOMOrient(
                        histology_image, _BLESSED_DIRECTION
                    )
                self.histology_images["histology_registration"] = histology_image
                pattern = re.compile(r"^Ex_\d+_Em_\d+\.nrrd$")
                for other_channel in self.histology_path.iterdir():
                    if pattern.match(other_channel.name):
                        channel_name = other_channel.stem
                        if channel_name not in self.histology_images:
                            channel_image = sitk.ReadImage(str(other_channel))
                            if reorient:
                                channel_image = sitk.DICOMOrient(channel_image, "SRA")
                            self.histology_images[channel_name] = channel_image

        chn_x = np.unique(self.chn_coords_all[:, 0])
        if self.n_shanks > 1:
            shanks = {}
            for iShank in range(self.n_shanks):
                shanks[iShank] = [chn_x[iShank * 2], chn_x[(iShank * 2) + 1]]

            shank_chns = np.bitwise_and(
                self.chn_coords_all[:, 0] >= shanks[shank_idx][0],
                self.chn_coords_all[:, 0] <= shanks[shank_idx][1],
            )
            self.chn_coords = self.chn_coords_all[shank_chns, :]
        else:
            self.chn_coords = self.chn_coords_all

        chn_depths = self.chn_coords[:, 1]

        data = {}
        values = [
            "spikes",
            "clusters",
            "channels",
            "rms_AP",
            "rms_LF",
            "rms_AP_main",
            "rms_LF_main",
            "psd_lf",
            "psd_lf_main",
        ]
        objects = [
            "spikes",
            "clusters",
            "channels",
            "ephysTimeRmsAP",
            "ephysTimeRmsLF",
            "ephysTimeRmsAPMain",
            "ephysTimeRmsLFMain",
            "ephysSpectralDensityLF",
            "ephysSpectralDensityLFMain",
        ]
        for v, o in zip(values, objects):
            try:
                data[v] = alfio.load_object(self.folder_path, o)
                data[v]["exists"] = True
                if "rms" in v:
                    data[v]["xaxis"] = "Time (s)"
            except alf.exceptions.ALFObjectNotFound:
                logger.warning(f"{v} data was not found, some plots will not display")
                data[v] = {"exists": False}

        data["rf_map"] = {"exists": False}
        data["pass_stim"] = {"exists": False}
        data["gabor"] = {"exists": False}

        shank_indices_file = self.folder_path / "spike_shank_indices.npy"
        if shank_indices_file.exists():
            data["spike_shanks"] = np.load(shank_indices_file)

        unit_shank_indices_file = self.folder_path / "unit_shank_indices.npy"
        if unit_shank_indices_file.exists():
            data["unit_shank_indices"] = np.load(unit_shank_indices_file)

        # Read in notes for this experiment see if file exists in directory
        if self.folder_path.joinpath("session_notes.txt").exists():
            with open(self.folder_path.joinpath("session_notes.txt"), "r") as f:
                sess_notes = f.read()
        else:
            sess_notes = "No notes for this session"

        return self.folder_path, chn_depths, sess_notes, data

    def get_allen_csv(self):
        allen_path = Path(Path(atlas.__file__).parent, "allen_structure_tree.csv")
        self.allen = alfio.load_file_content(allen_path)

        return self.allen

    def get_xyzpicks(self, folder_path: Path, shank_idx: int):
        # Read in local xyz_picks file
        # This file must exist, otherwise we don't know where probe was
        xyz_file_name = (
            "*xyz_picks_image_space.json"
            if self.n_shanks == 1
            else f"*xyz_picks_shank{float(shank_idx) + 1}_image_space.json"
        )
        xyz_file = sorted(folder_path.glob(xyz_file_name))

        if len(xyz_file) == 0:
            raise FileNotFoundError(
                f"Missing required probe trajectory file: {xyz_file_name}\n"
                f"Expected location: {folder_path}\n"
                "This file must contain probe insertion coordinates in image space."
            )
        elif len(xyz_file) > 1:
            raise ValueError(
                f"Multiple trajectory files found: {[f.name for f in xyz_file]}. "
                "Please ensure only one exists."
            )
        with open(xyz_file[0], "r") as f:
            user_picks = json.load(f)

        xyz_picks = np.array(user_picks["xyz_picks"]) / 1e6  # convert to meters

        return xyz_picks

    def get_slice_images(self, xyz_channels):
        # Load the CCF images
        """
        index = self.brain_atlas.bc.xyz2i(xyz_channels)[
            :, self.brain_atlas.xyz2dims
        ]
        """
        # --- Get the ccf slice in image space ---
        #
        # BrainCoordinates converts from XYZ world to IJK (spacing only) but
        # doesn't handle permutations (xyz2dim). This converts world
        # coordinates to image indices, and then permutes to match atlas image
        # orientation
        index = self.brain_atlas.bc.xyz2i(xyz_channels)[:, self.brain_atlas.xyz2dims]
        # Build a tilted slice by getting horizontal lines at each index,
        # N x image.shape[1]
        ccf_slice = _cut_slice_from_atlas_image(
            self.brain_atlas.image,
            index,  # type: ignore
        )
        label_slice = _cut_slice_from_atlas_image(
            self.brain_atlas.label,
            index,
            self.brain_atlas._label2rgb,  # type: ignore
        )
        x_dimno = self.brain_atlas.xyz2dims[0]
        # ML span of the slice in world coordinates
        width = [
            self.brain_atlas.bc.i2x(0),
            self.brain_atlas.bc.i2x(
                self.brain_atlas.image.shape[x_dimno]
            ),  # was 456: CCF 25 width
        ]
        # DV span of the slice in world coordinates
        height = [
            self.brain_atlas.bc.i2z(index[0, 2]),
            self.brain_atlas.bc.i2z(index[-1, 2]),
        ]

        logger.debug(f"Ccf slice: {ccf_slice.shape}")
        slice_data = {
            "ccf": ccf_slice,
            "label": label_slice,
            "scale": np.array(
                [
                    (width[-1] - width[0]) / ccf_slice.shape[0],
                    (height[-1] - height[0]) / ccf_slice.shape[1],
                ]
            ),
            "offset": np.array([width[0], height[0]]),
        }

        # --- Get the histology slices in image space ---
        for channel_name, hist_image in self.histology_images.items():
            hist_arr = sitk.GetArrayViewFromImage(hist_image)
            hist_slice = _cut_slice_from_atlas_image(
                hist_arr,
                index,  # type: ignore
            )
            slice_data[channel_name] = hist_slice

        return slice_data, None

    def get_region_description(self, region_idx):
        struct_idx = np.where(self.allen["id"] == region_idx)[0][0]
        # Haven't yet incorporated how to have region descriptions when not on Alyx
        # For now always have this as blank
        description = ""
        region_lookup = (
            self.allen["acronym"][struct_idx] + ": " + self.allen["name"][struct_idx]
        )

        if region_lookup == "void: void":
            region_lookup = "root: root"

        if not description:
            description = region_lookup + "\nNo information available for this region"
        else:
            description = region_lookup + "\n" + description

        return description, region_lookup

    def _transform_to_ccf(
        self,
        xyz_channels: NDArray,
        tx_chain_files: AntsTransformChainFiles,
        channel_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        channel_coords_mm = 1e-3 * xyz_channels  # convert to mm

        # Have to convert these to the physical space of the pipeline image first
        # We will do that go going through simpleITK indices for the paired images
        intensity_img = self.brain_atlas.intensity_sitk_image
        pipeline_img = self.brain_atlas.pipeline_sitk_image

        reg_pipeline_physical_points: list[list[float]] = []
        for point in channel_coords_mm:
            intensity_index = intensity_img.TransformPhysicalPointToContinuousIndex(
                tuple(point.tolist())
            )
            pipeline_point = pipeline_img.TransformContinuousIndexToPhysicalPoint(
                intensity_index
            )
            reg_pipeline_physical_points.append(list(pipeline_point))

        reg_pipeline_physical_points_array = np.array(reg_pipeline_physical_points)

        logger.info("Warping to ccf")
        this_probe_df = pandas.DataFrame(
            reg_pipeline_physical_points_array, columns=list("xyz")
        )

        logger.info("applying transforms ...")
        ccf_coordinates_dataframe: pandas.DataFrame = ants.apply_transforms_to_points(
            ANTS_DIMENSION,
            this_probe_df,
            tx_chain_files.as_list(),
            whichtoinvert=tx_chain_files.which_to_invert(),
        )
        logger.info("Done warping to ccf")

        ccf_channel_dict = {}
        pattern = re.compile(r"channel_(\d+)")

        # Collect indices and vectorize extraction of coordinates
        channel_indices = []
        channel_names = []
        for ch in channel_dict.keys():
            m = pattern.match(ch)
            if m:
                channel_indices.append(int(m.group(1)))
                channel_names.append(ch)

        # Slice once, ensure float64
        xyz_array = ccf_coordinates_dataframe.loc[
            channel_indices, ["x", "y", "z"]
        ].to_numpy(dtype=np.float64)

        for ch, (x, y, z) in zip(channel_names, xyz_array):
            info = channel_dict[ch]
            # channel_index = int(channel[channel.index('_')+1:])
            ccf_channel_dict[ch] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "axial": info["axial"],
                "lateral": info["lateral"],
                "brain_region_id": info["brain_region_id"],
                "brain_region": info["brain_region"],
            }
        return ccf_channel_dict

    def upload_data(
        self,
        feature: NDArray,
        track: NDArray,
        xyz_channels: NDArray,
        tx_chain_files: AntsTransformChainFiles,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[list[float]]], bool]:
        logger.info("Saving channel locations and previous alignments locally")
        logger.debug(f"Channels: {xyz_channels}")
        if self.brain_atlas is None:
            raise ValueError("Brain atlas not loaded, cannot save channel locations")
        regions: BrainRegions = self.brain_atlas.regions
        brain_regions = regions.get(self.brain_atlas.get_labels(xyz_channels))
        brain_regions["xyz"] = xyz_channels
        brain_regions["lateral"] = self.chn_coords[:, 0]
        brain_regions["axial"] = self.chn_coords[:, 1]

        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        channel_dict = self.create_channel_dict(brain_regions)
        self.channel_dict = channel_dict

        ccf_channel_dict = self._transform_to_ccf(
            xyz_channels, tx_chain_files, channel_dict
        )

        date = datetime.now().replace(microsecond=0).isoformat()
        self.alignments[date] = [feature.tolist(), track.tolist()]

        multi_shank = self.n_shanks > 1

        return channel_dict, self.alignments, ccf_channel_dict, multi_shank

    @staticmethod
    def create_channel_dict(brain_regions: Bunch) -> dict[str, dict[str, Any]]:
        """
        Create channel dictionary in form to write to json file
        :param brain_regions: information about location of electrode channels in brain atlas
        :type brain_regions: Bunch
        :return channel_dict:
        :type channel_dict: dictionary of dictionaries
        """
        channel_dict: dict[str, dict[str, Any]] = {}

        for i in range(brain_regions.id.size):
            channel = {
                "x": np.float64(brain_regions.xyz[i, 0] * 1e6),
                "y": np.float64(brain_regions.xyz[i, 1] * 1e6),
                "z": np.float64(brain_regions.xyz[i, 2] * 1e6),
                "axial": np.float64(brain_regions.axial[i]),
                "lateral": np.float64(brain_regions.lateral[i]),
                "brain_region_id": int(brain_regions.id[i]),
                "brain_region": brain_regions.acronym[i],
            }
            data = {"channel_" + str(i): channel}
            channel_dict.update(data)

        return channel_dict
