from __future__ import annotations

import json
import logging
import os
import re

# temporarily add this in for neuropixel course
# until figured out fix to problem on win32
import ssl
from datetime import datetime
from pathlib import Path
from typing import Callable

import iblatlas.atlas as atlas
import numpy as np
import one.alf.io as alfio
import SimpleITK as sitk
from aind_data_access_api.helpers.data_schema import get_quality_control_by_id
from aind_registration_utils.annotations import expand_compacted_image
from numpy.typing import NDArray
from one import alf

from ephys_alignment_gui.docdb import docdb_api_client, query_docdb_id

from .custom_atlas import BrainAtlasAnatomical

ssl._create_default_https_context = ssl._create_unverified_context
logger = logging.getLogger("ibllib")


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


class LoadDataLocal:
    def __init__(self):
        self.brain_atlas = None
        self.franklin_atlas = None
        self.folder_path = None
        self.atlas_path = Path(__file__).parents[2].joinpath("atlas_data")
        self.histology_path = None
        self.chn_coords = None
        self.chn_coords_all = None
        self.sess_path = None
        self.shank_idx = 0
        self.n_shanks = 1
        self.data_root = None
        self.output_directory = None
        self.previous_directory = None
        self.histology_atlases: dict[str, BrainAtlasAnatomical] = {}

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

        print("Checking docdb for existing records")
        self.shank_idx = shank_idx

        quality_control = None
        try:
            docdb_id = query_docdb_id(folder_path.parent.stem)[0]
            quality_control = get_quality_control_by_id(docdb_api_client, docdb_id)
        except ValueError as e:
            print(
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
                print(
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
                print(f"No alignment found in docdb for {evaluation_name}")
                self.alignments = []
                self.prev_align = ["original"]
        else:
            # If previous alignment json file exists, read in previous alignments
            prev_align_filename = (
                "prev_alignments.json"
                if self.n_shanks == 1
                else f"prev_alignments_shank{self.shank_idx + 1}.json"
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
                self.alignments = []
                self.prev_align = ["original"]

        return self.prev_align

    def get_starting_alignment(self, idx, shank_idx=0, folder_path: Path | None = None):
        """
        Find out the starting alignmnet
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

    def get_data(self, reload_data: bool = True):
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
                histology_image = sitk.ReadImage(self.histology_image_path[0])
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

                self.histology_atlases["histology_registration"] = BrainAtlasAnatomical(
                    intensity_img=histology_image,
                    label_img=sitk.Image(label_image),
                    pipeline_img=sitk.Image(pipeline_image),
                )
                pattern = re.compile(r"^Ex_\d+_Em_\d+\.nrrd$")
                for other_channel in self.histology_path.iterdir():
                    if pattern.match(other_channel.name):
                        channel_name = other_channel.stem
                        if channel_name not in self.histology_atlases:
                            channel_image = sitk.ReadImage(str(other_channel))
                            self.histology_atlases[channel_name] = BrainAtlasAnatomical(
                                intensity_img=channel_image,
                                label_img=sitk.Image(label_image),
                                pipeline_img=sitk.Image(pipeline_image),
                            )

        chn_x = np.unique(self.chn_coords_all[:, 0])
        if self.n_shanks > 1:
            shanks = {}
            for iShank in range(self.n_shanks):
                shanks[iShank] = [chn_x[iShank * 2], chn_x[(iShank * 2) + 1]]

            shank_chns = np.bitwise_and(
                self.chn_coords_all[:, 0] >= shanks[self.shank_idx][0],
                self.chn_coords_all[:, 0] <= shanks[self.shank_idx][1],
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

        print("Ccf slice", ccf_slice.shape)
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
        for channel_name, hist_atlas in self.histology_atlases.items():
            hist_slice = _cut_slice_from_atlas_image(
                hist_atlas.image,
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

    def upload_data(self, feature, track, xyz_channels, shank_idx):
        print("Channels", xyz_channels)
        region_ids = []
        index = np.round(xyz_channels).astype(np.int64)
        index = index[
            (index[:, 0] < self.brain_atlas.image.shape[0])
            & (index[:, 1] < self.brain_atlas.image.shape[1])
            & (index[:, 2] < self.brain_atlas.image.shape[2])
        ]

        for coord in index:
            region_ids.append(self.brain_atlas.label[coord[0], coord[1], coord[2]])

        brain_regions = self.brain_atlas.regions.get(region_ids)
        brain_regions["xyz"] = xyz_channels
        brain_regions["lateral"] = self.chn_coords[:, 0]
        brain_regions["axial"] = self.chn_coords[:, 1]

        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        channel_dict = self.create_channel_dict(brain_regions)
        self.channel_dict = channel_dict
        bregma = atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"].tolist()
        origin = {"origin": {"bregma": bregma}}
        channel_dict.update(origin)
        # Save the channel locations
        chan_loc_filename = (
            "channel_locations.json"
            if self.n_shanks == 1
            else f"channel_locations_shank{shank_idx}.json"
        )

        os.makedirs(self.output_directory, exist_ok=True)
        with open(self.output_directory.joinpath(chan_loc_filename), "w") as f:
            json.dump(channel_dict, f, indent=2, separators=(",", ": "))
        original_json = self.alignments
        date = datetime.now().replace(microsecond=0).isoformat()
        data = {date: [feature.tolist(), track.tolist()]}
        if original_json:
            original_json.update(data)
        else:
            original_json = data
        # Save the new alignment
        prev_align_filename = (
            "prev_alignments.json"
            if self.n_shanks == 1
            else f"prev_alignments_shank{self.shank_idx + 1}.json"
        )
        with open(self.output_directory.joinpath(prev_align_filename), "w") as f:
            json.dump(original_json, f, indent=2, separators=(",", ": "))

    @staticmethod
    def create_channel_dict(brain_regions):
        """
        Create channel dictionary in form to write to json file
        :param brain_regions: information about location of electrode channels in brain atlas
        :type brain_regions: Bunch
        :return channel_dict:
        :type channel_dict: dictionary of dictionaries
        """
        channel_dict = {}

        for i in np.arange(brain_regions.id.size):
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
