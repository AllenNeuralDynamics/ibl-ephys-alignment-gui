from __future__ import annotations

import json
import logging
import re

# temporarily add this in for neuropixel course
# until figured out fix to problem on win32
import ssl
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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

from ephys_alignment_gui.custom_atlas import (
    _BLESSED_DIRECTION,
    BrainAtlasAnatomical,
)
from ephys_alignment_gui.docdb import _default_doc_db_api_client, query_docdb_id

ssl._create_default_https_context = ssl._create_unverified_context
logger = logging.getLogger(__name__)

ANTS_DIMENSION = 3


class SmartSliceDict(dict):
    """Dict that loads images and computes slices on-demand, tracking trajectory per slice."""

    def __init__(
        self,
        eager_data,
        lazy_channel_names,
        trajectory_id,
        load_and_slice_callback,
    ) -> None:
        """
        Parameters:
        - eager_data: dict with pre-computed data (ccf, label, scale, offset, histology_registration)
        - lazy_channel_names: list of channel names for lazy loading
        - trajectory_id: unique ID for current trajectory
        - load_and_slice_callback: callable(channel_name) -> slice_array
        """
        super().__init__(eager_data)
        self._lazy_channel_names = set(lazy_channel_names)
        self._trajectory_id = trajectory_id
        self._load_and_slice_callback = load_and_slice_callback

        # Add lazy channel keys with None values (enables menu creation)
        for channel in lazy_channel_names:
            if channel not in self:
                super().__setitem__(channel, None)

    def __getitem__(self, key):
        """Load/compute slice on first access for current trajectory."""
        # Metadata keys always return directly
        if key in ["ccf", "label", "scale", "offset"]:
            return super().__getitem__(key)

        value = super().__getitem__(key)

        # Lazy channels: trigger load if None
        if key in self._lazy_channel_names and value is None:
            logger.info(f"Lazy loading and slicing channel: {key}")
            slice_data = self._load_and_slice_callback(key)
            super().__setitem__(key, slice_data)
            return slice_data

        return value


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


@dataclass(frozen=True)
class ImageSpacePaths:
    atlas_image_path: Path
    atlas_labels_path: Path
    pipeline_image_path: Path
    histology_image_path: Path
    other_channel_paths: list[Path] = field(default_factory=list)

    @classmethod
    def from_folder(cls, input_path: Path) -> ImageSpacePaths:
        def _glob_first(pattern: str) -> Path:
            return next(input_path.glob(pattern))

        atlas_image_path = _glob_first("ccf_in_*.nrrd")
        atlas_labels_path = _glob_first("labels_in_*.nrrd")
        pipeline_image_path = _glob_first("histology_registration_pipeline.nrrd")
        histology_image_path = _glob_first("histology_registration.nrrd")

        pattern = re.compile(r"^Ex_\d+_Em_\d+\.nrrd$")
        other_channel_paths: list[Path] = []
        for other_channel in input_path.iterdir():
            if pattern.match(other_channel.name):
                other_channel_paths.append(other_channel)
        return cls(
            atlas_image_path=atlas_image_path,
            atlas_labels_path=atlas_labels_path,
            pipeline_image_path=pipeline_image_path,
            histology_image_path=histology_image_path,
            other_channel_paths=other_channel_paths,
        )


@dataclass
class LoadDataLocal:
    image_space_paths: ImageSpacePaths | None = None
    brain_atlas: BrainAtlasAnatomical | None = None
    input_path: Path | None = None
    data_root: Path | None = None
    histology_path: Path | None = None
    we_are_in_code_ocean: bool = True
    chn_coords: NDArray | None = None
    chn_coords_all: NDArray | None = None
    sess_path: Path | None = None
    n_shanks: int = 0
    previous_directory: Path | None = None
    tx_chain_files: AntsTransformChainFiles | None = None

    histology_images: dict[str, sitk.Image] = field(default_factory=dict)
    channel_dict: dict[str, dict[str, Any]] = field(default_factory=dict)
    alignments: dict[str, list[list[float]]] = field(default_factory=dict)
    prev_align: list[str] = field(default_factory=lambda: ["original"])

    def load_previous_alignments_docdb(
        self,
        input_path: Path,
        shank_idx=0,
    ) -> tuple[dict[str, list[list[float]]], list[str]] | None:
        docdb_id = query_docdb_id(input_path.parent.stem)[0]
        quality_control = get_quality_control_by_id(
            _default_doc_db_api_client(), docdb_id
        )

        if quality_control is None:
            return None

        evaluations = quality_control.evaluations

        evaluation_name = f"{input_path.parent.stem}_{input_path.stem}_{shank_idx}"
        alignment_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation.name == f"Probe Alignment for {evaluation_name}"
        ]

        n_eval = len(alignment_evaluations)

        if n_eval == 0:
            logger.info(f"No alignment found in docdb for {evaluation_name}")
            return None

        logger.info(
            f"Found existing record for {evaluation_name}. Loading alignment now"
        )
        latest_alignment_evaluation = max(
            alignment_evaluations, key=lambda x: x.created
        )  # pull latest alignment evaluation
        curation_metric = latest_alignment_evaluation.metrics[0].value["curations"]
        alignments = json.loads(curation_metric[0])[
            "previous_alignments"
        ]  # load in the previous alignment
        prev_align = list(alignments.keys())
        prev_align = sorted(prev_align, reverse=True)
        prev_align.append("original")
        return alignments, prev_align

    def load_previous_alignments_local(
        self,
        input_path: Path | None = None,
        shank_idx: int = 0,
    ) -> tuple[dict[str, list[list[float]]], list[str]] | None:
        input_path = self._check_input_path_arg(input_path)
        suffix = f"_shank{shank_idx + 1}" if self.n_shanks > 1 else ""
        prev_align_filename = f"prev_alignments{suffix}.json"
        if input_path.joinpath(prev_align_filename).exists():
            with open(input_path.joinpath(prev_align_filename)) as f:
                alignments: dict[str, list[list[float]]] = json.load(f)
            prev_align = list(alignments.keys())
            prev_align = sorted(prev_align, reverse=True)
            prev_align.append("original")
        else:
            return None
        return alignments, prev_align

    def load_previous_alignments(
        self, shank_idx=0, input_path: Path | None = None, use_docdb=True
    ) -> None:
        input_path = self._check_input_path_arg(input_path)

        maybe_alignments = None
        load_local = not use_docdb
        if use_docdb:
            logger.debug("Using docdb to get previous alignments")
            try:
                maybe_alignments = self.load_previous_alignments_docdb(
                    input_path=input_path, shank_idx=shank_idx
                )
                if maybe_alignments is None:
                    load_local = True
            except ValueError as e:
                logger.warning(
                    f"Failed to load previous alignments from docdb with exception {e}. "
                    "Falling back to local file."
                )
                load_local = True
        if load_local:
            maybe_alignments = self.load_previous_alignments_local(
                input_path=input_path, shank_idx=shank_idx
            )
        if maybe_alignments is not None:
            alignments, prev_align = maybe_alignments
            self.alignments = alignments
            self.prev_align = prev_align

        return None

    def get_alignment_idx(self, idx: int) -> tuple[NDArray | None, NDArray | None]:
        """
        Find out the starting alignment
        """
        if len(self.prev_align) <= idx:
            return None, None
        alignment = self.prev_align[idx]
        if alignment == "original":
            feature = None
            track = None
        else:
            feature = np.array(self.alignments[alignment][0])
            track = np.array(self.alignments[alignment][1])

        return feature, track

    def _check_input_path_arg(self, input_path: Path | None) -> Path:
        if input_path is None:
            if self.input_path is None:
                raise RuntimeError(
                    "input_path must be provided if not set in the class"
                )
            input_path = self.input_path
        return input_path

    def load_channel_info(self, input_path: Path | None = None) -> None:
        """
        Load channel local coordinates from the alf files
        """
        input_path = self._check_input_path_arg(input_path)
        self.chn_coords_all = np.load(
            input_path.joinpath("channels.localCoordinates.npy")
        )
        chn_x = np.unique(self.chn_coords_all[:, 0])
        chn_x_diff = np.diff(chn_x)
        self.n_shanks = np.sum(chn_x_diff > 100) + 1

    def set_input_paths(self, input_path: Path, we_are_in_code_ocean: bool) -> None:
        if not we_are_in_code_ocean:
            raise RuntimeError("Only Code Ocean path resolution is supported currently")
        if self.input_path == input_path:
            logger.debug("Input path already set, skipping reset")
            return
        self.we_are_in_code_ocean = we_are_in_code_ocean
        self.input_path = input_path
        self.chn_coords_all = None
        data_root = input_path.parents[3]
        self.data_root = data_root
        maybe_tx_chain = self._find_transform_files()
        if maybe_tx_chain is None:
            raise FileNotFoundError(
                "No transform chain files found in input directory."
            )
        self.tx_chain_files = maybe_tx_chain
        histology_path: Path = self.input_path.parent.parent / "image_space_histology"
        if histology_path != self.histology_path:
            logger.debug("Atlas and histology path changed, invalidating cached data")
            self.brain_atlas = None
            self.histology_path = histology_path
            # Invalidate histology image cache
            self.histology_images = {}
            # Invalidate lazy loading metadata
            if hasattr(self, "_lazy_channel_paths"):
                delattr(self, "_lazy_channel_paths")
            if hasattr(self, "_lazy_channel_reorient"):
                delattr(self, "_lazy_channel_reorient")
            if hasattr(self, "_slice_index"):
                delattr(self, "_slice_index")
        self.image_space_paths = ImageSpacePaths.from_folder(histology_path)

    def get_shank_list(self) -> list[str] | None:
        """
        Generate shank list without setting n_shanks
        """
        if self.n_shanks == 1:
            shank_list = ["1/1"]
        elif self.n_shanks > 1:
            shank_list = [
                f"{iShank + 1}/{self.n_shanks}" for iShank in range(self.n_shanks)
            ]
        else:
            shank_list = None
        return shank_list

    def load_atlas_and_histology(self) -> None:
        logger.debug(f"Loading atlas and histology from {self.histology_path}")
        if self.image_space_paths is None:
            raise RuntimeError(
                "Image space paths not set, cannot load atlas and histology"
            )
        intensity_image = sitk.ReadImage(self.image_space_paths.atlas_image_path)
        label_image = sitk.ReadImage(self.image_space_paths.atlas_labels_path)
        if label_image.GetPixelID() is not sitk.sitkInt32:
            # This is a hack that I need to fix in the processing pipeline
            unq_annotations = np.load(
                "/data/allen_mouse_ccf_annotations_lateralized_compact/ccf_2017_annotation_25_lateralized_unique_vals.npz"
            )["unique_labels"]
            label_image = expand_compacted_image(label_image, unq_annotations)
        pipeline_image = sitk.ReadImage(self.image_space_paths.pipeline_image_path)
        self.brain_atlas = BrainAtlasAnatomical(
            intensity_img=intensity_image,
            label_img=label_image,
            pipeline_img=pipeline_image,
        )

        histology_image = sitk.ReadImage(self.image_space_paths.histology_image_path)
        dicom_orient_str = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                histology_image.GetDirection()
            )
        )
        if dicom_orient_str == _BLESSED_DIRECTION:
            reorient = False
        else:
            reorient = True
            histology_image = sitk.DICOMOrient(histology_image, _BLESSED_DIRECTION)
        self.histology_images["histology_registration"] = histology_image

        # Store metadata for lazy loading other channels
        self._lazy_channel_paths = {}
        self._lazy_channel_reorient = reorient
        for other_channel in self.image_space_paths.other_channel_paths:
            channel_name = other_channel.stem
            self._lazy_channel_paths[channel_name] = other_channel
        logger.debug(f"Setup lazy loading for {len(self._lazy_channel_paths)} channels")

    def set_channels_for_shank(self, shank_idx: int):
        """Filter cached channel coordinates for selected shank. No disk I/O."""
        if self.chn_coords_all is None:
            raise RuntimeError("Must call load_channel_info() first")
        chn_coords_all = self.chn_coords_all
        chn_x = np.unique(chn_coords_all[:, 0])

        if self.n_shanks > 1:
            shanks = {}
            for iShank in range(self.n_shanks):
                shanks[iShank] = [chn_x[iShank * 2], chn_x[(iShank * 2) + 1]]

            shank_chns = np.bitwise_and(
                chn_coords_all[:, 0] >= shanks[shank_idx][0],
                chn_coords_all[:, 0] <= shanks[shank_idx][1],
            )
            chn_coords = chn_coords_all[shank_chns, :]
        else:
            chn_coords = chn_coords_all
        self.chn_coords = chn_coords

        return chn_coords[:, 1]  # Return depths

    def get_ephys_data(self, shank_idx, input_path: Path | None = None):
        input_path = self._check_input_path_arg(input_path)
        if self.chn_coords_all is None:
            raise RuntimeError("Must call load_channel_info() first")
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
                data[v] = alfio.load_object(input_path, o)
                data[v]["exists"] = True
                if "rms" in v:
                    data[v]["xaxis"] = "Time (s)"
            except alf.exceptions.ALFObjectNotFound:
                logger.warning(f"{v} data was not found, some plots will not display")
                data[v] = {"exists": False}

        data["rf_map"] = {"exists": False}
        data["pass_stim"] = {"exists": False}
        data["gabor"] = {"exists": False}

        shank_indices_file = input_path / "spike_shank_indices.npy"
        if shank_indices_file.exists():
            data["spike_shanks"] = np.load(shank_indices_file)

        unit_shank_indices_file = input_path / "unit_shank_indices.npy"
        if unit_shank_indices_file.exists():
            data["unit_shank_indices"] = np.load(unit_shank_indices_file)

        # Read in notes for this experiment see if file exists in directory
        if input_path.joinpath("session_notes.txt").exists():
            with open(input_path.joinpath("session_notes.txt")) as f:
                sess_notes = f.read()
        else:
            sess_notes = "No notes for this session"

        return input_path, chn_depths, sess_notes, data

    def load_allen_csv(self):
        allen_path = Path(Path(atlas.__file__).parent, "allen_structure_tree.csv")
        self.allen = alfio.load_file_content(allen_path)

        return self.allen

    def get_track_annotations(
        self, shank_idx: int, input_path: Path | None = None
    ) -> NDArray[np.floating]:
        # Read in local xyz_picks file
        # This file must exist, otherwise we don't know where probe was
        input_path = self._check_input_path_arg(input_path)
        glob_suffix = "" if self.n_shanks == 1 else f"_shank{shank_idx + 1}"
        xyz_glob_str = f"*xyz_picks{glob_suffix}_image_space.json"
        xyz_file = sorted(input_path.glob(xyz_glob_str))

        if len(xyz_file) == 0:
            raise FileNotFoundError(
                f"Missing required probe trajectory file: {xyz_glob_str}\n"
                f"Expected location: {input_path}\n"
                "This file must contain probe insertion coordinates in image space."
            )
        elif len(xyz_file) > 1:
            raise ValueError(
                f"Multiple trajectory files found: {[f.name for f in xyz_file]}. "
                "Please ensure only one exists."
            )
        with open(xyz_file[0]) as f:
            user_picks = json.load(f)

        track_annotations_ras = (
            np.array(user_picks["xyz_picks"]) / 1e6
        )  # convert to meters

        return track_annotations_ras

    def get_slice_images(self, track_interpolation_ras):
        # Load the CCF images
        """
        index = self.brain_atlas.bc.xyz2i(channel_locations_ras)[
            :, self.brain_atlas.xyz2dims
        ]
        """
        # --- Get the ccf slice in image space ---
        #
        # BrainCoordinates converts from XYZ world to IJK (spacing only) but
        # doesn't handle permutations (xyz2dim). This converts world
        # coordinates to image indices, and then permutes to match atlas image
        # orientation
        index = self.brain_atlas.bc.xyz2i(track_interpolation_ras)[
            :, self.brain_atlas.xyz2dims
        ]
        # Store for lazy loading
        self._slice_index = index
        trajectory_id = id(track_interpolation_ras)  # Unique ID for this trajectory

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

        logger.debug(f"CCF slice: {ccf_slice.shape}")

        # Eager data: ccf, label, metadata, and default histology
        eager_data = {
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

        # --- Compute default histology slice (eager) ---
        if "histology_registration" in self.histology_images:
            hist_image = self.histology_images["histology_registration"]
            hist_arr = sitk.GetArrayViewFromImage(hist_image)
            hist_slice = _cut_slice_from_atlas_image(
                hist_arr,
                index,  # type: ignore
            )
            eager_data["histology_registration"] = hist_slice
            logger.debug("Computed eager slice for histology_registration")

        # --- Setup lazy loading for other channels ---
        lazy_channel_names = []
        if hasattr(self, "_lazy_channel_paths"):
            lazy_channel_names = list(self._lazy_channel_paths.keys())
            logger.debug(
                f"Setting up lazy loading for {len(lazy_channel_names)} channels"
            )

        slice_data = SmartSliceDict(
            eager_data=eager_data,
            lazy_channel_names=lazy_channel_names,
            trajectory_id=trajectory_id,
            load_and_slice_callback=self._load_and_slice_channel,
        )

        return slice_data, None

    def _load_and_slice_channel(self, channel_name: str) -> NDArray:
        """
        Load histology channel image (if needed) and compute slice for current trajectory.
        Called lazily when user selects channel from menu.

        Parameters:
        - channel_name: Channel to load/slice

        Returns:
        - slice_array: 2D slice along current trajectory
        """
        # Load image if not already cached
        if channel_name not in self.histology_images:
            if channel_name not in self._lazy_channel_paths:
                raise ValueError(f"Unknown channel: {channel_name}")

            channel_path = self._lazy_channel_paths[channel_name]
            logger.info(f"Loading channel image from disk: {channel_path.name}")
            channel_image = sitk.ReadImage(str(channel_path))

            if self._lazy_channel_reorient:
                channel_image = sitk.DICOMOrient(channel_image, _BLESSED_DIRECTION)

            # Cache image for future use
            self.histology_images[channel_name] = channel_image
            logger.debug(f"Cached {channel_name} in histology_images")
        else:
            logger.debug(f"Using cached image for {channel_name}")

        # Compute slice for current trajectory
        if not hasattr(self, "_slice_index") or self._slice_index is None:
            raise RuntimeError("Cannot compute slice: trajectory index not set")

        hist_image = self.histology_images[channel_name]
        hist_arr = sitk.GetArrayViewFromImage(hist_image)
        hist_slice = _cut_slice_from_atlas_image(
            hist_arr,
            self._slice_index,  # type: ignore
        )

        logger.debug(f"Computed slice for {channel_name}: shape {hist_slice.shape}")
        return hist_slice

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

    def _find_transform_files(self) -> AntsTransformChainFiles:
        logger.info("Loading transforms from stitched smartspim asset ...")
        subject_id = self.input_path.parent.parent.stem
        smartspim_template_affine_transform = tuple(
            self.data_root.glob(
                f"SmartSPIM_{subject_id}*/image_atlas_alignment/*/ls_to_template_SyN_0GenericAffine.mat"
            )
        )
        if not smartspim_template_affine_transform:
            # try legacy way
            smartspim_template_affine_transform = tuple(
                self.data_root.glob(
                    f"SmartSPIM_{subject_id}*/registration/ls_to_template_SyN_0GenericAffine.mat"
                )
            )
            if not smartspim_template_affine_transform:
                raise FileNotFoundError(
                    "No affine transform from spim to template. Check attached assets"
                )

        smartspim_template_warp_transform = tuple(
            self.data_root.glob(
                f"SmartSPIM_{subject_id}*/image_atlas_alignment/*/ls_to_template_SyN_1InverseWarp.nii.gz"
            )
        )
        if not smartspim_template_warp_transform:
            smartspim_template_warp_transform = tuple(
                self.data_root.glob(
                    f"SmartSPIM_{subject_id}*/registration/ls_to_template_SyN_1InverseWarp.nii.gz"
                )
            )
            if not smartspim_template_warp_transform:
                raise FileNotFoundError(
                    "No warp transform from spim to template. Check attached assets"
                )

        template_to_ccf_affine_transform = tuple(
            self.data_root.glob("spim_template_to_ccf/syn_0GenericAffine.mat")
        )
        if not template_to_ccf_affine_transform:
            raise FileNotFoundError(
                "No affine transform from template to ccf. Check attached assets"
            )

        template_to_ccf_warp_transform = tuple(
            self.data_root.glob("spim_template_to_ccf/syn_1InverseWarp.nii.gz")
        )
        if not template_to_ccf_warp_transform:
            raise FileNotFoundError(
                "No warp transform from template to ccf. Check attached assets"
            )

        return AntsTransformChainFiles(
            smartspim_template_affine_transform=smartspim_template_affine_transform[0],
            smartspim_template_warp_transform=smartspim_template_warp_transform[0],
            template_to_ccf_affine_transform=template_to_ccf_affine_transform[0],
            template_to_ccf_warp_transform=template_to_ccf_warp_transform[0],
        )

    def _transform_to_ccf(
        self,
        channel_locations_ras: NDArray,
        channel_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if self.tx_chain_files is None:
            raise RuntimeError("Transform chain files not set, cannot transform to CCF")
        channel_coords_mm = 1e-3 * channel_locations_ras  # convert to mm

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
            self.tx_chain_files.as_list(),
            whichtoinvert=self.tx_chain_files.which_to_invert(),
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

    def get_alignment_results(
        self,
        feature: NDArray,
        track: NDArray,
        channel_locations_ras: NDArray,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, list[list[float]]],
        dict[str, dict[str, Any]],
        bool,
    ]:
        logger.info("Saving channel locations and previous alignments locally")
        logger.debug(f"Channels: {channel_locations_ras}")
        if self.brain_atlas is None:
            raise ValueError("Brain atlas not loaded, cannot save channel locations")
        regions: BrainRegions = self.brain_atlas.regions
        brain_regions = regions.get(self.brain_atlas.get_labels(channel_locations_ras))
        brain_regions["xyz"] = channel_locations_ras
        brain_regions["lateral"] = self.chn_coords[:, 0]
        brain_regions["axial"] = self.chn_coords[:, 1]

        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        channel_dict = self.create_channel_dict(brain_regions)
        self.channel_dict = channel_dict

        ccf_channel_dict = self._transform_to_ccf(channel_locations_ras, channel_dict)

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
