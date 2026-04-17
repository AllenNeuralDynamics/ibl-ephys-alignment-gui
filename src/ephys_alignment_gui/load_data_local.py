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
from iblatlas import atlas
from iblatlas.regions import BrainRegions
from iblutil.util import Bunch
from numpy.typing import NDArray
from one import alf

from ephys_alignment_gui.anatomical_atlas import (
    _BLESSED_DIRECTION,
    BrainAtlasAnatomical,
)
from ephys_alignment_gui.datapackage_loader import (
    DataPackageError,
    MouseRoot,
    ProbeInfo,
    load_mouse_root,
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


@dataclass
class LoadDataLocal:
    """Loader driven by a preprocessed mouse-root directory.

    The entry-point is :meth:`set_mouse_root`, which reads ``datapackage.json``
    and surfaces the sessions and probes available. :meth:`select_probe` picks
    one for subsequent loading. All file paths come from the datapackage; the
    loader makes no assumptions about directory layout beyond that contract.
    """

    mouse_root: MouseRoot | None = None
    probe_info: ProbeInfo | None = None
    brain_atlas: BrainAtlasAnatomical | None = None
    chn_coords: NDArray | None = None
    chn_coords_all: NDArray | None = None
    n_shanks: int = 0

    histology_images: dict[str, sitk.Image] = field(default_factory=dict)
    channel_dict: dict[str, dict[str, Any]] = field(default_factory=dict)
    alignments: dict[str, list[list[float]]] = field(default_factory=dict)
    prev_align: list[str] = field(default_factory=lambda: ["original"])

    # ------------------------------------------------------------------
    # Mouse-root / probe selection
    # ------------------------------------------------------------------

    def set_mouse_root(self, mouse_root: Path) -> MouseRoot:
        """Load a mouse-root directory. Resets probe-specific state.

        Parameters
        ----------
        mouse_root : Path
            Directory containing ``datapackage.json``.

        Returns
        -------
        MouseRoot
            Resolved view of the mouse-root.
        """
        logger.info(f"set_mouse_root: {mouse_root}")
        mr = load_mouse_root(Path(mouse_root))
        if self.mouse_root is not None and self.mouse_root.root != mr.root:
            # Invalidate atlas/histology caches on mouse switch.
            self.brain_atlas = None
            self.histology_images = {}
            if hasattr(self, "_lazy_channel_paths"):
                delattr(self, "_lazy_channel_paths")
            if hasattr(self, "_lazy_channel_reorient"):
                delattr(self, "_lazy_channel_reorient")
            if hasattr(self, "_slice_index"):
                delattr(self, "_slice_index")
        self.mouse_root = mr
        self.probe_info = None
        self.chn_coords = None
        self.chn_coords_all = None
        self.n_shanks = 0
        return mr

    def list_sessions(self) -> list[str]:
        """Recording IDs available in the current mouse root."""
        if self.mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        return self.mouse_root.sessions

    def list_probes(self, recording_id: str) -> list[str]:
        """Probe names for a given recording in the current mouse root."""
        if self.mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        return self.mouse_root.probes_for_session(recording_id)

    def select_probe(self, recording_id: str, probe_name: str) -> ProbeInfo:
        """Select a probe for loading. Resets per-probe data caches."""
        if self.mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        probe = self.mouse_root.get_probe(recording_id, probe_name)
        logger.info(
            f"select_probe: recording={recording_id!r}, probe={probe_name!r}, "
            f"num_shanks={probe.num_shanks}, ephys_dir={probe.ephys_dir}"
        )
        self.probe_info = probe
        self.chn_coords = None
        self.chn_coords_all = None
        self.n_shanks = probe.num_shanks
        return probe

    @property
    def probe_id(self) -> str | None:
        """Shortcut for the current probe ID (if selected)."""
        return self.probe_info.probe_id if self.probe_info is not None else None

    # ------------------------------------------------------------------
    # Previous alignments
    # ------------------------------------------------------------------

    def load_previous_alignments_docdb(
        self,
        recording_id: str,
        probe_name: str,
        shank_idx: int = 0,
    ) -> tuple[dict[str, list[list[float]]], list[str]] | None:
        """Fetch alignment history from DocDB keyed by (recording, probe, shank)."""
        docdb_id = query_docdb_id(recording_id)[0]
        quality_control = get_quality_control_by_id(
            _default_doc_db_api_client(), docdb_id
        )

        if quality_control is None:
            return None

        evaluations = quality_control.evaluations

        evaluation_name = f"{recording_id}_{probe_name}_{shank_idx}"
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
        folder: Path,
        shank_idx: int = 0,
    ) -> tuple[dict[str, list[list[float]]], list[str]] | None:
        """Load ``prev_alignments.json`` (or shank variant) from *folder*."""
        suffix = f"_shank{shank_idx + 1}" if self.n_shanks > 1 else ""
        prev_align_filename = f"prev_alignments{suffix}.json"
        p = folder / prev_align_filename
        if not p.exists():
            return None
        with open(p) as f:
            alignments: dict[str, list[list[float]]] = json.load(f)
        prev_align = list(alignments.keys())
        prev_align = sorted(prev_align, reverse=True)
        prev_align.append("original")
        return alignments, prev_align

    def load_previous_alignments(
        self,
        folder: Path,
        shank_idx: int = 0,
        use_docdb: bool = True,
    ) -> bool:
        """Load previous alignments for the selected probe.

        Parameters
        ----------
        folder : Path
            Directory to check for ``prev_alignments.json`` on the local-file
            fallback path.
        shank_idx : int
            0-based shank index.
        use_docdb : bool
            Attempt DocDB first; fall back to local file if unavailable.
        """
        if self.probe_info is None:
            raise RuntimeError("No probe selected — call select_probe() first")

        maybe_alignments = None
        load_local = not use_docdb
        if use_docdb:
            logger.debug("Using docdb to get previous alignments")
            try:
                maybe_alignments = self.load_previous_alignments_docdb(
                    recording_id=self.probe_info.recording_id,
                    probe_name=self.probe_info.probe_name,
                    shank_idx=shank_idx,
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
                folder=folder, shank_idx=shank_idx
            )
        if maybe_alignments is None:
            return False
        alignments, prev_align = maybe_alignments
        self.alignments = alignments
        self.prev_align = prev_align
        return True

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

    # ------------------------------------------------------------------
    # Channel info / ephys / atlas loading
    # ------------------------------------------------------------------

    def load_channel_info(self) -> None:
        """Load channel local coordinates from the selected probe's ephys ALF."""
        if self.probe_info is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        if self.probe_info.ephys_dir is None:
            raise DataPackageError(
                f"Probe {self.probe_info.probe_name!r} has no ephys dir "
                "(preprocessing ran with skip_ephys=true)."
            )
        path = self.probe_info.ephys_dir / "channels.localCoordinates.npy"
        self.chn_coords_all = np.load(path)

        chn_x = np.unique(self.chn_coords_all[:, 0])
        chn_x_diff = np.diff(chn_x)
        geom_n_shanks = int(np.sum(chn_x_diff > 100) + 1)
        if geom_n_shanks != self.probe_info.num_shanks:
            logger.warning(
                "Channel geometry implies %d shanks but manifest says %d; "
                "trusting manifest.",
                geom_n_shanks,
                self.probe_info.num_shanks,
            )
        self.n_shanks = self.probe_info.num_shanks

    def get_shank_list(self) -> list[str] | None:
        """Build the shank-picker list for the current probe."""
        if self.n_shanks == 1:
            return ["1/1"]
        if self.n_shanks > 1:
            return [f"{i + 1}/{self.n_shanks}" for i in range(self.n_shanks)]
        return None

    def load_atlas_and_histology(self) -> None:
        """Load atlas + default histology channel from the mouse-root datapackage."""
        if self.mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        hist = self.mouse_root.histology
        logger.debug(f"Loading atlas and histology from {hist.registration.parent}")

        intensity_image = sitk.ReadImage(str(hist.ccf_template))
        label_image = sitk.ReadImage(str(hist.labels))
        pipeline_image = sitk.ReadImage(str(hist.registration_pipeline))
        self.brain_atlas = BrainAtlasAnatomical(
            intensity_img=intensity_image,
            label_img=label_image,
            pipeline_img=pipeline_image,
        )

        histology_image = sitk.ReadImage(str(hist.registration))
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

        # Store metadata for lazy loading other channels.
        self._lazy_channel_paths = dict(hist.additional_channels)
        self._lazy_channel_reorient = reorient
        logger.debug(
            f"Setup lazy loading for {len(self._lazy_channel_paths)} channels"
        )

    def set_channels_for_shank(self, shank_idx: int):
        """Filter cached channel coordinates for selected shank. No disk I/O."""
        if self.chn_coords_all is None:
            raise RuntimeError("Must call load_channel_info() first")
        chn_coords_all = self.chn_coords_all
        chn_x = np.unique(chn_coords_all[:, 0])

        if self.n_shanks > 1:
            shanks = {}
            for i in range(self.n_shanks):
                shanks[i] = [chn_x[i * 2], chn_x[(i * 2) + 1]]
            mask = np.bitwise_and(
                chn_coords_all[:, 0] >= shanks[shank_idx][0],
                chn_coords_all[:, 0] <= shanks[shank_idx][1],
            )
            chn_coords = chn_coords_all[mask, :]
        else:
            chn_coords = chn_coords_all
        self.chn_coords = chn_coords

        return chn_coords[:, 1]  # Return depths

    def get_ephys_data(self, shank_idx: int):
        """Load ephys ALF for the current probe + shank.

        Returns
        -------
        tuple
            ``(ephys_dir, chn_depths, sess_notes, data)``. The ``ephys_dir`` is
            what downstream plot code stores as ``probe_path`` (it contains
            ``band_corr/`` etc.).
        """
        if self.probe_info is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        if self.probe_info.ephys_dir is None:
            raise DataPackageError(
                f"Probe {self.probe_info.probe_name!r} has no ephys dir"
            )
        if self.chn_coords_all is None:
            raise RuntimeError("Must call load_channel_info() first")

        ephys_dir = self.probe_info.ephys_dir
        logger.info(f"get_ephys_data: loading from {ephys_dir}, shank_idx={shank_idx}")

        chn_x = np.unique(self.chn_coords_all[:, 0])
        if self.n_shanks > 1:
            shanks = {}
            for i in range(self.n_shanks):
                shanks[i] = [chn_x[i * 2], chn_x[(i * 2) + 1]]
            mask = np.bitwise_and(
                self.chn_coords_all[:, 0] >= shanks[shank_idx][0],
                self.chn_coords_all[:, 0] <= shanks[shank_idx][1],
            )
            self.chn_coords = self.chn_coords_all[mask, :]
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
                data[v] = alfio.load_object(ephys_dir, o)
                data[v]["exists"] = True
                if "rms" in v:
                    data[v]["xaxis"] = "Time (s)"
            except alf.exceptions.ALFObjectNotFound:
                logger.warning(f"{v} data was not found, some plots will not display")
                data[v] = {"exists": False}

        data["rf_map"] = {"exists": False}
        data["pass_stim"] = {"exists": False}
        data["gabor"] = {"exists": False}

        shank_indices_file = ephys_dir / "spike_shank_indices.npy"
        if shank_indices_file.exists():
            data["spike_shanks"] = np.load(shank_indices_file)

        unit_shank_indices_file = ephys_dir / "unit_shank_indices.npy"
        if unit_shank_indices_file.exists():
            data["unit_shank_indices"] = np.load(unit_shank_indices_file)

        notes_file = ephys_dir / "session_notes.txt"
        if notes_file.exists():
            sess_notes = notes_file.read_text()
        else:
            sess_notes = "No notes for this session"

        return ephys_dir, chn_depths, sess_notes, data

    def load_allen_csv(self):
        allen_path = Path(Path(atlas.__file__).parent, "allen_structure_tree.csv")
        self.allen = alfio.load_file_content(allen_path)
        return self.allen

    def get_track_annotations(self, shank_idx: int) -> NDArray[np.floating]:
        """Read xyz-picks (image space) for the current probe + shank."""
        if self.probe_info is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        picks = self.probe_info.picks_for_shank(shank_idx)
        path = picks.image_space
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing probe trajectory file: {path}. "
                "This file must contain probe insertion coordinates in image space."
            )
        with open(path) as f:
            user_picks = json.load(f)
        return np.array(user_picks["xyz_picks"]) / 1e6  # µm -> m

    # ------------------------------------------------------------------
    # Slice images
    # ------------------------------------------------------------------

    def get_slice_images(self, track_interpolation_ras):
        # Load the CCF images
        """
        Get slice images
        """
        # --- Get the ccf slice in image space ---
        #
        # BrainCoordinates converts from XYZ world to IJK (spacing only) but
        # doesn't handle permutations (xyz2dim). This converts world
        # coordinates to image indices, and then permutes to match atlas image
        # orientation
        index = self.brain_atlas.physical_points_to_indices(
            track_interpolation_ras, round=True
        )
        # Store for lazy loading
        self._slice_index = index
        trajectory_id = id(track_interpolation_ras)

        # Build a tilted slice by getting horizontal lines at each index.
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
        width = [
            self.brain_atlas.bc.i2x(0),
            self.brain_atlas.bc.i2x(self.brain_atlas.image.shape[x_dimno]),
        ]
        height = [
            self.brain_atlas.bc.i2z(index[0, 2]),
            self.brain_atlas.bc.i2z(index[-1, 2]),
        ]

        logger.debug(f"CCF slice: {ccf_slice.shape}")

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

        if "histology_registration" in self.histology_images:
            hist_image = self.histology_images["histology_registration"]
            hist_arr = sitk.GetArrayViewFromImage(hist_image)
            hist_slice = _cut_slice_from_atlas_image(
                hist_arr,
                index,  # type: ignore
            )
            eager_data["histology_registration"] = hist_slice
            logger.debug("Computed eager slice for histology_registration")

        lazy_channel_names: list[str] = []
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
        """Load histology channel (if needed) and slice for the current trajectory."""
        if channel_name not in self.histology_images:
            if channel_name not in self._lazy_channel_paths:
                raise ValueError(f"Unknown channel: {channel_name}")
            channel_path = self._lazy_channel_paths[channel_name]
            logger.info(f"Loading channel image from disk: {channel_path.name}")
            channel_image = sitk.ReadImage(str(channel_path))
            if self._lazy_channel_reorient:
                channel_image = sitk.DICOMOrient(channel_image, _BLESSED_DIRECTION)
            self.histology_images[channel_name] = channel_image
            logger.debug(f"Cached {channel_name} in histology_images")
        else:
            logger.debug(f"Using cached image for {channel_name}")

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

    # ------------------------------------------------------------------
    # CCF transform + alignment result export
    # ------------------------------------------------------------------

    def _transform_to_ccf(
        self,
        channel_locations_ras: NDArray,
        channel_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if self.mouse_root is None or self.brain_atlas is None:
            raise RuntimeError(
                "Mouse root or brain atlas not loaded; cannot transform to CCF"
            )
        # Convert from the alignment-GUI physical space (SimpleITK-rotated
        # intensity image) back to the pipeline image's physical space.
        histology_img = self.brain_atlas.intensity_sitk_image
        pipeline_img = self.brain_atlas.pipeline_sitk_image
        ras_to_lps = np.array([-1, -1, 1])
        channel_locations_lps_mm = 1e3 * ras_to_lps * channel_locations_ras
        reg_pipeline_physical_points: list[list[float]] = []
        for point in channel_locations_lps_mm:
            index = histology_img.TransformPhysicalPointToContinuousIndex(point)
            pipeline_point = pipeline_img.TransformContinuousIndexToPhysicalPoint(index)
            reg_pipeline_physical_points.append(list(pipeline_point))

        reg_pipeline_physical_points_array = np.array(reg_pipeline_physical_points)

        logger.info("Warping to ccf")
        this_probe_df = pandas.DataFrame(
            reg_pipeline_physical_points_array, columns=list("xyz")
        )

        tx = self.mouse_root.transforms
        tx_list = [
            str(tx.image_to_template_affine),
            str(tx.image_to_template_warp),
            str(tx.template_to_ccf_affine),
            str(tx.template_to_ccf_warp),
        ]
        invert_list = [True, False, True, False]

        logger.info("applying transforms ...")
        ccf_coordinates_dataframe: pandas.DataFrame = ants.apply_transforms_to_points(
            ANTS_DIMENSION,
            this_probe_df,
            tx_list,
            whichtoinvert=invert_list,
        )
        logger.info("Done warping to ccf")

        ccf_channel_dict: dict[str, dict[str, Any]] = {}
        pattern = re.compile(r"channel_(\d+)")

        channel_indices = []
        channel_names = []
        for ch in channel_dict.keys():
            m = pattern.match(ch)
            if m:
                channel_indices.append(int(m.group(1)))
                channel_names.append(ch)

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
        if self.chn_coords is None:
            raise RuntimeError("Must call set_channels_for_shank() first")
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
