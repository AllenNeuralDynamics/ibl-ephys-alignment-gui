from __future__ import annotations

import json
import logging
import re

# temporarily add this in for neuropixel course
# until figured out fix to problem on win32
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ants
import numpy as np
import one.alf.io as alfio
import pandas
import SimpleITK as sitk
from iblatlas import atlas
from iblatlas.regions import BrainRegions
from iblutil.util import Bunch
from numpy.typing import NDArray

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.anatomical_atlas import BrainAtlasAnatomical
from ephys_alignment_gui.ephys_data_service import (
    ChannelCollectionView,
    ChannelTable,
    EphysStreamData,
)
from ephys_alignment_gui.histology_data_service import HistologyRuntimeData
from ephys_alignment_gui.slice_service import SliceService

ssl._create_default_https_context = ssl._create_unverified_context
logger = logging.getLogger(__name__)

ANTS_DIMENSION = 3


@dataclass
class LoadDataLocal:
    """Legacy adapter for histology, slices, and output helpers.

    Selected mouse/probe/channel metadata is owned by ``AlignmentDataContext``.
    Ephys stream loading is owned by ``ProbeDataWorkflow``. This adapter keeps
    the remaining legacy plotting and save-output helpers working while those
    responsibilities are split into smaller services.
    """

    data_context: AlignmentDataContext
    brain_atlas: BrainAtlasAnatomical | None = None
    chn_coords: NDArray | None = None
    chn_coords_all: NDArray | None = None
    chn_contact_id_all: NDArray | None = None
    chn_shank_ind_all: NDArray | None = None
    ephys_stream: EphysStreamData | None = None
    channel_collection: ChannelCollectionView | None = None
    slice_service: SliceService = field(default_factory=SliceService)

    histology_images: dict[str, sitk.Image] = field(default_factory=dict)
    channel_dict: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mouse-root / probe selection
    # ------------------------------------------------------------------

    def _channel_table(self) -> ChannelTable | None:
        return self.data_context.channel_table

    def _n_shanks(self) -> int:
        return self.data_context.n_shanks

    def _clear_channel_cache(self) -> None:
        """Clear legacy channel adapter fields derived from a selected stream."""
        self.chn_coords = None
        self.chn_coords_all = None
        self.chn_contact_id_all = None
        self.chn_shank_ind_all = None
        self.ephys_stream = None
        self.channel_collection = None

    def _cache_channel_table_arrays(self, channel_table: ChannelTable) -> None:
        """Refresh legacy array views from canonical channel metadata."""
        self.chn_coords_all = channel_table.local_coordinates
        self.chn_contact_id_all = channel_table.contact_ids
        self.chn_shank_ind_all = channel_table.shank_indices

    def reset_for_mouse_root_selection(self, *, root_changed: bool) -> None:
        """Clear loader-side caches after the selected mouse root changes."""
        if root_changed:
            self.brain_atlas = None
            self.histology_images = {}
            if hasattr(self, "_lazy_channel_paths"):
                delattr(self, "_lazy_channel_paths")
        self._clear_channel_cache()

    def reset_for_probe_selection(self) -> None:
        """Clear loader-side stream caches after the selected probe changes."""
        self._clear_channel_cache()

    def set_histology_data(self, histology_data: HistologyRuntimeData) -> None:
        """Attach already-loaded atlas and histology runtime data."""
        self.brain_atlas = histology_data.brain_atlas
        self.histology_images = dict(histology_data.histology_images)
        self._lazy_channel_paths = dict(histology_data.lazy_channel_paths)

    @property
    def probe_id(self) -> str | None:
        """Shortcut for the current probe ID (if selected)."""
        probe = self.data_context.probe_info
        return probe.probe_id if probe is not None else None

    # ------------------------------------------------------------------
    # Channel info / ephys / atlas loading
    # ------------------------------------------------------------------

    def set_ephys_stream(self, stream: EphysStreamData) -> None:
        """Attach an already-loaded runtime stream to this loader adapter."""
        probe = self.data_context.probe_info
        if probe is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        if stream.recording_id != probe.recording_id:
            raise ValueError(
                "Cached stream recording does not match selected recording: "
                f"{stream.recording_id!r} != {probe.recording_id!r}"
            )
        if stream.ephys_collection != probe.ephys_collection:
            raise ValueError(
                "Cached stream collection does not match selected collection: "
                f"{stream.ephys_collection!r} != {probe.ephys_collection!r}"
            )
        self.ephys_stream = stream
        self._set_channel_table(stream.channel_table)
        self.channel_collection = None

    def set_channel_collection(self, collection: ChannelCollectionView) -> None:
        """Attach an already-selected runtime channel collection."""
        self.set_ephys_stream(collection.stream)
        self._set_channel_collection(collection)

    def _set_channel_table(self, channel_table: ChannelTable) -> None:
        """Update legacy channel-table adapter fields from a runtime model."""
        self.data_context.attach_channel_table(channel_table)
        self._cache_channel_table_arrays(channel_table)

    def _set_channel_collection(self, collection: ChannelCollectionView) -> None:
        self.channel_collection = collection
        self.chn_coords = collection.local_coordinates

    def set_channels_for_shank(self, shank_idx: int) -> NDArray:
        """Filter cached channel coordinates for selected shank. No disk I/O."""
        channel_table = self._channel_table()
        probe = self.data_context.probe_info
        if channel_table is None:
            raise RuntimeError("Channel info not loaded. Please select a probe first.")
        if probe is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        self._cache_channel_table_arrays(channel_table)

        if self.ephys_stream is not None:
            collection = self.ephys_stream.channel_collection(shank_idx)
        else:
            rows = channel_table.rows_for_shank(shank_idx)
            collection = ChannelCollectionView(
                stream=EphysStreamData(
                    recording_id=probe.recording_id,
                    ephys_collection=probe.ephys_collection,
                    ephys_dir=probe.ephys_dir or Path(),
                    channel_table=channel_table,
                    alf_data={},
                    session_notes="",
                    probe_id=probe.probe_id,
                    probe_name=probe.probe_name,
                    logical_probe=probe.logical_probe,
                ),
                shank_idx=shank_idx,
                rows=rows,
            )

        self._set_channel_collection(collection)

        return collection.depths

    def load_allen_csv(self):
        allen_path = Path(Path(atlas.__file__).parent, "allen_structure_tree.csv")
        self.allen = alfio.load_file_content(allen_path)
        return self.allen

    def get_track_annotations(self, shank_idx: int) -> NDArray[np.floating]:
        """Read xyz-picks (image space) for the current probe + shank."""
        probe = self.data_context.probe_info
        if probe is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        picks = probe.picks_for_shank(shank_idx)
        path = picks.image_space
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing probe trajectory file: {path}. "
                "This file must contain probe insertion coordinates in image space."
            )
        with open(path) as f:
            user_picks = json.load(f)

        # xyz_picks on disk are SPIM-native image-space RAS, in microns.
        # The rest of the GUI operates in the rotated canonical frame, so
        # rotate into canonical here. If no rotation is configured the helper
        # returns the input unchanged.
        track_annotations_ras_spim = np.array(user_picks["xyz_picks"]) / 1e6
        if self.brain_atlas is None:
            raise RuntimeError(
                "brain_atlas not yet loaded; attach histology data first"
            )
        return self.brain_atlas.rotate_to_canonical(track_annotations_ras_spim)

    # ------------------------------------------------------------------
    # Slice images
    # ------------------------------------------------------------------

    def get_slice_images(self, track_interpolation_ras):
        """Get atlas and histology slices for the current shank track."""
        if self.brain_atlas is None:
            raise RuntimeError(
                "brain_atlas not yet loaded; attach histology data first"
            )
        return (
            self.slice_service.build_slice_set(
                brain_atlas=self.brain_atlas,
                histology_images=self.histology_images,
                lazy_channel_paths=getattr(self, "_lazy_channel_paths", {}),
                track_interpolation_ras=track_interpolation_ras,
            ),
            None,
        )

    def get_perpendicular_slice_image(
        self,
        ephysalign,
        feature_ref: NDArray,
        track_ref: NDArray,
        feature_grid_m: NDArray,
        channel_name: str = "histology_registration",
        extent_m: float = 500e-6,
        n_perp_samples: int = 41,
        sigma_samples: float = 2.0,
    ) -> NDArray[np.float64]:
        """Build the perpendicular slice image for the current alignment.

        See :func:`ephys_alignment_gui.perpendicular_slice.build_perpendicular_slice`
        for the full contract. This wrapper handles channel resolution (lazy
        loading via :meth:`_load_and_slice_channel`) and sources the atlas
        volume that matches the blessed orientation used by the rest of the
        GUI.

        Returns
        -------
        NDArray (n_perp_samples, len(feature_grid_m))
            NaN for samples that fall outside the rotated histology volume.
        """
        if self.brain_atlas is None:
            raise RuntimeError("brain_atlas not yet loaded")

        return self.slice_service.build_perpendicular_slice_image(
            brain_atlas=self.brain_atlas,
            histology_images=self.histology_images,
            lazy_channel_paths=getattr(self, "_lazy_channel_paths", {}),
            ephysalign=ephysalign,
            feature_ref=feature_ref,
            track_ref=track_ref,
            feature_grid_m=feature_grid_m,
            channel_name=channel_name,
            extent_m=extent_m,
            n_perp_samples=n_perp_samples,
            sigma_samples=sigma_samples,
        )

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
        mouse_root = self.data_context.mouse_root
        if mouse_root is None or self.brain_atlas is None:
            raise RuntimeError(
                "Mouse root or brain atlas not loaded; cannot transform to CCF"
            )
        # Unrotate from the canonical (rotated) frame back to SPIM-native, then
        # use the pre-rotation (SPIM-native) sitk images for the index<->physical
        # math. The ANTs CCF chain was computed in SPIM-native coords and is
        # invalid for rotated inputs.
        channel_locations_ras_spim = self.brain_atlas.unrotate_to_spim_native(
            channel_locations_ras
        )
        histology_img = self.brain_atlas.intensity_sitk_image_spim_native
        pipeline_img = self.brain_atlas.pipeline_sitk_image_spim_native
        ras_to_lps = np.array([-1, -1, 1])
        # Convert IBL app world coordinates, RAS m, to ITK world coordinates, LPS mm
        channel_locations_lps_mm = 1e3 * ras_to_lps * channel_locations_ras_spim
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

        tx = mouse_root.transforms
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
        channel_locations_ras: NDArray,
        chn_coords: NDArray,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        bool,
    ]:
        """Compute the histology-space + CCF channel dicts for a save.

        IO-only: the alignment history itself is owned per-shank by
        :class:`~ephys_alignment_gui.shank_alignment.ShankAlignment` (the caller
        records the new alignment there and persists it), so this no longer
        keeps a resident ``alignments`` dict. ``chn_coords`` is passed in rather
        than read from loader scratch so a save is decoupled from loader state.
        """
        logger.info("Saving channel locations locally")
        logger.debug(f"Channels: {channel_locations_ras}")
        if self.brain_atlas is None:
            raise ValueError("Brain atlas not loaded, cannot save channel locations")
        regions: BrainRegions = self.brain_atlas.regions
        brain_regions = regions.get(self.brain_atlas.get_labels(channel_locations_ras))
        # Persist xyz in SPIM-native coords so external tools reading the
        # output don't need to know about the GUI's display rotation.
        brain_regions["xyz"] = self.brain_atlas.unrotate_to_spim_native(
            channel_locations_ras
        )
        brain_regions["lateral"] = chn_coords[:, 0]
        brain_regions["axial"] = chn_coords[:, 1]

        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        channel_dict = self.create_channel_dict(brain_regions)
        self.channel_dict = channel_dict

        ccf_channel_dict = self._transform_to_ccf(channel_locations_ras, channel_dict)

        multi_shank = self._n_shanks() > 1

        return channel_dict, ccf_channel_dict, multi_shank

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
