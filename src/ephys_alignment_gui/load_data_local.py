from __future__ import annotations

import json
import logging

# temporarily add this in for neuropixel course
# until figured out fix to problem on win32
import ssl
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import one.alf.io as alfio
import SimpleITK as sitk
from iblatlas import atlas
from numpy.typing import NDArray

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.anatomical_atlas import BrainAtlasAnatomical
from ephys_alignment_gui.ephys_data_service import (
    ChannelCollectionView,
    ChannelTable,
    EphysStreamData,
)
from ephys_alignment_gui.histology_data_service import (
    HistologyDataContext,
    HistologyRuntimeData,
)
from ephys_alignment_gui.slice_service import SliceService

ssl._create_default_https_context = ssl._create_unverified_context
logger = logging.getLogger(__name__)


@dataclass
class LoadDataLocal:
    """Legacy adapter for histology, slices, and output helpers.

    Selected mouse/probe/channel metadata is owned by ``AlignmentDataContext``.
    Ephys stream loading is owned by ``ProbeDataWorkflow``. This adapter keeps
    the remaining legacy plotting and save-output helpers working while those
    responsibilities are split into smaller services.
    """

    data_context: AlignmentDataContext
    histology_context: HistologyDataContext
    chn_coords: NDArray | None = None
    chn_coords_all: NDArray | None = None
    chn_contact_id_all: NDArray | None = None
    chn_shank_ind_all: NDArray | None = None
    ephys_stream: EphysStreamData | None = None
    channel_collection: ChannelCollectionView | None = None
    slice_service: SliceService = field(default_factory=SliceService)

    # ------------------------------------------------------------------
    # Mouse-root / probe selection
    # ------------------------------------------------------------------

    def _channel_table(self) -> ChannelTable | None:
        return self.data_context.channel_table

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
            self.histology_context.clear()
        self._clear_channel_cache()

    def reset_for_probe_selection(self) -> None:
        """Clear loader-side stream caches after the selected probe changes."""
        self._clear_channel_cache()

    def set_histology_data(self, histology_data: HistologyRuntimeData) -> None:
        """Attach already-loaded atlas and histology runtime data."""
        self.histology_context.set(histology_data)

    @property
    def brain_atlas(self) -> BrainAtlasAnatomical | None:
        """Loaded anatomical atlas, if available."""
        return self.histology_context.brain_atlas

    @property
    def histology_images(self) -> dict[str, sitk.Image]:
        """Loaded histology image channels."""
        return self.histology_context.histology_images

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
                lazy_channel_paths=self.histology_context.lazy_channel_paths,
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
            lazy_channel_paths=self.histology_context.lazy_channel_paths,
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
