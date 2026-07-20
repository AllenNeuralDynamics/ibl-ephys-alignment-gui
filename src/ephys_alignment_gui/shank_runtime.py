"""Runtime state for one loaded shank/channel collection."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray

from ephys_alignment_gui.ephys_data_service import ChannelCollectionView
from ephys_alignment_gui.slice_runtime import SliceCacheEntry, SliceRuntime


@dataclass
class ShankRuntime:
    """Heavy/derived runtime state for one shank of a loaded ephys stream.

    This object owns loaded-data-derived state, not editable alignment history
    or Qt/pyqtgraph items. ``ShankAlignment`` may project these fields for
    compatibility while older plotting code is hollowed out.
    """

    collection: ChannelCollectionView

    # -- Channel geometry for this shank --
    chn_coords: NDArray | None = None
    chn_depths: NDArray | None = None

    # -- Track / channel locations in atlas (RAS) space --
    track_annotations_ras: NDArray | None = None
    track_annos_and_ends_ras: NDArray | None = None
    channel_locations_ras: NDArray | None = None
    tip_location_ras: NDArray | None = None

    # -- Alignment engine + derived region overlays for this shank --
    ephysalign: Any = None
    region_fp: Any = None
    region_label_fp: Any = None
    region_colour_fp: Any = None

    # -- Cached PlotData and atlas/histology slices for this shank --
    plotdata: Any = None
    slice_runtime: SliceRuntime = field(default_factory=SliceRuntime)

    def __post_init__(self) -> None:
        if self.chn_coords is None:
            self.chn_coords = self.collection.local_coordinates
        if self.chn_depths is None:
            self.chn_depths = self.collection.depths

    @property
    def shank_idx(self) -> int:
        """Zero-based shank index within the stream."""
        return self.collection.shank_idx

    @property
    def slice_data(self) -> Any:
        """Active coronal slice data, projected for legacy callers."""
        return self.slice_runtime.active_slice_data

    @slice_data.setter
    def slice_data(self, value: Any) -> None:
        self.slice_runtime.set_active_slice_data(value, self.fp_slice_data)

    @property
    def fp_slice_data(self) -> Any:
        """Active feature-space slice data, projected for legacy callers."""
        return self.slice_runtime.active_fp_slice_data

    @fp_slice_data.setter
    def fp_slice_data(self, value: Any) -> None:
        self.slice_runtime.set_active_slice_data(self.slice_data, value)

    def cached_slice(
        self,
        track: NDArray,
        alignment_key: Hashable | None = None,
    ) -> tuple[Any, Any] | None:
        """Return cached ``(slice_data, fp_slice_data)`` for ``track``."""
        entry = self.slice_runtime.cached_coronal_slice(
            alignment_key=self._slice_alignment_key(alignment_key),
            track_interpolation_ras=track,
        )
        if entry is None:
            return None
        return entry.slice_data, entry.fp_slice_data

    def set_slice(
        self,
        slice_data: Any,
        fp_slice_data: Any,
        track: NDArray,
        alignment_key: Hashable | None = None,
    ) -> SliceCacheEntry:
        """Cache slice data built for ``track``."""
        return self.slice_runtime.set_coronal_slice(
            alignment_key=self._slice_alignment_key(alignment_key),
            track_interpolation_ras=track,
            slice_data=slice_data,
            fp_slice_data=fp_slice_data,
        )

    def clear_slice_cache(self) -> None:
        """Clear cached anatomical slice data for this shank."""
        self.slice_runtime.clear()

    def _slice_alignment_key(self, alignment_key: Hashable | None) -> Hashable:
        if alignment_key is not None:
            return alignment_key
        return ("legacy", self.shank_idx)
