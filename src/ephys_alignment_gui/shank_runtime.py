"""Runtime state for one loaded shank/channel collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.ephys_data_service import ChannelCollectionView


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
    slice_data: Any = None
    fp_slice_data: Any = None
    _slice_track: NDArray | None = None

    def __post_init__(self) -> None:
        if self.chn_coords is None:
            self.chn_coords = self.collection.local_coordinates
        if self.chn_depths is None:
            self.chn_depths = self.collection.depths

    @property
    def shank_idx(self) -> int:
        """Zero-based shank index within the stream."""
        return self.collection.shank_idx

    def cached_slice(self, track: NDArray) -> tuple[Any, Any] | None:
        """Return cached ``(slice_data, fp_slice_data)`` for ``track``."""
        if self._slice_track is not None and np.array_equal(self._slice_track, track):
            return self.slice_data, self.fp_slice_data
        return None

    def set_slice(self, slice_data: Any, fp_slice_data: Any, track: NDArray) -> None:
        """Cache slice data built for ``track``."""
        self.slice_data = slice_data
        self.fp_slice_data = fp_slice_data
        self._slice_track = track
