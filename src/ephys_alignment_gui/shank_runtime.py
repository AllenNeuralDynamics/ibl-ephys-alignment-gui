"""Runtime state for one loaded shank/channel collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray

from ephys_alignment_gui.ephys_data_service import ChannelCollectionView
from ephys_alignment_gui.slice_runtime import SliceRuntime


@dataclass
class ShankRuntime:
    """Heavy/derived runtime state for one shank of a loaded ephys stream.

    This object owns loaded-data-derived state, not editable alignment history
    or Qt/pyqtgraph items.
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
    nearby_boundaries: Any = None

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
