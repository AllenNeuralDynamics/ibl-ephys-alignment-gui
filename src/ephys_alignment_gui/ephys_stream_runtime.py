"""Runtime cache for one loaded ephys stream."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.ephys_data_service import (
    ChannelCollectionView,
    EphysStreamData,
)
from ephys_alignment_gui.plot_data import PlotData
from ephys_alignment_gui.plot_data_factory import PlotDataFactory

StreamKey = tuple[str, str]


@dataclass
class EphysStreamRuntime:
    """Own loaded ephys data and shank-level runtime caches for one stream."""

    stream: EphysStreamData
    plot_data_factory: PlotDataFactory
    current_shank_idx: int = 0
    plot_data_by_shank: dict[int, PlotData] = field(default_factory=dict)

    @property
    def stream_key(self) -> StreamKey:
        """Stable runtime key for this stream."""
        return self.stream.stream_key

    def collection_for_shank(self, shank_idx: int) -> ChannelCollectionView:
        """Return a channel-collection view for a 0-based shank index."""
        self.current_shank_idx = shank_idx
        return self.stream.channel_collection(shank_idx)

    def plot_data_for_shank(self, shank_idx: int) -> PlotData:
        """Return cached PlotData for one shank, building it on first use."""
        if shank_idx not in self.plot_data_by_shank:
            collection = self.collection_for_shank(shank_idx)
            self.plot_data_by_shank[shank_idx] = self.plot_data_factory.build(
                collection
            )
        else:
            self.current_shank_idx = shank_idx
        return self.plot_data_by_shank[shank_idx]

    def invalidate_plot_data(self, shank_idx: int | None = None) -> None:
        """Clear cached PlotData for one shank, or all shanks."""
        if shank_idx is None:
            self.plot_data_by_shank.clear()
            return
        self.plot_data_by_shank.pop(shank_idx, None)
