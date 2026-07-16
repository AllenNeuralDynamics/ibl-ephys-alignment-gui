"""Runtime cache for one loaded ephys stream."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.ephys_data_service import (
    ChannelCollectionView,
    EphysStreamData,
)
from ephys_alignment_gui.plot_data import PlotData
from ephys_alignment_gui.plot_data_factory import PlotDataFactory
from ephys_alignment_gui.shank_runtime import ShankRuntime

StreamKey = tuple[str, str]


@dataclass
class EphysStreamRuntime:
    """Own loaded ephys data and shank-level runtime caches for one stream."""

    stream: EphysStreamData
    plot_data_factory: PlotDataFactory
    current_shank_idx: int = 0
    shank_runtime_by_idx: dict[int, ShankRuntime] = field(default_factory=dict)

    @property
    def stream_key(self) -> StreamKey:
        """Stable runtime key for this stream."""
        return self.stream.stream_key

    def collection_for_shank(self, shank_idx: int) -> ChannelCollectionView:
        """Return a channel-collection view for a 0-based shank index."""
        return self.shank_runtime_for(shank_idx).collection

    def shank_runtime_for(self, shank_idx: int) -> ShankRuntime:
        """Return runtime state for a shank, creating it on first use."""
        self.current_shank_idx = shank_idx
        if shank_idx not in self.shank_runtime_by_idx:
            self.shank_runtime_by_idx[shank_idx] = ShankRuntime(
                self.stream.channel_collection(shank_idx)
            )
        return self.shank_runtime_by_idx[shank_idx]

    def visited_shank_runtimes(self) -> dict[int, ShankRuntime]:
        """Runtime states for shanks initialized in this stream runtime."""
        return dict(sorted(self.shank_runtime_by_idx.items()))

    def plot_data_for_shank(self, shank_idx: int) -> PlotData:
        """Return cached PlotData for one shank, building it on first use."""
        runtime = self.shank_runtime_for(shank_idx)
        if runtime.plotdata is None:
            runtime.plotdata = self.plot_data_factory.build(runtime.collection)
        return runtime.plotdata

    def invalidate_plot_data(self, shank_idx: int | None = None) -> None:
        """Clear cached PlotData for one shank, or all shanks."""
        if shank_idx is None:
            for runtime in self.shank_runtime_by_idx.values():
                runtime.plotdata = None
            return
        runtime = self.shank_runtime_by_idx.get(shank_idx)
        if runtime is not None:
            runtime.plotdata = None
