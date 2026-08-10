"""Runtime cache for one loaded ephys stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ephys_alignment_gui.plotting.payload_cache import EphysPlotPayloadCache
from ephys_alignment_gui.plotting.payload_cache_factory import (
    EphysPlotPayloadCacheFactory,
)
from ephys_alignment_gui.plotting.registry import PlotSpec, resolve_plot_payload
from ephys_alignment_gui.runtime.shank import ShankRuntime
from ephys_alignment_gui.services.ephys_data import (
    ChannelCollectionView,
    EphysStreamData,
)

StreamKey = tuple[str, str]


@dataclass
class EphysStreamRuntime:
    """Own loaded ephys data and shank-level runtime caches for one stream."""

    stream: EphysStreamData
    plot_payload_cache_factory: EphysPlotPayloadCacheFactory
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
        if shank_idx not in self.shank_runtime_by_idx:
            self.shank_runtime_by_idx[shank_idx] = ShankRuntime(
                self.stream.channel_collection(shank_idx)
            )
        self.current_shank_idx = shank_idx
        return self.shank_runtime_by_idx[shank_idx]

    def visited_shank_runtimes(self) -> dict[int, ShankRuntime]:
        """Runtime states for shanks initialized in this stream runtime."""
        return dict(sorted(self.shank_runtime_by_idx.items()))

    def plot_payload_cache_for_shank(self, shank_idx: int) -> EphysPlotPayloadCache:
        """Return cached plot payloads for one shank, building on first use."""
        runtime = self.shank_runtime_for(shank_idx)
        if runtime.plot_payload_cache is None:
            runtime.plot_payload_cache = self.plot_payload_cache_factory.build(
                runtime.collection
            )
        return runtime.plot_payload_cache

    def filtered_plot_payload_cache_for_shank(
        self,
        shank_idx: int,
        *,
        unit_filter: str,
    ) -> EphysPlotPayloadCache:
        """Return cached plot payloads for one shank with unit filtering applied."""
        payload_cache = self.plot_payload_cache_for_shank(shank_idx)
        payload_cache.filter_units(unit_filter)
        return payload_cache

    def plot_payload_for_shank(self, shank_idx: int, spec: PlotSpec | str) -> Any:
        """Return a plot payload for one shank by declarative plot-spec key."""
        return resolve_plot_payload(self.plot_payload_cache_for_shank(shank_idx), spec)

    def invalidate_plot_payload_cache(self, shank_idx: int | None = None) -> None:
        """Clear cached plot payloads for one shank, or all shanks."""
        if shank_idx is None:
            for runtime in self.shank_runtime_by_idx.values():
                runtime.plot_payload_cache = None
            return
        runtime = self.shank_runtime_by_idx.get(shank_idx)
        if runtime is not None:
            runtime.plot_payload_cache = None
