"""Runtime ownership for cached ephys streams."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.plotting.payload_cache_factory import (
    EphysPlotPayloadCacheFactory,
)
from ephys_alignment_gui.runtime.ephys_stream import EphysStreamRuntime, StreamKey
from ephys_alignment_gui.services.ephys_data import EphysStreamData


@dataclass(frozen=True)
class LoadDataTarget:
    """Selected stream/shank target for a load-data request."""

    stream_key: StreamKey | None
    shank_idx: int


@dataclass(frozen=True)
class LoadDataAlreadyActive:
    """The requested stream/shank is already active and loaded."""

    target: LoadDataTarget


@dataclass(frozen=True)
class LoadDataCachedStreamAvailable:
    """The requested stream is cached and can be activated without heavy IO."""

    target: LoadDataTarget
    cached_shank_idx: int


@dataclass(frozen=True)
class LoadDataFreshRequired:
    """The requested stream needs a fresh heavy load."""

    target: LoadDataTarget


LoadDataPlan = (
    LoadDataAlreadyActive | LoadDataCachedStreamAvailable | LoadDataFreshRequired
)


@dataclass
class SessionRuntime:
    """Owns active and cached ephys stream runtimes.

    This object is part of the Qt-free workspace. Desktop view-session objects
    own pyqtgraph lifetimes and are managed by the desktop shell.
    """

    active_stream_runtime: EphysStreamRuntime | None = None
    stream_cache: dict[StreamKey, EphysStreamRuntime] = field(default_factory=dict)
    current_stream_key: StreamKey | None = None
    max_cached_streams: int | None = 3

    def __post_init__(self) -> None:
        if self.max_cached_streams is not None and self.max_cached_streams < 1:
            raise ValueError("max_cached_streams must be positive or None")
        self._enforce_stream_cache_limit()

    def clear_active_stream(self) -> None:
        """Clear the active stream selection without evicting cached streams."""
        self.active_stream_runtime = None
        self.current_stream_key = None

    def clear_stream_cache(self) -> None:
        """Clear all cached stream runtimes and active stream selection."""
        self.stream_cache.clear()
        self.clear_active_stream()

    def cached_stream(self, stream_key: StreamKey) -> EphysStreamRuntime | None:
        """Return a cached stream runtime, if present."""
        runtime = self.stream_cache.get(stream_key)
        if runtime is not None:
            self._touch_cached_stream(stream_key)
        return runtime

    def is_active_stream_shank(self, stream_key: StreamKey, shank_idx: int) -> bool:
        """Whether the active runtime matches one stream/shank target."""
        stream_runtime = self.active_stream_runtime
        return (
            stream_runtime is not None
            and self.current_stream_key == stream_key
            and stream_runtime.stream_key == stream_key
            and stream_runtime.current_shank_idx == shank_idx
        )

    def plan_load_data(
        self,
        target: LoadDataTarget,
        *,
        data_loaded: bool,
    ) -> LoadDataPlan:
        """Return the cache action for one load-data target."""
        if target.stream_key is not None:
            if data_loaded and self.is_active_stream_shank(
                target.stream_key,
                target.shank_idx,
            ):
                return LoadDataAlreadyActive(target)
            cached = self.cached_stream(target.stream_key)
            if cached is not None:
                return LoadDataCachedStreamAvailable(
                    target=target,
                    cached_shank_idx=cached.current_shank_idx,
                )
        return LoadDataFreshRequired(target)

    def pop_cached_stream(self, stream_key: StreamKey) -> EphysStreamRuntime | None:
        """Remove and return a cached stream runtime, if present."""
        return self.stream_cache.pop(stream_key, None)

    def prepare_fresh_load(
        self,
        stream_key: StreamKey | None,
    ) -> EphysStreamRuntime | None:
        """Discard stale active/cache state before rebuilding one stream."""
        stale_runtime = None
        if stream_key is not None:
            stale_runtime = self.pop_cached_stream(stream_key)
        self.clear_active_stream()
        return stale_runtime

    def activate_cached_stream_for_shank(
        self,
        stream_key: StreamKey,
        *,
        shank_idx: int,
    ) -> EphysStreamRuntime:
        """Activate a cached stream after initializing the requested shank."""
        runtime = self.stream_cache[stream_key]
        runtime.shank_runtime_for(shank_idx)
        self.active_stream_runtime = runtime
        self.current_stream_key = stream_key
        self._touch_cached_stream(stream_key)
        return runtime

    def cache_loaded_stream(
        self,
        runtime: EphysStreamRuntime,
        *,
        activate: bool = True,
    ) -> None:
        """Cache a freshly loaded stream runtime and optionally mark it active."""
        self.stream_cache[runtime.stream_key] = runtime
        if activate:
            self.active_stream_runtime = runtime
            self.current_stream_key = runtime.stream_key
        self._touch_cached_stream(runtime.stream_key)
        self._enforce_stream_cache_limit()

    def activate_stream_runtime(
        self,
        runtime: EphysStreamRuntime,
        *,
        shank_idx: int,
    ) -> EphysStreamRuntime:
        """Activate a cached stream runtime after initializing one shank."""
        runtime.shank_runtime_for(shank_idx)
        self.stream_cache[runtime.stream_key] = runtime
        self.active_stream_runtime = runtime
        self.current_stream_key = runtime.stream_key
        self._touch_cached_stream(runtime.stream_key)
        self._enforce_stream_cache_limit()
        return runtime

    def cache_loaded_stream_data(
        self,
        stream: EphysStreamData,
        plot_payload_cache_factory: EphysPlotPayloadCacheFactory,
        *,
        shank_idx: int,
        activate: bool = True,
    ) -> EphysStreamRuntime:
        """Build, cache, and initialize runtime ownership for a loaded stream."""
        runtime = EphysStreamRuntime(
            stream=stream,
            plot_payload_cache_factory=plot_payload_cache_factory,
        )
        runtime.shank_runtime_for(shank_idx)
        self.cache_loaded_stream(runtime, activate=activate)
        return runtime

    def _touch_cached_stream(self, stream_key: StreamKey) -> None:
        """Mark a cached stream as recently used while preserving its object."""
        try:
            runtime = self.stream_cache.pop(stream_key)
        except KeyError:
            return
        self.stream_cache[stream_key] = runtime

    def _enforce_stream_cache_limit(self) -> None:
        """Evict least-recently-used inactive streams until within budget."""
        if self.max_cached_streams is None:
            return

        while len(self.stream_cache) > self.max_cached_streams:
            evict_key = self._oldest_evictable_stream_key()
            if evict_key is None:
                return
            evicted = self.stream_cache.pop(evict_key, None)
            if evicted is not None:
                evicted.clear_derived_caches()

    def _oldest_evictable_stream_key(self) -> StreamKey | None:
        """Return the oldest cached stream that is not the active stream."""
        for stream_key in self.stream_cache:
            if stream_key != self.current_stream_key:
                return stream_key
        return None
