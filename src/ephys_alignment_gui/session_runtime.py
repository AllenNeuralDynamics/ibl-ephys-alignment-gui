"""Runtime ownership for cached ephys streams."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.ephys_data_service import EphysStreamData
from ephys_alignment_gui.ephys_stream_runtime import EphysStreamRuntime, StreamKey
from ephys_alignment_gui.plot_data_factory import PlotDataFactory


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
        return self.stream_cache.get(stream_key)

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

    def activate_cached_stream(self, stream_key: StreamKey) -> EphysStreamRuntime:
        """Make a cached stream runtime active."""
        runtime = self.stream_cache[stream_key]
        self.active_stream_runtime = runtime
        self.current_stream_key = stream_key
        return runtime

    def cache_loaded_stream(self, runtime: EphysStreamRuntime) -> None:
        """Cache a freshly loaded stream runtime and mark it active."""
        self.stream_cache[runtime.stream_key] = runtime
        self.active_stream_runtime = runtime
        self.current_stream_key = runtime.stream_key

    def cache_loaded_stream_data(
        self,
        stream: EphysStreamData,
        plot_data_factory: PlotDataFactory,
        *,
        shank_idx: int,
    ) -> EphysStreamRuntime:
        """Build, cache, and initialize runtime ownership for a loaded stream."""
        runtime = EphysStreamRuntime(
            stream=stream,
            plot_data_factory=plot_data_factory,
        )
        runtime.shank_runtime_for(shank_idx)
        self.cache_loaded_stream(runtime)
        return runtime
