"""Runtime ownership for cached ephys streams."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.ephys_stream_runtime import EphysStreamRuntime, StreamKey


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

    def has_cached_stream(self, stream_key: StreamKey) -> bool:
        """Whether an ephys stream runtime is cached."""
        return stream_key in self.stream_cache

    def cached_stream(self, stream_key: StreamKey) -> EphysStreamRuntime | None:
        """Return a cached stream runtime, if present."""
        return self.stream_cache.get(stream_key)

    def pop_cached_stream(self, stream_key: StreamKey) -> EphysStreamRuntime | None:
        """Remove and return a cached stream runtime, if present."""
        return self.stream_cache.pop(stream_key, None)

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
