"""Runtime ownership for active view sessions and cached ephys streams."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.ephys_stream_runtime import EphysStreamRuntime, StreamKey
from ephys_alignment_gui.probe_session import ProbeSession


@dataclass
class SessionRuntime:
    """Owns active ProbeSession and stream-runtime cache transitions.

    The runtime deliberately does not detach or tear down sessions, because
    those operations still touch pyqtgraph/Qt objects. The active
    ``ProbeSession`` remains a view adapter; cached heavy ephys data lives in
    ``EphysStreamRuntime`` objects.
    """

    active_session: ProbeSession | None = field(default_factory=ProbeSession)
    active_stream_runtime: EphysStreamRuntime | None = None
    stream_cache: dict[StreamKey, EphysStreamRuntime] = field(default_factory=dict)
    current_stream_key: StreamKey | None = None

    def new_session(self) -> ProbeSession:
        """Replace the active session with a fresh ProbeSession."""
        self.active_session = ProbeSession()
        self.active_stream_runtime = None
        self.current_stream_key = None
        return self.active_session

    def detach_active_for_cache(self) -> ProbeSession | None:
        """Clear active view-session ownership and return it for view cleanup."""
        session = self.active_session
        if session is None:
            return None
        self.active_session = None
        self.active_stream_runtime = None
        self.current_stream_key = None
        return session

    def sessions_for_stream_eviction(self) -> list[ProbeSession]:
        """Return current view sessions that should be torn down by the view."""
        sessions = [self.active_session] if self.active_session is not None else []
        self.active_session = None
        self.stream_cache.clear()
        self.active_stream_runtime = None
        self.current_stream_key = None
        return sessions

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
