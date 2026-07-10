"""Runtime ownership for active and cached probe sessions."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.probe_session import ProbeSession


@dataclass
class SessionRuntime:
    """Owns active ProbeSession and stream-cache transitions.

    The runtime deliberately does not detach or tear down sessions, because
    those operations still touch pyqtgraph/Qt objects. It only decides which
    sessions remain active, cached, or need view-layer cleanup.
    """

    active_session: ProbeSession | None = field(default_factory=ProbeSession)
    stream_cache: dict[str, ProbeSession] = field(default_factory=dict)
    current_stream_key: str | None = None

    def new_session(self) -> ProbeSession:
        """Replace the active session with a fresh ProbeSession."""
        self.active_session = ProbeSession()
        self.current_stream_key = None
        return self.active_session

    def cache_active_session(self) -> None:
        """Cache the active session under the current stream key, if keyed."""
        if self.active_session is not None and self.current_stream_key is not None:
            self.stream_cache[self.current_stream_key] = self.active_session

    def detach_active_for_cache(self) -> ProbeSession | None:
        """Move the active session into the cache and clear active ownership."""
        session = self.active_session
        if session is None:
            return None
        self.cache_active_session()
        self.active_session = None
        self.current_stream_key = None
        return session

    def sessions_for_stream_eviction(self) -> list[ProbeSession]:
        """Return cached/current sessions that should be torn down by the view."""
        self.cache_active_session()
        self.active_session = None
        sessions = list(self.stream_cache.values())
        self.stream_cache.clear()
        self.current_stream_key = None
        return sessions

    def has_cached_stream(self, stream_key: str) -> bool:
        """Whether a stream session is cached."""
        return stream_key in self.stream_cache

    def cached_stream(self, stream_key: str) -> ProbeSession | None:
        """Return a cached stream session, if present."""
        return self.stream_cache.get(stream_key)

    def pop_cached_stream(self, stream_key: str) -> ProbeSession | None:
        """Remove and return a cached stream session, if present."""
        return self.stream_cache.pop(stream_key, None)

    def activate_cached_stream(self, stream_key: str) -> ProbeSession:
        """Make a cached stream session active."""
        session = self.stream_cache[stream_key]
        self.active_session = session
        self.current_stream_key = stream_key
        return session

    def cache_loaded_stream(
        self,
        stream_key: str,
        session: ProbeSession | None = None,
    ) -> None:
        """Cache a freshly loaded stream and mark it active."""
        session = session if session is not None else self.active_session
        if session is None:
            return
        self.stream_cache[stream_key] = session
        self.active_session = session
        self.current_stream_key = stream_key
