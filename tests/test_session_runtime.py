"""Tests for active/cached ProbeSession runtime ownership."""

from __future__ import annotations

from ephys_alignment_gui.probe_session import ProbeSession
from ephys_alignment_gui.session_runtime import SessionRuntime


def test_new_session_replaces_active_and_clears_current_stream() -> None:
    runtime = SessionRuntime()
    old = runtime.active_session
    runtime.current_stream_key = "probeA"

    new = runtime.new_session()

    assert new is runtime.active_session
    assert new is not old
    assert runtime.current_stream_key is None


def test_cache_active_session_requires_stream_key() -> None:
    runtime = SessionRuntime()
    active = runtime.active_session

    runtime.cache_active_session()
    assert runtime.stream_cache == {}

    runtime.current_stream_key = "probeA"
    runtime.cache_active_session()

    assert runtime.stream_cache["probeA"] is active


def test_detach_active_for_cache_returns_session_and_clears_active() -> None:
    runtime = SessionRuntime()
    active = runtime.active_session
    runtime.current_stream_key = "probeA"

    detached = runtime.detach_active_for_cache()

    assert detached is active
    assert runtime.active_session is None
    assert runtime.current_stream_key is None
    assert runtime.stream_cache["probeA"] is active


def test_sessions_for_stream_eviction_returns_cached_and_active_sessions() -> None:
    runtime = SessionRuntime()
    active = runtime.active_session
    cached = ProbeSession()
    runtime.current_stream_key = "probeA"
    runtime.stream_cache["probeB"] = cached

    sessions = runtime.sessions_for_stream_eviction()

    assert sessions == [cached, active]
    assert runtime.active_session is None
    assert runtime.stream_cache == {}
    assert runtime.current_stream_key is None


def test_activate_cached_stream_sets_active_and_current_key() -> None:
    runtime = SessionRuntime()
    cached = ProbeSession()
    runtime.stream_cache["probeA"] = cached

    active = runtime.activate_cached_stream("probeA")

    assert active is cached
    assert runtime.active_session is cached
    assert runtime.current_stream_key == "probeA"


def test_cache_loaded_stream_uses_active_session_by_default() -> None:
    runtime = SessionRuntime()
    active = runtime.active_session

    runtime.cache_loaded_stream("probeA")

    assert runtime.stream_cache["probeA"] is active
    assert runtime.active_session is active
    assert runtime.current_stream_key == "probeA"
