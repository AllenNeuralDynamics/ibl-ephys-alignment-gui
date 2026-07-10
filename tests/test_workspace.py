"""Tests for the Qt-free alignment workspace."""

from __future__ import annotations

from ephys_alignment_gui.probe_session import ProbeSession
from ephys_alignment_gui.workspace import AlignmentWorkspace


def test_workspace_wires_shared_services() -> None:
    workspace = AlignmentWorkspace()

    assert workspace.controller.document is workspace.document
    assert workspace.controller.loader is workspace.loader
    assert workspace.controller.workflow_policy is workspace.workflow_policy
    assert workspace.controller.alignment_repository is workspace.alignment_repository
    assert workspace.loader.ephys_data_service is workspace.ephys_data_service


def test_workspace_owns_stream_cache_boundary() -> None:
    workspace = AlignmentWorkspace()
    session = ProbeSession()

    workspace.cache_current_session(session)
    assert workspace.stream_cache == {}

    workspace.set_current_stream("probeA")
    workspace.cache_current_session(session)

    assert workspace.cached_stream("probeA") is session
    assert workspace.pop_cached_stream("missing") is None
    assert workspace.pop_cached_stream("probeA") is session
    assert workspace.stream_cache == {}


def test_workspace_clear_stream_cache_returns_sessions_for_teardown() -> None:
    workspace = AlignmentWorkspace()
    session_a = ProbeSession()
    session_b = ProbeSession()
    workspace.stream_cache["probeA"] = session_a
    workspace.stream_cache["probeB"] = session_b
    workspace.set_current_stream("probeB")

    sessions = workspace.clear_stream_cache()

    assert sessions == [session_a, session_b]
    assert workspace.stream_cache == {}
    assert workspace.current_stream_key is None
