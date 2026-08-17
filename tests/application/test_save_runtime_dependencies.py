"""Tests for save runtime dependency planning."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ephys_alignment_gui.application.save_runtime_dependencies import (
    describe_save_runtime_dependencies,
    plan_save_runtime_dependencies,
)
from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.runtime.session import SessionRuntime


class FakeDataContext:
    def __init__(self, *, mouse_root=True) -> None:
        self.mouse_root = (
            SimpleNamespace(root=Path("/tmp/mouse"), mouse_id="mouse")
            if mouse_root
            else None
        )
        self.probe_info = None
        self.channel_table = None

    def probe_for_stream_key(self, recording_id: str, ephys_collection: str):
        return SimpleNamespace(
            recording_id=recording_id,
            ephys_collection=ephys_collection,
            probe_name=f"probe-{ephys_collection}",
        )


def test_plan_save_runtime_dependencies_identifies_cached_dirty_runtime() -> None:
    key = AlignmentKey("rec", "stream", 1)
    document = _document_with_dirty_alignment(key)
    runtime = SessionRuntime()
    runtime.stream_cache[("rec", "stream")] = _stream_runtime(("rec", "stream"))

    plan = plan_save_runtime_dependencies(
        document=document,
        data_context=FakeDataContext(),
        runtime=runtime,
    )

    assert len(plan.dependencies) == 1
    dependency = plan.dependencies[0]
    assert dependency.key == key
    assert dependency.status == "cached"
    assert dependency.available
    assert dependency.load_target is not None
    assert dependency.load_target.probe_name == "probe-stream"
    assert plan.unavailable == ()
    assert plan.eviction_protected == (dependency,)
    assert describe_save_runtime_dependencies(plan.eviction_protected) == (
        "rec/stream shank 2"
    )


def test_plan_save_runtime_dependencies_reports_missing_runtime_reload_need() -> None:
    key = AlignmentKey("rec", "stream", 0)
    document = _document_with_dirty_alignment(key)

    plan = plan_save_runtime_dependencies(
        document=document,
        data_context=FakeDataContext(),
        runtime=SessionRuntime(),
    )

    dependency = plan.dependencies[0]
    assert dependency.status == "missing"
    assert dependency.needs_reload
    assert not dependency.available
    assert dependency.load_target is None
    assert plan.failure_message() is not None
    assert "stream runtime is not loaded" in plan.failure_message()


def test_plan_save_runtime_dependencies_defaults_to_dirty_scope() -> None:
    dirty_key = AlignmentKey("rec", "dirty-stream", 0)
    clean_key = AlignmentKey("rec", "clean-stream", 0)
    document = _document_with_dirty_alignment(dirty_key)
    clean_state = document.alignment_state_for(clean_key)
    clean_state.active_alignment = ActiveAlignment(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )

    plan = plan_save_runtime_dependencies(
        document=document,
        data_context=FakeDataContext(),
        runtime=SessionRuntime(),
    )

    assert [dependency.key for dependency in plan.dependencies] == [dirty_key]


def test_plan_save_runtime_dependencies_accepts_explicit_saveable_scope() -> None:
    dirty_key = AlignmentKey("rec", "dirty-stream", 0)
    clean_key = AlignmentKey("rec", "clean-stream", 0)
    document = _document_with_dirty_alignment(dirty_key)
    clean_state = document.alignment_state_for(clean_key)
    clean_state.active_alignment = ActiveAlignment(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )

    plan = plan_save_runtime_dependencies(
        document=document,
        data_context=FakeDataContext(),
        runtime=SessionRuntime(),
        keys=document.saveable_alignment_states(),
    )

    assert [dependency.key for dependency in plan.dependencies] == [
        clean_key,
        dirty_key,
    ]


def test_plan_save_runtime_dependencies_protects_runtime_without_metadata() -> None:
    key = AlignmentKey("rec", "stream", 0)
    document = _document_with_dirty_alignment(key)
    runtime = SessionRuntime()
    runtime.stream_cache[("rec", "stream")] = _stream_runtime(("rec", "stream"))

    plan = plan_save_runtime_dependencies(
        document=document,
        data_context=FakeDataContext(mouse_root=False),
        runtime=runtime,
    )

    dependency = plan.dependencies[0]
    assert dependency.status == "cached"
    assert dependency.available
    assert dependency.load_target is None
    assert plan.eviction_protected == (dependency,)


def _document_with_dirty_alignment(key: AlignmentKey) -> AlignmentDocument:
    document = AlignmentDocument()
    state = document.alignment_state_for(key)
    state.active_alignment = ActiveAlignment(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )
    state.mark_alignment_changed()
    return document


def _stream_runtime(stream_key: tuple[str, str]):
    return SimpleNamespace(
        stream_key=stream_key,
        stream=SimpleNamespace(stream_key=stream_key, channel_table="channel-table"),
    )
