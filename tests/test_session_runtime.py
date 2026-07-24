"""Tests for cached stream-runtime ownership."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ephys_alignment_gui.ephys_data_service import ChannelTable, EphysStreamData
from ephys_alignment_gui.ephys_stream_runtime import EphysStreamRuntime
from ephys_alignment_gui.session_runtime import SessionRuntime


class FakePlotDataFactory:
    def build(self, collection):
        return {"rows": collection.rows.copy()}


def _stream_runtime(collection: str = "streamA") -> EphysStreamRuntime:
    table = ChannelTable(
        local_coordinates=np.array([[0.0, 0.0], [0.0, 20.0]]),
        shank_indices=np.array([0, 0]),
    )
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection=collection,
        ephys_dir=Path("/tmp/ephys"),
        channel_table=table,
        alf_data={"channels": {"exists": True}},
        session_notes="notes",
    )
    return EphysStreamRuntime(stream, FakePlotDataFactory())


def test_clear_active_stream_clears_current_stream_without_evicting_cache() -> None:
    runtime = SessionRuntime()
    stream_runtime = _stream_runtime()
    runtime.cache_loaded_stream(stream_runtime)

    runtime.clear_active_stream()

    assert runtime.active_stream_runtime is None
    assert runtime.current_stream_key is None
    assert runtime.stream_cache[("rec1", "streamA")] is stream_runtime


def test_clear_stream_cache_removes_cached_streams_and_active_selection() -> None:
    runtime = SessionRuntime()
    runtime.cache_loaded_stream(_stream_runtime("streamA"))
    runtime.cache_loaded_stream(_stream_runtime("streamB"))

    runtime.clear_stream_cache()

    assert runtime.stream_cache == {}
    assert runtime.active_stream_runtime is None
    assert runtime.current_stream_key is None


def test_activate_cached_stream_sets_active_runtime_and_current_key() -> None:
    runtime = SessionRuntime()
    stream_runtime = _stream_runtime()
    runtime.stream_cache[stream_runtime.stream_key] = stream_runtime

    active = runtime.activate_cached_stream(stream_runtime.stream_key)

    assert active is stream_runtime
    assert runtime.active_stream_runtime is stream_runtime
    assert runtime.current_stream_key == ("rec1", "streamA")


def test_cache_loaded_stream_stores_runtime_by_stream_key() -> None:
    runtime = SessionRuntime()
    stream_runtime = _stream_runtime()

    runtime.cache_loaded_stream(stream_runtime)

    assert runtime.stream_cache[("rec1", "streamA")] is stream_runtime
    assert runtime.active_stream_runtime is stream_runtime
    assert runtime.current_stream_key == ("rec1", "streamA")
