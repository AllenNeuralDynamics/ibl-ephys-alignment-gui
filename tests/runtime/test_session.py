"""Tests for cached stream-runtime ownership."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.runtime.ephys_stream import EphysStreamRuntime
from ephys_alignment_gui.runtime.session import (
    LoadDataAlreadyActive,
    LoadDataCachedStreamAvailable,
    LoadDataFreshRequired,
    LoadDataTarget,
    SessionRuntime,
)
from ephys_alignment_gui.services.ephys_data import ChannelTable, EphysStreamData


class FakePlotPayloadCacheFactory:
    def build(self, collection):
        return {"rows": collection.rows.copy()}


def _stream_runtime(collection: str = "streamA") -> EphysStreamRuntime:
    table = ChannelTable(
        local_coordinates=np.array([[0.0, 0.0], [0.0, 20.0]]),
        shank_indices=np.array([0, 1]),
    )
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection=collection,
        ephys_dir=Path("/tmp/ephys"),
        channel_table=table,
        alf_data={"channels": {"exists": True}},
        session_notes="notes",
    )
    return EphysStreamRuntime(stream, FakePlotPayloadCacheFactory())


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


def test_activate_cached_stream_for_shank_initializes_requested_shank() -> None:
    runtime = SessionRuntime()
    stream_runtime = _stream_runtime()
    runtime.stream_cache[stream_runtime.stream_key] = stream_runtime

    active = runtime.activate_cached_stream_for_shank(
        stream_runtime.stream_key,
        shank_idx=1,
    )

    assert active is stream_runtime
    assert runtime.active_stream_runtime is stream_runtime
    assert runtime.current_stream_key == ("rec1", "streamA")
    assert stream_runtime.current_shank_idx == 1
    assert 1 in stream_runtime.shank_runtime_by_idx


def test_activate_cached_stream_for_shank_leaves_active_clear_on_failure() -> None:
    runtime = SessionRuntime()
    stream_runtime = _stream_runtime()
    runtime.stream_cache[stream_runtime.stream_key] = stream_runtime

    with pytest.raises(IndexError):
        runtime.activate_cached_stream_for_shank(
            stream_runtime.stream_key,
            shank_idx=3,
        )

    assert runtime.active_stream_runtime is None
    assert runtime.current_stream_key is None
    assert stream_runtime.current_shank_idx == 0


def test_cache_loaded_stream_stores_runtime_by_stream_key() -> None:
    runtime = SessionRuntime()
    stream_runtime = _stream_runtime()

    runtime.cache_loaded_stream(stream_runtime)

    assert runtime.stream_cache[("rec1", "streamA")] is stream_runtime
    assert runtime.active_stream_runtime is stream_runtime
    assert runtime.current_stream_key == ("rec1", "streamA")


def test_cache_loaded_stream_data_builds_runtime_and_initializes_shank() -> None:
    runtime = SessionRuntime()
    stream = _stream_runtime().stream

    stream_runtime = runtime.cache_loaded_stream_data(
        stream,
        FakePlotPayloadCacheFactory(),
        shank_idx=1,
    )

    assert stream_runtime.stream is stream
    assert runtime.stream_cache[("rec1", "streamA")] is stream_runtime
    assert runtime.active_stream_runtime is stream_runtime
    assert runtime.current_stream_key == ("rec1", "streamA")
    assert stream_runtime.current_shank_idx == 1
    assert 1 in stream_runtime.shank_runtime_by_idx


def test_cache_loaded_stream_data_can_store_without_activation() -> None:
    runtime = SessionRuntime()
    stream = _stream_runtime().stream

    stream_runtime = runtime.cache_loaded_stream_data(
        stream,
        FakePlotPayloadCacheFactory(),
        shank_idx=1,
        activate=False,
    )

    assert runtime.stream_cache[("rec1", "streamA")] is stream_runtime
    assert runtime.active_stream_runtime is None
    assert runtime.current_stream_key is None
    assert stream_runtime.current_shank_idx == 1


def test_cache_loaded_stream_data_skips_cache_on_shank_init_failure() -> None:
    runtime = SessionRuntime()
    stream = _stream_runtime().stream

    with pytest.raises(IndexError):
        runtime.cache_loaded_stream_data(
            stream,
            FakePlotPayloadCacheFactory(),
            shank_idx=3,
        )

    assert runtime.stream_cache == {}
    assert runtime.active_stream_runtime is None
    assert runtime.current_stream_key is None


def test_plan_load_data_returns_already_active_for_loaded_active_stream_shank() -> None:
    runtime = SessionRuntime()
    stream_runtime = _stream_runtime()
    stream_runtime.shank_runtime_for(1)
    runtime.cache_loaded_stream(stream_runtime)

    plan = runtime.plan_load_data(
        LoadDataTarget(stream_key=("rec1", "streamA"), shank_idx=1),
        data_loaded=True,
    )

    assert isinstance(plan, LoadDataAlreadyActive)
    assert plan.target.stream_key == ("rec1", "streamA")
    assert plan.target.shank_idx == 1


def test_plan_load_data_returns_cached_when_stream_is_cached_but_not_active() -> None:
    runtime = SessionRuntime()
    stream_runtime = _stream_runtime()
    stream_runtime.shank_runtime_for(1)
    runtime.stream_cache[stream_runtime.stream_key] = stream_runtime

    plan = runtime.plan_load_data(
        LoadDataTarget(stream_key=("rec1", "streamA"), shank_idx=0),
        data_loaded=True,
    )

    assert isinstance(plan, LoadDataCachedStreamAvailable)
    assert plan.target.stream_key == ("rec1", "streamA")
    assert plan.target.shank_idx == 0
    assert plan.cached_shank_idx == 1


def test_plan_load_data_returns_fresh_when_stream_is_not_cached() -> None:
    runtime = SessionRuntime()

    plan = runtime.plan_load_data(
        LoadDataTarget(stream_key=("rec1", "streamA"), shank_idx=0),
        data_loaded=True,
    )

    assert isinstance(plan, LoadDataFreshRequired)
    assert plan.target.stream_key == ("rec1", "streamA")
    assert plan.target.shank_idx == 0


def test_plan_load_data_treats_missing_stream_key_as_fresh_load() -> None:
    runtime = SessionRuntime()

    plan = runtime.plan_load_data(
        LoadDataTarget(stream_key=None, shank_idx=0),
        data_loaded=True,
    )

    assert isinstance(plan, LoadDataFreshRequired)
    assert plan.target.stream_key is None


def test_prepare_fresh_load_discards_stale_cache_entry_and_active_stream() -> None:
    runtime = SessionRuntime()
    stream_runtime = _stream_runtime()
    other_runtime = _stream_runtime("streamB")
    runtime.cache_loaded_stream(stream_runtime)
    runtime.stream_cache[other_runtime.stream_key] = other_runtime

    stale = runtime.prepare_fresh_load(("rec1", "streamA"))

    assert stale is stream_runtime
    assert ("rec1", "streamA") not in runtime.stream_cache
    assert runtime.stream_cache[("rec1", "streamB")] is other_runtime
    assert runtime.active_stream_runtime is None
    assert runtime.current_stream_key is None
