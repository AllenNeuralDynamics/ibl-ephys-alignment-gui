"""Tests for ephys stream runtime cache ownership."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ephys_alignment_gui.ephys_data_service import ChannelTable, EphysStreamData
from ephys_alignment_gui.ephys_stream_runtime import EphysStreamRuntime


class FakePlotDataFactory:
    def __init__(self) -> None:
        self.calls = []

    def build(self, collection):
        self.calls.append(collection)
        return {"rows": collection.rows.copy()}


class FakeRegistryPlotData:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.calls = []

    def cached(self, method: str, args: tuple = ()) -> Any:
        self.calls.append((method, args))
        if method == "get_fr_img":
            return {"rows": self.rows.copy()}
        return None


class FakeRegistryPlotDataFactory:
    def __init__(self) -> None:
        self.plotdata = None

    def build(self, collection):
        self.plotdata = FakeRegistryPlotData(collection.rows.copy())
        return self.plotdata


def _stream() -> EphysStreamData:
    table = ChannelTable(
        local_coordinates=np.array(
            [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
        ),
        shank_indices=np.array([0, 0, 1, 1]),
    )
    return EphysStreamData(
        recording_id="rec1",
        ephys_collection="streamA",
        ephys_dir=Path("/tmp/ephys"),
        channel_table=table,
        alf_data={"channels": {"exists": True}},
        session_notes="notes",
    )


def test_collection_for_shank_returns_view_and_records_current_shank() -> None:
    runtime = EphysStreamRuntime(_stream(), FakePlotDataFactory())

    collection = runtime.collection_for_shank(1)

    assert runtime.stream_key == ("rec1", "streamA")
    assert runtime.current_shank_idx == 1
    assert collection.rows.tolist() == [2, 3]


def test_shank_runtime_for_shank_is_cached() -> None:
    runtime = EphysStreamRuntime(_stream(), FakePlotDataFactory())

    first = runtime.shank_runtime_for(1)
    second = runtime.shank_runtime_for(1)

    assert first is second
    assert first.collection.rows.tolist() == [2, 3]
    assert first.chn_depths.tolist() == [0.0, 20.0]
    assert runtime.visited_shank_runtimes() == {1: first}


def test_plot_data_for_shank_is_cached_per_shank() -> None:
    factory = FakePlotDataFactory()
    runtime = EphysStreamRuntime(_stream(), factory)

    first = runtime.plot_data_for_shank(1)
    second = runtime.plot_data_for_shank(1)

    assert first is second
    assert first["rows"].tolist() == [2, 3]
    assert len(factory.calls) == 1


def test_plot_payload_for_shank_resolves_registered_plot_spec() -> None:
    factory = FakeRegistryPlotDataFactory()
    runtime = EphysStreamRuntime(_stream(), factory)

    payload = runtime.plot_payload_for_shank(1, "image.fr")

    assert payload["rows"].tolist() == [2, 3]
    assert factory.plotdata.calls == [("get_fr_img", ())]


def test_invalidate_plot_data_clears_one_or_all_shanks() -> None:
    runtime = EphysStreamRuntime(_stream(), FakePlotDataFactory())
    runtime.plot_data_for_shank(0)
    runtime.plot_data_for_shank(1)

    runtime.invalidate_plot_data(0)

    assert runtime.shank_runtime_for(0).plotdata is None
    assert runtime.shank_runtime_for(1).plotdata is not None

    runtime.invalidate_plot_data()

    assert runtime.shank_runtime_for(0).plotdata is None
    assert runtime.shank_runtime_for(1).plotdata is None
