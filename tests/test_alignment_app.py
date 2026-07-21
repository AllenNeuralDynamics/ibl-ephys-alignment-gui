"""Tests for the UI-facing alignment app port."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.app import AlignmentQueries
from ephys_alignment_gui.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.workspace import AlignmentWorkspace


class FakePlotData:
    def __init__(self, label: str = "plot") -> None:
        self.label = label
        self.data = {
            "spikes": {"exists": True},
            "clusters": {"exists": False},
            "rms_AP": {"exists": False},
            "rms_AP_main": {"exists": False},
            "rms_LF": {"exists": False},
            "rms_LF_main": {"exists": False},
            "psd_lf": {"exists": False},
            "psd_lf_main": {"exists": False},
        }

    def cached(self, method: str, args: tuple = ()) -> Any:
        if method == "get_fr_img":
            return {"label": self.label}
        if method == "get_lfp_correlation_data_img":
            return {}
        if method == "get_passive_events":
            return {}
        if method == "get_lfp_spectrum_data":
            return None, {}
        if method == "get_rfmap_data":
            return {}, None
        return None


class FakeStreamRuntime:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.plotdata_by_shank = {
            1: FakePlotData("shank-1"),
            2: FakePlotData("shank-2"),
        }

    def plot_data_for_shank(self, shank_idx: int) -> FakePlotData:
        self.calls.append(shank_idx)
        return self.plotdata_by_shank[shank_idx]


def test_workspace_exposes_app_port() -> None:
    workspace = AlignmentWorkspace()

    assert workspace.app.events is workspace.events
    assert workspace.app.queries.document is workspace.document
    assert workspace.app.queries.runtime is workspace.runtime


def test_queries_build_plot_menu_state_from_active_runtime_shank() -> None:
    document = AlignmentDocument()
    document.select_alignment_key(
        AlignmentKey(
            recording_id="rec",
            ephys_collection="stream",
            shank_idx=2,
        )
    )
    stream_runtime = FakeStreamRuntime()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=stream_runtime),
    )

    state = queries.active_plot_menu_state()

    assert state.group("image").selected_key == "image.fr"
    assert stream_runtime.calls == [2]


def test_queries_resolve_plot_payload_from_active_runtime_shank() -> None:
    document = AlignmentDocument(selected_shank=1)
    stream_runtime = FakeStreamRuntime()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=stream_runtime),
    )

    payload = queries.active_plot_payload("image.fr")

    assert payload == {"label": "shank-1"}
    assert stream_runtime.calls == [1]


def test_queries_can_resolve_raw_payload_without_plotdata() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    state = queries.active_plot_menu_state(
        previous_selected_keys={"image": "image.raw.raw_ap"},
        raw_image_payloads={"raw_ap": "raw-image"},
    )
    payload = queries.active_plot_payload(
        "image.raw.raw_ap",
        raw_image_payloads={"raw_ap": "raw-image"},
    )

    assert state.group("image").selected_key == "image.raw.raw_ap"
    assert payload == "raw-image"


def test_queries_fail_closed_without_plotdata_or_raw_payloads() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    state = queries.active_plot_menu_state()

    assert not state.group("image").enabled
    assert queries.active_plot_payload("image.fr") is None
