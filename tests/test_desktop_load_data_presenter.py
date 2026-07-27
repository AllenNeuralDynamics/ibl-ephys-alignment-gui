"""Tests for desktop load-data presentation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.app import (
    CachedEphysDataActivated,
    FreshEphysDataLoaded,
)
from ephys_alignment_gui.controller import LoadDataPrepared, ProbeSelected
from ephys_alignment_gui.desktop_load_data_presenter import (
    DesktopLoadDataCallbacks,
    DesktopLoadDataPresenter,
)
from ephys_alignment_gui.histology_data_workflow import (
    HistologyDataLoaded,
    HistologyDataUnavailable,
)
from ephys_alignment_gui.session_runtime import (
    LoadDataAlreadyActive,
    LoadDataCachedStreamAvailable,
    LoadDataFreshRequired,
    LoadDataTarget,
)
from ephys_alignment_gui.workflow import Failed


class FakeBusyContext:
    def __init__(
        self,
        calls: list[tuple],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.calls = calls
        self.calls.append(("busy", args, kwargs))

    def __enter__(self):
        self.calls.append(("busy-enter",))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.calls.append(("busy-exit", exc_type))
        return False

    def update_message(self, message: str) -> None:
        self.calls.append(("message", message))


class FakeQueries:
    def __init__(
        self,
        *,
        plan: Any,
        shank_idx: int = 0,
        histology_loaded: bool = False,
    ) -> None:
        self.plan = plan
        self.shank_idx = shank_idx
        self.histology_loaded = histology_loaded
        self.stream_key_calls: list[tuple[str, str]] = []
        self.plan_calls: list[tuple[Any, int]] = []

    def active_shank_selection(self):
        return SimpleNamespace(shank_idx=self.shank_idx)

    def stream_key_for_selection(self, session_name: str, probe_name: str):
        self.stream_key_calls.append((session_name, probe_name))
        return ("rec", "stream")

    def plan_load_data(self, stream_key, target_shank: int):
        self.plan_calls.append((stream_key, target_shank))
        return self.plan

    def histology_data_loaded(self) -> bool:
        return self.histology_loaded


class FakeCommands:
    def __init__(
        self,
        *,
        cached_result: Any | None = None,
        histology_result: Any | None = None,
    ) -> None:
        self.cached_result = cached_result or _cached_result(shank_idx=0)
        self.histology_result = histology_result or HistologyDataLoaded()
        self.prepare_calls: list[Any] = []
        self.cached_calls: list[dict[str, Any]] = []
        self.fresh_calls: list[int] = []

    def prepare_fresh_ephys_load(self, stream_key):
        self.prepare_calls.append(stream_key)
        return LoadDataPrepared(preserve_plot_selection=True)

    def activate_cached_ephys_data(self, **kwargs):
        self.cached_calls.append(kwargs)
        return self.cached_result

    def load_fresh_ephys_data(self, shank_idx: int):
        self.fresh_calls.append(shank_idx)
        stream_runtime = SimpleNamespace(
            stream=SimpleNamespace(ephys_dir=Path("/tmp/ephys"))
        )
        return FreshEphysDataLoaded(stream_runtime=stream_runtime, shank_idx=shank_idx)

    def load_histology_data(self):
        return self.histology_result


class FakeSelectionView:
    def __init__(
        self,
        calls: list[tuple],
        *,
        session_name: str = "rec",
        probe_name: str = "probeA",
    ) -> None:
        self.calls = calls
        self.session_name = session_name
        self.probe_name = probe_name

    def current_session(self) -> str:
        return self.session_name

    def current_probe(self) -> str:
        return self.probe_name

    def populate_loaded_shanks(self, shanks: list[str], shank_idx: int) -> None:
        self.calls.append(("populate", shanks, shank_idx))

    def set_load_data_enabled(self, enabled: bool) -> None:
        self.calls.append(("enable-load", enabled))

    def load_data_widget(self) -> str:
        return "load-button"


def _cached_result(shank_idx: int) -> CachedEphysDataActivated:
    return CachedEphysDataActivated(
        stream_runtime=object(),
        shank_idx=shank_idx,
        probe=ProbeSelected(
            recording_id="rec",
            probe_name="probeA",
            shanks=["1/2", "2/2"],
            n_shanks=2,
            output_directory=Path("/tmp/out"),
        ),
    )


def _callbacks(calls: list[tuple]) -> DesktopLoadDataCallbacks:
    return DesktopLoadDataCallbacks(
        capture_pending_reference_lines=lambda: calls.append(("capture",)),
        stash_and_detach_current=lambda: calls.append(("stash",)),
        teardown_session=lambda: calls.append(("teardown",)),
        init_session_variables=lambda: calls.append(("init",)),
        select_shank_for_view=lambda shank_idx, source: (
            calls.append(("select-shank", shank_idx, source)) or shank_idx
        ),
        display_output_directory=lambda path: calls.append(("output", path)),
        setup_session_view=lambda preserve, shank_idx: calls.append(
            ("setup", preserve, shank_idx)
        ),
        clear_empty_state=lambda: calls.append(("clear-empty",)),
        set_histology_available=lambda available: calls.append(
            ("histology", available)
        ),
        busy_context=lambda *args, **kwargs: FakeBusyContext(calls, *args, **kwargs),
    )


def _presenter(
    *,
    plan: Any,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
    histology_loaded: bool = False,
) -> tuple[DesktopLoadDataPresenter, FakeCommands, FakeQueries, list[tuple]]:
    calls = calls if calls is not None else []
    queries = FakeQueries(plan=plan, histology_loaded=histology_loaded)
    commands = commands or FakeCommands()
    app = SimpleNamespace(queries=queries, commands=commands)
    selection_view = FakeSelectionView(calls)
    return (
        DesktopLoadDataPresenter(app, selection_view, _callbacks(calls)),
        commands,
        queries,
        calls,
    )


def test_load_heavy_data_skips_already_active_stream_shank() -> None:
    presenter, commands, queries, calls = _presenter(
        plan=LoadDataAlreadyActive(LoadDataTarget(("rec", "stream"), shank_idx=0))
    )

    assert presenter.load_heavy_data()

    assert queries.plan_calls == [(("rec", "stream"), 0)]
    assert commands.prepare_calls == []
    assert calls == []


def test_load_heavy_data_presents_cached_stream_for_selected_shank() -> None:
    plan = LoadDataCachedStreamAvailable(
        target=LoadDataTarget(("rec", "stream"), shank_idx=0),
        cached_shank_idx=1,
    )
    presenter, commands, _queries, calls = _presenter(plan=plan)

    assert presenter.load_heavy_data()

    assert commands.cached_calls == [
        {
            "recording_id": "rec",
            "probe_name": "probeA",
            "stream_key": ("rec", "stream"),
            "shank_idx": 0,
        }
    ]
    assert calls == [
        ("capture",),
        ("stash",),
        ("init",),
        ("clear-empty",),
        ("populate", ["1/2", "2/2"], 0),
        ("output", Path("/tmp/out")),
        ("setup", True, 0),
        ("enable-load", True),
    ]


def test_probe_selection_presents_cached_stream_for_cached_shank() -> None:
    plan = LoadDataCachedStreamAvailable(
        target=LoadDataTarget(("rec", "stream"), shank_idx=0),
        cached_shank_idx=1,
    )
    presenter, commands, _queries, calls = _presenter(
        plan=plan,
        commands=FakeCommands(cached_result=_cached_result(shank_idx=1)),
    )

    assert presenter.present_cached_probe_selection(
        session_name="rec",
        probe_name="probeA",
        target_shank=0,
    )

    assert commands.cached_calls[0]["shank_idx"] == 1
    assert ("setup", True, 1) in calls
    assert ("enable-load", True) in calls


def test_load_heavy_data_runs_fresh_load_and_renders_result() -> None:
    plan = LoadDataFreshRequired(LoadDataTarget(("rec", "stream"), shank_idx=0))
    presenter, commands, _queries, calls = _presenter(plan=plan)

    assert presenter.load_heavy_data()

    assert commands.prepare_calls == [("rec", "stream")]
    assert commands.fresh_calls == [0]
    assert ("message", "Loading ephys data...") in calls
    assert ("message", "Loading atlas and histology...") in calls
    assert ("message", "Setting up visualization...") in calls
    assert ("setup", True, 0) in calls
    assert ("clear-empty",) in calls


def test_load_heavy_data_marks_histology_unavailable_nonfatal() -> None:
    plan = LoadDataFreshRequired(LoadDataTarget(("rec", "stream"), shank_idx=0))
    presenter, _commands, _queries, calls = _presenter(
        plan=plan,
        commands=FakeCommands(
            histology_result=HistologyDataUnavailable("no histology")
        ),
    )

    assert presenter.load_heavy_data()

    assert ("histology", False) in calls
    assert ("setup", True, 0) in calls


def test_present_cached_stream_returns_false_on_command_failure() -> None:
    presenter, commands, _queries, calls = _presenter(
        plan=LoadDataFreshRequired(LoadDataTarget(("rec", "stream"), shank_idx=0)),
        commands=FakeCommands(cached_result=Failed("missing cache")),
    )

    assert not presenter.present_cached_stream(
        session_name="rec",
        probe_name="probeA",
        stream_key=("rec", "stream"),
        shank_idx=0,
    )

    assert commands.cached_calls
    assert calls == [("init",)]
