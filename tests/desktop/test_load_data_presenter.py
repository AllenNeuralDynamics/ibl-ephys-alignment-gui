"""Tests for desktop load-data presentation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.application.results import (
    CachedEphysDataActivated,
    FreshEphysDataLoaded,
    LoadDataAlreadyActiveResult,
    LoadDataCachedActivated,
    LoadDataFreshCompleted,
    LoadDataFreshPrepared,
    LoadDataFreshRequiredResult,
)
from ephys_alignment_gui.application.results.metadata import ProbeSelected
from ephys_alignment_gui.application.workflow import Failed
from ephys_alignment_gui.desktop.load_data_presenter import (
    DesktopLoadDataCallbacks,
    DesktopLoadDataPresenter,
)
from ephys_alignment_gui.histology_runtime_loader import (
    HistologyDataLoaded,
    HistologyDataUnavailable,
)


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
    def __init__(self, *, shank_idx: int = 0) -> None:
        self.shank_idx = shank_idx
        self.workspace = SimpleNamespace(
            active_shank_selection=self.active_shank_selection,
        )

    def active_shank_selection(self):
        return SimpleNamespace(shank_idx=self.shank_idx)


class FakeCommands:
    def __init__(
        self,
        *,
        begin_result: Any | None = None,
        complete_result: Any | None = None,
        probe_cache_result: Any | None = None,
    ) -> None:
        self.begin_result = begin_result or _fresh_prepared(shank_idx=0)
        self.complete_result = complete_result or _fresh_completed(shank_idx=0)
        self.probe_cache_result = probe_cache_result or LoadDataFreshRequiredResult(
            ("rec", "stream"), 0
        )
        self.begin_calls: list[dict[str, Any]] = []
        self.complete_calls: list[LoadDataFreshPrepared] = []
        self.probe_cache_calls: list[dict[str, Any]] = []
        self.load = self

    def begin_load_data(self, **kwargs):
        self.begin_calls.append(kwargs)
        return self.begin_result

    def complete_fresh_load_data(self, prepared: LoadDataFreshPrepared):
        self.complete_calls.append(prepared)
        return self.complete_result

    def activate_cached_probe_selection(self, **kwargs):
        self.probe_cache_calls.append(kwargs)
        return self.probe_cache_result


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


def _cached_transaction(shank_idx: int) -> LoadDataCachedActivated:
    return LoadDataCachedActivated(
        stream_key=("rec", "stream"),
        activated=_cached_result(shank_idx),
    )


def _fresh_prepared(
    *,
    shank_idx: int,
    preserve_plot_selection: bool = True,
) -> LoadDataFreshPrepared:
    return LoadDataFreshPrepared(
        stream_key=("rec", "stream"),
        shank_idx=shank_idx,
        preserve_plot_selection=preserve_plot_selection,
    )


def _fresh_completed(
    *,
    shank_idx: int,
    histology_result: Any | None = None,
    preserve_plot_selection: bool = True,
) -> LoadDataFreshCompleted:
    stream_runtime = SimpleNamespace(
        stream=SimpleNamespace(ephys_dir=Path("/tmp/ephys"))
    )
    return LoadDataFreshCompleted(
        stream_key=("rec", "stream"),
        ephys=FreshEphysDataLoaded(
            stream_runtime=stream_runtime,
            shank_idx=shank_idx,
        ),
        histology=histology_result or HistologyDataLoaded(),
        preserve_plot_selection=preserve_plot_selection,
    )


def _callbacks(calls: list[tuple]) -> DesktopLoadDataCallbacks:
    return DesktopLoadDataCallbacks(
        reference_line_positions=lambda: calls.append(("positions",)) or ([1.0], [2.0]),
        prepare_for_fresh_stream_load=lambda: calls.append(("prepare-fresh",)),
        display_output_directory=lambda path: calls.append(("output", path)),
        render_loaded_shank=lambda shank_idx, preserve: calls.append(
            ("render-shank", shank_idx, preserve)
        ),
        clear_empty_state=lambda: calls.append(("clear-empty",)),
        busy_context=lambda *args, **kwargs: FakeBusyContext(calls, *args, **kwargs),
    )


def _presenter(
    *,
    begin_result: Any | None = None,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
) -> tuple[DesktopLoadDataPresenter, FakeCommands, FakeQueries, list[tuple]]:
    calls = calls if calls is not None else []
    queries = FakeQueries()
    commands = commands or FakeCommands(begin_result=begin_result)
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
        begin_result=LoadDataAlreadyActiveResult(("rec", "stream"), 0)
    )

    assert presenter.load_heavy_data()

    assert queries.shank_idx == 0
    assert commands.begin_calls == [
        {
            "recording_id": "rec",
            "probe_name": "probeA",
            "target_shank": 0,
            "outgoing_reference_lines": ([1.0], [2.0]),
        }
    ]
    assert commands.complete_calls == []
    assert calls == [("positions",)]


def test_load_heavy_data_presents_cached_stream_for_selected_shank() -> None:
    presenter, commands, _queries, calls = _presenter(
        begin_result=_cached_transaction(shank_idx=0),
    )

    assert presenter.load_heavy_data()

    assert commands.begin_calls == [
        {
            "recording_id": "rec",
            "probe_name": "probeA",
            "target_shank": 0,
            "outgoing_reference_lines": ([1.0], [2.0]),
        }
    ]
    assert calls == [
        ("positions",),
        ("clear-empty",),
        ("populate", ["1/2", "2/2"], 0),
        ("output", Path("/tmp/out")),
        ("render-shank", 0, True),
        ("enable-load", True),
    ]


def test_probe_selection_presents_cached_stream_for_cached_shank() -> None:
    presenter, commands, _queries, calls = _presenter(
        commands=FakeCommands(probe_cache_result=_cached_transaction(shank_idx=1)),
    )

    assert presenter.present_cached_probe_selection(
        session_name="rec",
        probe_name="probeA",
        target_shank=0,
    )

    assert commands.probe_cache_calls == [
        {
            "recording_id": "rec",
            "probe_name": "probeA",
            "target_shank": 0,
        }
    ]
    assert ("render-shank", 1, True) in calls
    assert ("enable-load", True) in calls


def test_probe_selection_noops_for_already_active_stream_shank() -> None:
    presenter, commands, _queries, calls = _presenter(
        commands=FakeCommands(
            probe_cache_result=LoadDataAlreadyActiveResult(("rec", "stream"), 0)
        ),
    )

    assert presenter.present_cached_probe_selection(
        session_name="rec",
        probe_name="probeA",
        target_shank=0,
    )

    assert commands.probe_cache_calls
    assert calls == [("enable-load", True)]


def test_load_heavy_data_runs_fresh_load_and_renders_result() -> None:
    prepared = _fresh_prepared(shank_idx=0, preserve_plot_selection=True)
    presenter, commands, _queries, calls = _presenter(begin_result=prepared)

    assert presenter.load_heavy_data()

    assert commands.complete_calls == [prepared]
    assert ("prepare-fresh",) in calls
    assert ("message", "Loading ephys and histology data...") in calls
    assert ("message", "Setting up visualization...") in calls
    assert ("render-shank", 0, True) in calls
    assert ("clear-empty",) in calls


def test_load_heavy_data_marks_histology_unavailable_nonfatal() -> None:
    prepared = _fresh_prepared(shank_idx=0, preserve_plot_selection=True)
    presenter, _commands, _queries, calls = _presenter(
        begin_result=prepared,
        commands=FakeCommands(
            begin_result=prepared,
            complete_result=_fresh_completed(
                shank_idx=0,
                histology_result=HistologyDataUnavailable("no histology"),
            ),
        ),
    )

    assert presenter.load_heavy_data()

    assert ("render-shank", 0, True) in calls


def test_probe_selection_returns_false_on_cached_command_failure() -> None:
    presenter, commands, _queries, calls = _presenter(
        commands=FakeCommands(probe_cache_result=Failed("missing cache")),
    )

    assert not presenter.present_cached_probe_selection(
        session_name="rec",
        probe_name="probeA",
        target_shank=0,
    )

    assert commands.probe_cache_calls
    assert calls == []
