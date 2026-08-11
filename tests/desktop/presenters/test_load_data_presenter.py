"""Tests for desktop load-data presentation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.application.results import (
    CachedEphysDataActivated,
    FreshEphysDataLoaded,
    FreshLoadExecution,
    LoadDataAlreadyActiveResult,
    LoadDataCachedActivated,
    LoadDataFreshCompleted,
    LoadDataFreshPrepared,
    LoadDataFreshRequiredResult,
)
from ephys_alignment_gui.application.results.metadata import ProbeSelected
from ephys_alignment_gui.application.workflow import Failed
from ephys_alignment_gui.core.alignment_events import (
    FreshLoadCompleted,
    HistologyLoadReported,
    LoadDataCancelled,
    LoadDataFailed,
    LoadDataProgressed,
    StreamActivated,
)
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.desktop.presenters.load_data_presenter import (
    DesktopLoadDataCallbacks,
    DesktopLoadDataPresenter,
)
from ephys_alignment_gui.io.load_data_job import (
    LoadDataJobCancelled,
    LoadDataJobCompleted,
    LoadDataJobProgress,
)
from ephys_alignment_gui.runtime.histology_loader import (
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
            active_probe_selection_state=self.active_probe_selection_state,
        )

    def active_shank_selection(self):
        return SimpleNamespace(shank_idx=self.shank_idx)

    def active_probe_selection_state(self):
        return SimpleNamespace(
            recording_id="rec",
            probe_name="probeA",
            shanks=["1/2", "2/2"],
            n_shanks=2,
            output_directory=Path("/tmp/out"),
        )


class FakeCommands:
    def __init__(
        self,
        *,
        events: EventBus | None = None,
        begin_result: Any | None = None,
        job_result: Any | None = None,
        complete_result: Any | None = None,
        probe_cache_result: Any | None = None,
    ) -> None:
        self.events = events
        self.begin_result = begin_result or _fresh_prepared(shank_idx=0)
        self.job_result = job_result or _job_completed()
        self.complete_result = complete_result or _fresh_completed(shank_idx=0)
        self.probe_cache_result = probe_cache_result or LoadDataFreshRequiredResult(
            ("rec", "stream"), 0
        )
        self.begin_calls: list[dict[str, Any]] = []
        self.start_calls: list[LoadDataFreshPrepared] = []
        self.run_calls: list[LoadDataFreshPrepared] = []
        self.activate_calls: list[tuple[LoadDataFreshPrepared, Any]] = []
        self.invocation_calls: list[LoadDataFreshPrepared] = []
        self.cancel_calls: list[str] = []
        self.probe_cache_calls: list[dict[str, Any]] = []
        self._next_load_id = 1
        self.load = self

    def begin_load_data(self, **kwargs):
        self.begin_calls.append(kwargs)
        self._emit_activation_for_result(self.begin_result)
        return self.begin_result

    def start_fresh_load_data(self, prepared: LoadDataFreshPrepared):
        self.start_calls.append(prepared)
        execution = FreshLoadExecution(self._next_load_id, prepared)
        self._next_load_id += 1
        return execution

    def run_fresh_load_data(self, prepared: LoadDataFreshPrepared, **kwargs):
        self.run_calls.append(prepared)
        self._emit_load_data_progress(
            LoadDataJobProgress(
                target=prepared.target,
                phase="ephys",
                status="started",
                message="Loading ephys data...",
            )
        )
        if isinstance(self.job_result, Failed):
            self._emit_load_data_failed(self.job_result.message, prepared)
        elif isinstance(self.job_result, LoadDataJobCancelled):
            self._emit_load_data_cancelled(self.job_result.reason, prepared)
        else:
            self._emit_fresh_load_completed(prepared)
        return self.job_result

    def run_started_fresh_load_data(self, execution: FreshLoadExecution, **kwargs):
        self.run_calls.append(execution.prepared)
        self._emit_load_data_progress(
            LoadDataJobProgress(
                target=execution.prepared.target,
                phase="ephys",
                status="started",
                message="Loading ephys data...",
                load_id=execution.load_id,
            )
        )
        if isinstance(self.job_result, Failed):
            self._emit_load_data_failed(
                self.job_result.message,
                execution.prepared,
                load_id=execution.load_id,
            )
        elif isinstance(self.job_result, LoadDataJobCancelled):
            self._emit_load_data_cancelled(
                self.job_result.reason,
                execution.prepared,
                load_id=execution.load_id,
            )
        else:
            self._emit_fresh_load_completed(
                execution.prepared,
                load_id=execution.load_id,
            )
        return self.job_result

    def fresh_load_job_invocation(self, execution: FreshLoadExecution):
        self.invocation_calls.append(execution.prepared)
        return SimpleNamespace(
            execution=execution,
            request=SimpleNamespace(
                target=execution.prepared.target,
                load_id=execution.load_id,
            ),
            cancel_token=SimpleNamespace(reason=None),
        )

    def run_fresh_load_job(self, invocation, *, progress=None):
        prepared = invocation.execution.prepared
        self.run_calls.append(prepared)
        if callable(progress):
            progress(
                LoadDataJobProgress(
                    target=prepared.target,
                    phase="ephys",
                    status="started",
                    message="Loading ephys data...",
                    load_id=invocation.execution.load_id,
                )
            )
        return self.job_result

    def publish_fresh_load_progress(
        self,
        execution: FreshLoadExecution,
        progress: LoadDataJobProgress,
    ) -> None:
        self._emit_load_data_progress(progress)

    def publish_started_fresh_load_job_result(
        self,
        execution: FreshLoadExecution,
        job_result: Any,
    ):
        prepared = execution.prepared
        if isinstance(job_result, Failed):
            self._emit_load_data_failed(
                job_result.message,
                prepared,
                load_id=execution.load_id,
            )
            return job_result
        if isinstance(job_result, LoadDataJobCancelled):
            self._emit_load_data_cancelled(
                job_result.reason,
                prepared,
                load_id=execution.load_id,
            )
            return job_result
        self._emit_fresh_load_completed(prepared, load_id=execution.load_id)
        return job_result

    def activate_completed_fresh_load_data(
        self,
        prepared: LoadDataFreshPrepared,
        job_result: Any,
    ):
        self.activate_calls.append((prepared, job_result))
        if isinstance(self.complete_result, Failed):
            self._emit_load_data_failed(self.complete_result.message, prepared)
            return self.complete_result
        self._emit_histology_report(self.complete_result, prepared)
        self._emit_activation_for_result(self.complete_result)
        return self.complete_result

    def activate_started_fresh_load_data(
        self,
        execution: FreshLoadExecution,
        job_result: Any,
    ):
        self.activate_calls.append((execution.prepared, job_result))
        if isinstance(self.complete_result, Failed):
            self._emit_load_data_failed(
                self.complete_result.message,
                execution.prepared,
                load_id=execution.load_id,
            )
            return self.complete_result
        self._emit_histology_report(
            self.complete_result,
            execution.prepared,
            load_id=execution.load_id,
        )
        self._emit_activation_for_result(
            self.complete_result,
            load_id=execution.load_id,
        )
        return self.complete_result

    def activate_cached_probe_selection(self, **kwargs):
        self.probe_cache_calls.append(kwargs)
        self._emit_activation_for_result(self.probe_cache_result)
        return self.probe_cache_result

    def cancel_active_fresh_load(self, reason: str):
        self.cancel_calls.append(reason)
        return None

    def _emit_activation_for_result(
        self,
        result: Any,
        *,
        load_id: int | None = None,
    ) -> None:
        if self.events is None:
            return
        if isinstance(result, LoadDataCachedActivated):
            self.events.emit(
                StreamActivated(
                    source="cached",
                    stream_key=result.stream_key,
                    shank_idx=result.activated.shank_idx,
                    active_key=None,
                    preserve_plot_selection=True,
                    load_id=load_id,
                )
            )
        if isinstance(result, LoadDataFreshCompleted):
            self.events.emit(
                StreamActivated(
                    source="fresh",
                    stream_key=result.stream_key,
                    shank_idx=result.ephys.shank_idx,
                    active_key=None,
                    preserve_plot_selection=result.preserve_plot_selection,
                    load_id=load_id,
                )
            )

    def _emit_load_data_progress(self, progress: LoadDataJobProgress) -> None:
        if self.events is None:
            return
        self.events.emit(
            LoadDataProgressed(
                stream_key=progress.target.stream_key,
                shank_idx=progress.target.shank_idx,
                phase=progress.phase,
                status=progress.status,
                message=progress.message,
                load_id=progress.load_id,
            )
        )

    def _emit_fresh_load_completed(
        self,
        prepared: LoadDataFreshPrepared,
        *,
        load_id: int | None = None,
    ) -> None:
        if self.events is None:
            return
        self.events.emit(
            FreshLoadCompleted(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                load_id=load_id,
            )
        )

    def _emit_load_data_failed(
        self,
        message: str,
        prepared: LoadDataFreshPrepared,
        *,
        load_id: int | None = None,
    ) -> None:
        if self.events is None:
            return
        self.events.emit(
            LoadDataFailed(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                message=message,
                load_id=load_id,
            )
        )

    def _emit_load_data_cancelled(
        self,
        reason: str,
        prepared: LoadDataFreshPrepared,
        *,
        load_id: int | None = None,
    ) -> None:
        if self.events is None:
            return
        self.events.emit(
            LoadDataCancelled(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                reason=reason,
                load_id=load_id,
            )
        )

    def _emit_histology_report(
        self,
        result: LoadDataFreshCompleted,
        prepared: LoadDataFreshPrepared,
        *,
        load_id: int | None = None,
    ) -> None:
        if self.events is None:
            return
        if isinstance(result.histology, HistologyDataUnavailable):
            self.events.emit(
                HistologyLoadReported(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    status="unavailable",
                    message=result.histology.message,
                    load_id=load_id,
                )
            )
            return
        self.events.emit(
            HistologyLoadReported(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                status="loaded",
                load_id=load_id,
            )
        )


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


class ImmediateFreshLoadRunner:
    def __init__(self) -> None:
        self.starts: list[Any] = []
        self.cancel_calls: list[str] = []
        self.shutdown_calls: list[tuple[str, int]] = []

    @property
    def is_running(self) -> bool:
        return False

    def start(
        self,
        *,
        execution,
        invocation,
        run_job,
        on_progress,
        on_finished,
    ) -> None:
        self.starts.append(invocation)
        result = run_job(
            invocation,
            progress=lambda event: on_progress(execution, event),
        )
        on_finished(execution, result)

    def cancel(self, reason: str) -> None:
        self.cancel_calls.append(reason)

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        self.shutdown_calls.append((reason, timeout_ms))
        return True


class ManualFreshLoadRunner:
    def __init__(self) -> None:
        self.active = False
        self.cancel_calls: list[str] = []
        self.shutdown_calls: list[tuple[str, int]] = []
        self.shutdown_result = True
        self.start_args: dict[str, Any] | None = None

    @property
    def is_running(self) -> bool:
        return self.active

    def start(
        self,
        *,
        execution,
        invocation,
        run_job,
        on_progress,
        on_finished,
    ) -> None:
        self.active = True
        self.start_args = {
            "execution": execution,
            "invocation": invocation,
            "run_job": run_job,
            "on_progress": on_progress,
            "on_finished": on_finished,
        }

    def finish(self) -> None:
        assert self.start_args is not None
        execution = self.start_args["execution"]
        invocation = self.start_args["invocation"]
        result = self.start_args["run_job"](
            invocation,
            progress=lambda event: self.start_args["on_progress"](
                execution,
                event,
            ),
        )
        self.active = False
        self.start_args["on_finished"](execution, result)

    def cancel(self, reason: str) -> None:
        self.cancel_calls.append(reason)
        self.active = False

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        self.shutdown_calls.append((reason, timeout_ms))
        if self.shutdown_result:
            self.active = False
        return self.shutdown_result


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
        target=SimpleNamespace(
            recording_id="rec",
            probe_name="probeA",
            stream_key=("rec", "stream"),
            shank_idx=shank_idx,
        ),
    )


def _job_completed() -> LoadDataJobCompleted:
    return LoadDataJobCompleted(
        target=SimpleNamespace(label="target"),
        ephys=SimpleNamespace(stream=SimpleNamespace(ephys_dir=Path("/tmp/ephys"))),
        histology=HistologyDataLoaded(),
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
        target=SimpleNamespace(label="target"),
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
    load_runner: Any | None = None,
) -> tuple[DesktopLoadDataPresenter, FakeCommands, FakeQueries, list[tuple]]:
    calls = calls if calls is not None else []
    events = EventBus()
    queries = FakeQueries()
    commands = commands or FakeCommands(begin_result=begin_result)
    commands.events = events
    app = SimpleNamespace(events=events, queries=queries, commands=commands)
    selection_view = FakeSelectionView(calls)
    presenter = DesktopLoadDataPresenter(
        app,
        selection_view,
        _callbacks(calls),
        load_runner=load_runner or ImmediateFreshLoadRunner(),
    )
    presenter.connect_load_events()
    return (
        presenter,
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
    assert commands.run_calls == []
    assert commands.activate_calls == []
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

    assert commands.start_calls == [prepared]
    assert commands.invocation_calls == [prepared]
    assert commands.run_calls == [prepared]
    assert commands.activate_calls == [(prepared, commands.job_result)]
    assert ("prepare-fresh",) in calls
    assert ("message", "Loading ephys data...") in calls
    assert ("message", "Setting up visualization...") in calls
    assert ("render-shank", 0, True) in calls
    assert ("clear-empty",) in calls


def test_load_heavy_data_starts_background_runner_before_completion() -> None:
    prepared = _fresh_prepared(shank_idx=0, preserve_plot_selection=True)
    runner = ManualFreshLoadRunner()
    presenter, commands, _queries, calls = _presenter(
        begin_result=prepared,
        load_runner=runner,
    )

    assert presenter.load_heavy_data()

    assert runner.is_running
    assert commands.start_calls == [prepared]
    assert commands.invocation_calls == [prepared]
    assert commands.run_calls == []
    assert commands.activate_calls == []
    assert ("prepare-fresh",) in calls
    assert ("busy-enter",) in calls
    assert ("render-shank", 0, True) not in calls

    runner.finish()

    assert not runner.is_running
    assert commands.run_calls == [prepared]
    assert commands.activate_calls == [(prepared, commands.job_result)]
    assert ("render-shank", 0, True) in calls
    assert ("busy-exit", None) in calls


def test_load_heavy_data_cancels_active_runner_without_beginning_new_load() -> None:
    runner = ManualFreshLoadRunner()
    runner.active = True
    presenter, commands, _queries, calls = _presenter(load_runner=runner)

    assert not presenter.load_heavy_data()

    assert commands.cancel_calls == ["superseded by a newer load request"]
    assert runner.cancel_calls == ["superseded by a newer load request"]
    assert commands.begin_calls == []
    assert calls == []


def test_shutdown_active_load_cancels_runner_and_closes_busy_context() -> None:
    prepared = _fresh_prepared(shank_idx=0, preserve_plot_selection=True)
    runner = ManualFreshLoadRunner()
    presenter, commands, _queries, calls = _presenter(
        begin_result=prepared,
        load_runner=runner,
    )
    assert presenter.load_heavy_data()

    assert presenter.shutdown_active_load("closing", timeout_ms=123)

    assert commands.cancel_calls == ["closing"]
    assert runner.shutdown_calls == [("closing", 123)]
    assert ("busy-exit", RuntimeError) in calls


def test_shutdown_active_load_keeps_context_open_when_runner_does_not_stop() -> None:
    prepared = _fresh_prepared(shank_idx=0, preserve_plot_selection=True)
    runner = ManualFreshLoadRunner()
    runner.shutdown_result = False
    presenter, commands, _queries, calls = _presenter(
        begin_result=prepared,
        load_runner=runner,
    )
    assert presenter.load_heavy_data()

    assert not presenter.shutdown_active_load("closing", timeout_ms=123)

    assert commands.cancel_calls == ["closing"]
    assert runner.shutdown_calls == [("closing", 123)]
    assert ("busy-exit", RuntimeError) not in calls


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


def test_load_heavy_data_returns_false_when_fresh_job_is_cancelled() -> None:
    prepared = _fresh_prepared(shank_idx=0, preserve_plot_selection=True)
    presenter, commands, _queries, calls = _presenter(
        begin_result=prepared,
        commands=FakeCommands(
            begin_result=prepared,
            job_result=LoadDataJobCancelled(
                target=prepared.target,
                reason="new probe selected",
            ),
        ),
    )

    assert presenter.load_heavy_data()

    assert commands.run_calls == [prepared]
    assert commands.activate_calls == []
    assert ("render-shank", 0, True) not in calls


def test_load_events_ignore_stale_load_ids() -> None:
    presenter, commands, _queries, calls = _presenter()
    presenter._active_load_context = FakeBusyContext(calls, "loading")
    presenter._active_load_id = 2

    assert commands.events is not None
    commands.events.emit(
        LoadDataProgressed(
            stream_key=("rec", "stream"),
            shank_idx=0,
            phase="ephys",
            status="started",
            message="stale progress",
            load_id=1,
        )
    )
    commands.events.emit(
        StreamActivated(
            source="fresh",
            stream_key=("rec", "stream"),
            shank_idx=0,
            active_key=None,
            preserve_plot_selection=True,
            load_id=1,
        )
    )

    assert ("message", "stale progress") not in calls
    assert ("render-shank", 0, True) not in calls


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
