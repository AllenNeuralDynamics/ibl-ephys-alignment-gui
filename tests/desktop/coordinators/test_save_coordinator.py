"""Tests for desktop save/QC coordination."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results import EditedAlignmentOutputsSaved
from ephys_alignment_gui.application.results.alignment_persistence import (
    AlignmentOutputsSaved,
)
from ephys_alignment_gui.application.save_runtime_rehydration import (
    SaveRuntimeRehydrated,
    SaveRuntimeRehydrationPlan,
)
from ephys_alignment_gui.application.workflow import (
    CHOOSE_OUTPUT_FOLDER,
    OUTPUT_REQUIRED,
    Blocked,
    Failed,
    Ok,
    Requirement,
)
from ephys_alignment_gui.core.alignment_events import (
    SaveCompleted,
    SaveDocDbStatus,
    SaveFailed,
    SaveProgressStarted,
    SaveProgressUpdated,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.desktop.coordinators.save_coordinator import (
    DesktopSaveCallbacks,
    DesktopSaveCoordinator,
)
from ephys_alignment_gui.io.load_data_job import LoadDataJobProgress
from ephys_alignment_gui.services.alignment_repository import SavedAlignmentOutputs


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


class FakeButton:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls
        self.text_value = "Save"
        self.tooltip_value = ""

    def text(self) -> str:
        return self.text_value

    def setText(self, value: str) -> None:
        self.text_value = value
        self.calls.append(("button-text", value))

    def toolTip(self) -> str:
        return self.tooltip_value

    def setToolTip(self, value: str) -> None:
        self.tooltip_value = value
        self.calls.append(("button-tooltip", value))


class FakeSaveProgressDialog:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls
        self.cancel_callback = None

    def set_cancel_callback(self, callback) -> None:
        self.cancel_callback = callback
        self.calls.append(("progress-cancel-callback",))

    def show_started(self, targets, *, message, cancel_enabled=False) -> None:
        self.calls.append(
            ("progress-started", tuple(targets), message, cancel_enabled)
        )

    def update_progress(
        self,
        *,
        key,
        phase_label,
        status_label,
        completed,
        total,
        message,
    ) -> None:
        self.calls.append(
            (
                "progress-update",
                key,
                phase_label,
                status_label,
                completed,
                total,
                message,
            )
        )

    def set_cancel_enabled(self, enabled: bool) -> None:
        self.calls.append(("progress-cancel-enabled", enabled))

    def show_finished(self, message: str, *, success: bool) -> None:
        self.calls.append(("progress-finished", message, success))

    def close_dialog(self) -> None:
        self.calls.append(("progress-close",))


class ManualSaveRehydrationRunner:
    def __init__(self, *, result: Any | None = None) -> None:
        self.result = result
        self.active = False
        self.start_calls: list[SaveRuntimeRehydrationPlan] = []
        self.cancel_calls: list[str] = []
        self.shutdown_calls: list[tuple[str, int]] = []
        self.shutdown_result = True
        self._plan = None
        self._run_job = None
        self._on_progress = None
        self._on_finished = None

    @property
    def is_running(self) -> bool:
        return self.active

    def start(self, *, plan, run_job, on_progress, on_finished) -> None:
        self.active = True
        self.start_calls.append(plan)
        self._plan = plan
        self._run_job = run_job
        self._on_progress = on_progress
        self._on_finished = on_finished

    def finish(self, result: Any | None = None) -> None:
        assert self._plan is not None
        assert self._run_job is not None
        assert self._on_finished is not None
        terminal = result if result is not None else self.result
        if terminal is None:
            terminal = self._run_job(self._plan, progress=self._on_progress)
        self.active = False
        self._on_finished(terminal)

    def cancel(self, reason: str) -> None:
        self.cancel_calls.append(reason)
        self.active = False

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        self.shutdown_calls.append((reason, timeout_ms))
        if self.shutdown_result:
            self.active = False
        return self.shutdown_result


class FakeCommands:
    def __init__(
        self,
        *,
        events: EventBus | None = None,
        ready_results: list[Any] | None = None,
        save_result: Any | None = None,
        prepare_result: Any | None = None,
        rehydration_result: Any | None = None,
    ) -> None:
        self.events = events
        self.ready_results = ready_results or [Ok()]
        self.save_result = save_result or _saved_result()
        self.prepare_result = prepare_result or Ok()
        self.rehydration_result = rehydration_result or SaveRuntimeRehydrated(1)
        self.ready_calls = 0
        self.save_calls: list[dict[str, Any]] = []
        self.save_rehydrate_missing: list[bool] = []
        self.prepare_calls = 0
        self.rehydrate_calls: list[SaveRuntimeRehydrationPlan] = []
        self.publish_calls: list[Any] = []

    def can_save_alignment_output(self):
        result = self.ready_results[min(self.ready_calls, len(self.ready_results) - 1)]
        self.ready_calls += 1
        return result

    def prepare_save_runtime_rehydration(self):
        self.prepare_calls += 1
        return self.prepare_result

    def run_save_runtime_rehydration(self, plan, *, progress=None, cancel_token=None):
        self.rehydrate_calls.append(plan)
        if callable(progress):
            progress(
                LoadDataJobProgress(
                    target=plan.dependencies[0].load_target,
                    phase="ephys",
                    status="started",
                    message="Reloading runtime data...",
                )
            )
        return self.rehydration_result

    def publish_save_runtime_rehydration_result(self, result):
        self.publish_calls.append(result)
        if isinstance(result, SaveRuntimeRehydrated):
            return Ok()
        failed = result if isinstance(result, Failed) else Failed(str(result))
        self._emit_save_event(failed)
        return failed

    def save_edited_alignment_outputs(
        self,
        *,
        use_docdb: bool,
        rehydrate_missing: bool = True,
    ):
        self.save_calls.append({"use_docdb": use_docdb})
        self.save_rehydrate_missing.append(rehydrate_missing)
        self._emit_save_event(self.save_result)
        return self.save_result

    def _emit_save_event(self, result: Any) -> None:
        if self.events is None:
            return
        if isinstance(result, Failed):
            self.events.emit(SaveFailed(message=result.message))
            return
        if isinstance(result, EditedAlignmentOutputsSaved):
            self.events.emit(
                SaveCompleted(
                    saved_count=result.saved_count,
                    active_choices=(
                        tuple(result.active_choices)
                        if result.active_choices is not None
                        else None
                    ),
                    docdb_statuses=tuple(
                        SaveDocDbStatus(
                            probe_name=saved.saved.docdb_probe_name,
                            error=saved.saved.docdb_error,
                        )
                        for saved in result.saved_outputs.values()
                        if saved.saved.docdb_probe_name is not None
                    ),
                )
            )


def _requirement() -> Requirement:
    return Requirement(
        OUTPUT_REQUIRED,
        "Choose an output folder before saving.",
        action=CHOOSE_OUTPUT_FOLDER,
    )


def _saved_result(
    *,
    docdb_probe_name: str | None = "probeA_0",
    docdb_error: str | None = None,
) -> EditedAlignmentOutputsSaved:
    key = AlignmentKey("rec", "stream", 1)
    return EditedAlignmentOutputsSaved(
        saved_count=1,
        saved_outputs={
            key: AlignmentOutputsSaved(
                saved=SavedAlignmentOutputs(
                    channel_results_path=Path("/results/channels.json"),
                    previous_alignments_path=Path("/results/alignments.json"),
                    ccf_channel_results_path=Path("/results/ccf.json"),
                    docdb_probe_name=docdb_probe_name,
                    docdb_error=docdb_error,
                ),
                previous_alignments={},
            )
        },
        active_choices=["saved", "original"],
    )


def _rehydration_plan() -> SaveRuntimeRehydrationPlan:
    key = AlignmentKey("rec", "stream", 0)
    target = type(
        "Target",
        (),
        {
            "recording_id": "rec",
            "probe_name": "probeA",
            "stream_key": ("rec", "stream"),
            "shank_idx": 0,
        },
    )()
    dependency = type("Dependency", (), {"key": key, "load_target": target})()
    return SaveRuntimeRehydrationPlan((dependency,))


def _coordinator(
    *,
    commands: FakeCommands | None = None,
    ensure_output: bool = True,
    use_docdb: bool = True,
    histology_available: bool = True,
    ephys_qc: str = "Pass",
    selected_descriptions: list[str] | None = None,
    rehydration_runner: Any | None = None,
    complete_button: Any | None = None,
) -> tuple[DesktopSaveCoordinator, FakeCommands, list[tuple]]:
    calls: list[tuple] = []
    events = EventBus()
    commands = commands or FakeCommands()
    commands.events = events
    progress_dialog = FakeSaveProgressDialog(calls)
    button = complete_button if complete_button is not None else "complete-button"
    coordinator = DesktopSaveCoordinator(
        commands=commands,
        events=events,
        callbacks=DesktopSaveCallbacks(
            ensure_output_directory=lambda requirement: (
                calls.append(("ensure-output", requirement)) or ensure_output
            ),
            log_requirement=lambda requirement: calls.append(
                ("requirement", requirement)
            ),
            use_docdb=lambda: use_docdb,
            render_alignment_choices=lambda choices: calls.append(("choices", choices)),
            busy_context=lambda *args, **kwargs: FakeBusyContext(
                calls,
                *args,
                **kwargs,
            ),
            complete_button=lambda: button,
            save_progress_dialog=lambda: progress_dialog,
            histology_available=lambda: histology_available,
            open_qc_dialog=lambda: calls.append(("open-qc",)),
            ephys_qc=lambda: ephys_qc,
            selected_qc_descriptions=lambda: selected_descriptions or [],
            warning=lambda title, message: calls.append(("warning", title, message)),
        ),
        rehydration_runner=rehydration_runner or ManualSaveRehydrationRunner(),
    )
    coordinator.connect_save_events()
    return coordinator, commands, calls


def test_save_prompts_for_output_then_saves() -> None:
    blocked = Blocked((_requirement(),))
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(ready_results=[blocked, Ok()]),
    )

    assert coordinator.save_alignment_outputs()

    assert commands.ready_calls == 2
    assert commands.prepare_calls == 1
    assert commands.save_calls == [{"use_docdb": True}]
    assert commands.save_rehydrate_missing == [False]
    assert calls == [
        ("ensure-output", blocked.first),
        (
            "busy",
            ("Saving...", "Saved successfully"),
            {"disable_widgets": "complete-button"},
        ),
        ("busy-enter",),
        ("choices", ["saved", "original"]),
        ("busy-exit", None),
    ]


def test_save_returns_false_when_output_prompt_is_cancelled() -> None:
    blocked = Blocked((_requirement(),))
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(ready_results=[blocked]),
        ensure_output=False,
    )

    assert not coordinator.save_alignment_outputs()

    assert commands.save_calls == []
    assert calls == [("ensure-output", blocked.first)]


def test_save_logs_failed_command() -> None:
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(save_result=Failed("save failed")),
    )

    assert not coordinator.save_alignment_outputs()

    assert commands.save_calls == [{"use_docdb": True}]
    assert commands.save_rehydrate_missing == [False]
    assert calls == [
        (
            "busy",
            ("Saving...", "Saved successfully"),
            {"disable_widgets": "complete-button"},
        ),
        ("busy-enter",),
        ("busy-exit", None),
    ]


def test_save_rehydrates_missing_runtime_in_background_before_saving() -> None:
    plan = _rehydration_plan()
    runner = ManualSaveRehydrationRunner()
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(prepare_result=plan),
        rehydration_runner=runner,
    )

    assert coordinator.save_alignment_outputs()

    assert runner.is_running
    assert runner.start_calls == [plan]
    assert commands.save_calls == []
    assert calls == [
        ("progress-cancel-callback",),
        (
            "progress-started",
            (AlignmentKey("rec", "stream", 0),),
            "Reloading runtime data for 1 edited alignment before saving...",
            True,
        ),
        (
            "busy",
            ("Reloading data needed for save...", "Saved successfully"),
            {"disable_widgets": "complete-button"},
        ),
        ("busy-enter",),
    ]

    runner.finish()

    assert commands.rehydrate_calls == [plan]
    assert isinstance(commands.publish_calls[0], SaveRuntimeRehydrated)
    assert commands.save_calls == [{"use_docdb": True}]
    assert commands.save_rehydrate_missing == [False]
    assert ("message", "Reloading runtime data...") in calls
    assert (
        "progress-update",
        AlignmentKey("rec", "stream", 0),
        "Reloading runtime",
        "Loading",
        0,
        1,
        "Reloading runtime data...",
    ) in calls
    assert ("message", "Saving output files...") in calls
    assert ("progress-cancel-enabled", False) in calls
    assert ("choices", ["saved", "original"]) in calls
    assert ("progress-finished", "Saved 1 edited alignment.", True) in calls
    assert ("busy-exit", None) in calls


def test_save_rehydration_failure_does_not_save() -> None:
    plan = _rehydration_plan()
    runner = ManualSaveRehydrationRunner(result=Failed("reload failed"))
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(prepare_result=plan),
        rehydration_runner=runner,
    )

    assert coordinator.save_alignment_outputs()
    runner.finish()

    assert commands.save_calls == []
    assert commands.publish_calls == [Failed("reload failed")]
    assert ("busy-exit", RuntimeError) in calls


def test_shutdown_active_save_settles_rehydration() -> None:
    runner = ManualSaveRehydrationRunner()
    runner.active = True
    coordinator, _commands, calls = _coordinator(rehydration_runner=runner)
    coordinator._open_save_context(
        "Reloading data needed for save...",
        "Saved successfully",
        disable_widgets="complete-button",
    )

    assert coordinator.shutdown_active_save("closing", timeout_ms=123)

    assert runner.shutdown_calls == [("closing", 123)]
    assert ("busy-exit", RuntimeError) in calls


def test_save_progress_events_update_dialog_and_button() -> None:
    button_calls: list[tuple] = []
    button = FakeButton(button_calls)
    coordinator, commands, calls = _coordinator(complete_button=button)
    key = AlignmentKey("rec", "stream", 2)

    commands.events.emit(
        SaveProgressStarted(
            targets=(key,),
            message="Saving 1 edited alignment...",
        )
    )
    commands.events.emit(
        SaveProgressUpdated(
            key=key,
            phase="writing_files",
            status="completed",
            completed=1,
            total=1,
            message="Wrote output files.",
        )
    )

    assert coordinator is not None
    assert ("button-text", "Saving 0/1") in button_calls
    assert ("button-text", "Saving 1/1") in button_calls
    assert (
        "progress-started",
        (key,),
        "Saving 1 edited alignment...",
        False,
    ) in calls
    assert (
        "progress-update",
        key,
        "Writing files",
        "Done",
        1,
        1,
        "Wrote output files.",
    ) in calls


def test_cancel_active_save_requests_rehydration_cancel() -> None:
    plan = _rehydration_plan()
    runner = ManualSaveRehydrationRunner()
    coordinator, _commands, calls = _coordinator(rehydration_runner=runner)

    assert coordinator._start_rehydration_then_save(plan, use_docdb=True)
    assert coordinator.cancel_active_save()

    assert runner.cancel_calls == ["cancelled by user"]
    assert (
        "progress-update",
        None,
        "Reloading runtime",
        "Cancelling",
        0,
        1,
        "Cancelling save runtime reload...",
    ) in calls


def test_display_qc_options_opens_dialog_when_histology_available() -> None:
    coordinator, _commands, calls = _coordinator()

    assert coordinator.display_qc_options()

    assert calls == [("open-qc",)]


def test_display_qc_options_noops_without_histology() -> None:
    coordinator, _commands, calls = _coordinator(histology_available=False)

    assert not coordinator.display_qc_options()

    assert calls == []


def test_qc_button_requires_description_for_failing_qc() -> None:
    coordinator, commands, calls = _coordinator(ephys_qc="Fail")

    assert not coordinator.qc_button_clicked()

    assert commands.save_calls == []
    assert calls == [
        ("warning", "Status", "You must select a reason for qc choice"),
        ("open-qc",),
    ]


def test_qc_button_saves_when_description_is_selected() -> None:
    coordinator, commands, calls = _coordinator(
        ephys_qc="Fail",
        selected_descriptions=["noise"],
    )

    assert coordinator.qc_button_clicked()

    assert commands.save_calls == [{"use_docdb": True}]
    assert commands.save_rehydrate_missing == [False]
    assert ("choices", ["saved", "original"]) in calls
