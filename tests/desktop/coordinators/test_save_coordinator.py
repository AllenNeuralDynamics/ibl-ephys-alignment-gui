"""Tests for desktop save/QC coordination."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.alignment_save_job import (
    AlignmentSaveJobCancelled,
    AlignmentSaveJobCompleted,
    PreparedAlignmentSave,
)
from ephys_alignment_gui.application.results import EditedAlignmentOutputsSaved
from ephys_alignment_gui.application.results.alignment_persistence import (
    AlignmentOutputsSaved,
)
from ephys_alignment_gui.core.alignment_events import (
    SaveCancelled,
    SaveCompleted,
    SaveDocDbStatus,
    SaveFailed,
    SaveProgressStarted,
    SaveProgressUpdated,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.workflow import (
    CHOOSE_OUTPUT_FOLDER,
    OUTPUT_REQUIRED,
    Blocked,
    Failed,
    Ok,
    Requirement,
)
from ephys_alignment_gui.desktop.coordinators.save_coordinator import (
    DesktopSaveCallbacks,
    DesktopSaveCoordinator,
)
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

    def show_started(
        self,
        targets,
        *,
        message,
        cancel_enabled=False,
        scope_message=None,
    ) -> None:
        self.calls.append(("progress-started", tuple(targets), message, cancel_enabled))
        if scope_message is not None:
            self.calls.append(("progress-scope", scope_message))

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

    def show_cancelled(self, message: str) -> None:
        self.calls.append(("progress-cancelled", message))

    def close_dialog(self) -> None:
        self.calls.append(("progress-close",))


class ManualAlignmentSaveRunner:
    def __init__(
        self,
        *,
        result: Any | None = None,
        auto_finish: bool = True,
    ) -> None:
        self.result = result
        self.auto_finish = auto_finish
        self.active = False
        self.start_calls: list[PreparedAlignmentSave] = []
        self.cancel_calls: list[str] = []
        self.shutdown_calls: list[tuple[str, int]] = []
        self.shutdown_result = True
        self._prepared = None
        self._run_job = None
        self._on_progress = None
        self._on_finished = None

    @property
    def is_running(self) -> bool:
        return self.active

    def start(self, *, prepared, run_job, on_progress, on_finished) -> None:
        self.active = True
        self.start_calls.append(prepared)
        self._prepared = prepared
        self._run_job = run_job
        self._on_progress = on_progress
        self._on_finished = on_finished
        if self.auto_finish:
            self.finish()

    def finish(self, result: Any | None = None) -> None:
        assert self._prepared is not None
        assert self._run_job is not None
        assert self._on_finished is not None
        terminal = result if result is not None else self.result
        if terminal is None:
            terminal = self._run_job(self._prepared, progress=self._on_progress)
        self.active = False
        self._on_finished(self._prepared, terminal)

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
    ) -> None:
        self.events = events
        self.ready_results = ready_results or [Ok()]
        self.save_result = save_result or _saved_result()
        self.prepare_result = prepare_result or Ok()
        self.ready_calls = 0
        self.save_calls: list[dict[str, Any]] = []
        self.prepare_calls = 0
        self.prepared_saves: list[PreparedAlignmentSave] = []
        self.publish_save_calls: list[Any] = []

    def can_save_alignment_output(self):
        result = self.ready_results[min(self.ready_calls, len(self.ready_results) - 1)]
        self.ready_calls += 1
        return result

    def save_edited_alignment_outputs(
        self,
        *,
        use_docdb: bool,
    ):
        self.save_calls.append({"use_docdb": use_docdb})
        self._emit_save_event(self.save_result)
        return self.save_result

    def prepare_edited_alignment_save(
        self,
        *,
        use_docdb: bool,
    ):
        self.prepare_calls += 1
        prepared = PreparedAlignmentSave((), use_docdb=use_docdb)
        self.prepared_saves.append(prepared)
        return prepared

    def run_prepared_alignment_save(
        self,
        prepared,
        *,
        progress=None,
        cancel_token=None,
    ):
        self.save_calls.append({"use_docdb": prepared.use_docdb})
        if isinstance(self.save_result, Failed):
            return self.save_result
        return AlignmentSaveJobCompleted(
            saved_outputs=dict(self.save_result.saved_outputs)
        )

    def publish_prepared_alignment_save_result(self, prepared, result):
        self.publish_save_calls.append(result)
        if isinstance(result, AlignmentSaveJobCancelled):
            if self.events is not None:
                self.events.emit(
                    SaveCancelled(
                        reason=result.reason,
                        message=f"Save cancelled: {result.reason}",
                    )
                )
            return result
        if isinstance(result, Failed):
            self._emit_save_event(result)
            return result
        published = EditedAlignmentOutputsSaved(
            saved_count=len(result.saved_outputs),
            saved_outputs=result.saved_outputs,
            active_choices=["saved", "original"],
        )
        self._emit_save_event(published)
        return published

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
                    metadata_path=Path("/results/metadata.json"),
                    docdb_probe_name=docdb_probe_name,
                    docdb_error=docdb_error,
                ),
                previous_alignments={},
            )
        },
        active_choices=["saved", "original"],
    )


def _coordinator(
    *,
    commands: FakeCommands | None = None,
    ensure_output: bool = True,
    use_docdb: bool = True,
    histology_available: bool = True,
    ephys_qc: str = "Pass",
    selected_descriptions: list[str] | None = None,
    save_runner: Any | None = None,
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
            save_blocking_widgets=lambda: [],
        ),
        save_runner=save_runner or ManualAlignmentSaveRunner(),
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
    assert calls == [
        ("ensure-output", blocked.first),
        ("progress-cancel-callback",),
        ("progress-started", (), "Preparing save...", False),
        (
            "busy",
            ("Saving...", "Saved successfully"),
            {"disable_widgets": ["complete-button"]},
        ),
        ("busy-enter",),
        ("progress-cancel-callback",),
        ("progress-cancel-enabled", True),
        ("choices", ["saved", "original"]),
        ("progress-finished", "Saved 1 alignment output.", True),
        ("progress-close",),
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

    assert coordinator.save_alignment_outputs()

    assert commands.save_calls == [{"use_docdb": True}]
    assert calls == [
        ("progress-cancel-callback",),
        ("progress-started", (), "Preparing save...", False),
        (
            "busy",
            ("Saving...", "Saved successfully"),
            {"disable_widgets": ["complete-button"]},
        ),
        ("busy-enter",),
        ("progress-cancel-callback",),
        ("progress-cancel-enabled", True),
        ("progress-finished", "save failed", False),
        ("busy-exit", RuntimeError),
    ]


def test_save_progress_events_update_dialog_and_button() -> None:
    button_calls: list[tuple] = []
    button = FakeButton(button_calls)
    coordinator, commands, calls = _coordinator(complete_button=button)
    key = AlignmentKey("rec", "stream", 2)

    commands.events.emit(
        SaveProgressStarted(
            targets=(key,),
            message="Saving 1 alignment output...",
            edited_count=1,
            unchanged_count=0,
        )
    )
    commands.events.emit(
        SaveProgressUpdated(
            key=None,
            phase="building_outputs",
            status="started",
            completed=0,
            total=1,
            message="Batching CCF transform points for 1 alignment output...",
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
    assert ("button-text", "Transforming CCF...") in button_calls
    assert ("button-text", "Saving 1/1") in button_calls
    assert (
        "progress-started",
        (key,),
        "Saving 1 alignment output...",
        False,
    ) in calls
    assert (
        "progress-scope",
        "Saving 1 visited shank: 1 edited alignment, 0 unchanged/original alignments.",
    ) in calls
    assert (
        "progress-update",
        None,
        "Transforming CCF",
        "Running",
        0,
        1,
        "Batching CCF transform points for 1 alignment output...",
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


def test_cancel_active_save_requests_final_save_cancel() -> None:
    runner = ManualAlignmentSaveRunner(auto_finish=False)
    coordinator, _commands, calls = _coordinator(save_runner=runner)

    assert coordinator.save_alignment_outputs()
    assert coordinator.cancel_active_save()

    assert runner.cancel_calls == ["cancelled by user"]
    assert ("progress-cancel-enabled", True) in calls
    assert ("progress-cancel-enabled", False) in calls
    assert (
        "progress-update",
        None,
        "Saving output",
        "Cancelling",
        0,
        1,
        "Cancelling alignment save...",
    ) in calls


def test_cancelled_final_save_closes_busy_context_without_saved_event() -> None:
    runner = ManualAlignmentSaveRunner(auto_finish=False)
    coordinator, commands, calls = _coordinator(save_runner=runner)

    assert coordinator.save_alignment_outputs()
    runner.finish(AlignmentSaveJobCancelled(reason="cancelled by user"))

    assert commands.publish_save_calls == [
        AlignmentSaveJobCancelled(reason="cancelled by user")
    ]
    assert ("progress-cancelled", "Save cancelled: cancelled by user") in calls
    assert ("choices", ["saved", "original"]) not in calls
    assert ("busy-exit", RuntimeError) in calls


def test_async_shutdown_treats_late_save_success_as_cancelled() -> None:
    runner = ManualAlignmentSaveRunner(auto_finish=False)
    coordinator, commands, calls = _coordinator(save_runner=runner)

    assert coordinator.save_alignment_outputs()
    assert coordinator.request_async_shutdown("closing")
    runner.finish(
        AlignmentSaveJobCompleted(saved_outputs=dict(_saved_result().saved_outputs))
    )

    assert commands.publish_save_calls == [AlignmentSaveJobCancelled(reason="closing")]
    assert ("progress-cancelled", "Save cancelled: closing") in calls
    assert ("choices", ["saved", "original"]) not in calls
    assert ("busy-exit", RuntimeError) in calls


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
    assert ("choices", ["saved", "original"]) in calls
