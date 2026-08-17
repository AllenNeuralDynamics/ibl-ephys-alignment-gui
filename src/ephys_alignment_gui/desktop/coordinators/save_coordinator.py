"""Desktop coordination shell for save and QC commands."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from ephys_alignment_gui.application.alignment_save_job import (
    AlignmentSaveJobCancelled,
    AlignmentSaveJobCompleted,
)
from ephys_alignment_gui.application.results import EditedAlignmentOutputsSaved
from ephys_alignment_gui.core.alignment_events import (
    SaveCancelled,
    SaveCompleted,
    SaveFailed,
    SaveProgressStarted,
    SaveProgressUpdated,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.event_bus import EventSubscription
from ephys_alignment_gui.core.workflow import Blocked, Failed, Ok, Requirement
from ephys_alignment_gui.desktop.workers.alignment_save_runner import (
    AlignmentSaveJobResult,
    AlignmentSaveRunner,
    QtAlignmentSaveRunner,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopSaveCallbacks:
    """Desktop side effects for saving alignment outputs."""

    ensure_output_directory: Callable[[Requirement], bool]
    log_requirement: Callable[[Requirement], None]
    use_docdb: Callable[[], bool]
    render_alignment_choices: Callable[[list[str]], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    complete_button: Callable[[], Any]
    save_progress_dialog: Callable[[], Any]
    histology_available: Callable[[], bool]
    open_qc_dialog: Callable[[], None]
    ephys_qc: Callable[[], str]
    selected_qc_descriptions: Callable[[], list[str]]
    warning: Callable[[str, str], Any]
    save_blocking_widgets: Callable[[], list[Any]] = lambda: []


@dataclass
class DesktopSaveCoordinator:
    """Coordinate desktop save and QC button behavior."""

    commands: Any
    events: Any
    callbacks: DesktopSaveCallbacks
    save_runner: AlignmentSaveRunner = field(default_factory=QtAlignmentSaveRunner)
    _active_save_context: Any | None = field(default=None, init=False, repr=False)
    _active_save_context_manager: Any | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _progress_dialog: Any | None = field(default=None, init=False, repr=False)
    _save_button_original_text: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _save_button_original_tooltip: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _save_targets: tuple[AlignmentKey, ...] = field(
        default=(),
        init=False,
        repr=False,
    )
    _save_cancel_requested_reason: str | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def connect_save_events(self) -> list[EventSubscription]:
        """Subscribe desktop save coordination to semantic save events."""
        return [
            self.events.subscribe(SaveProgressStarted, self.on_save_progress_started),
            self.events.subscribe(SaveProgressUpdated, self.on_save_progress_updated),
            self.events.subscribe(SaveCompleted, self.on_save_completed),
            self.events.subscribe(SaveCancelled, self.on_save_cancelled),
            self.events.subscribe(SaveFailed, self.on_save_failed),
        ]

    def on_save_progress_started(self, event: SaveProgressStarted) -> None:
        """Open or reset the desktop save progress dialog."""
        self._set_save_button_progress(
            f"Saving 0/{max(len(event.targets), 1)}",
            event.message,
        )
        dialog = self._save_progress_dialog()
        dialog.set_cancel_callback(lambda: self.cancel_active_save("cancelled by user"))
        dialog.show_started(
            event.targets,
            message=event.message,
            cancel_enabled=False,
            scope_message=_save_scope_message(event),
        )

    def on_save_progress_updated(self, event: SaveProgressUpdated) -> None:
        """Render one semantic save progress update."""
        if self._active_save_context is not None:
            self._active_save_context.update_message(event.message)
        phase_label = _phase_label(event.phase)
        status_label = _status_label(event.status)
        if event.key is None and event.status in {"started", "running"}:
            button_text = f"{phase_label}..."
        else:
            button_text = f"Saving {event.completed}/{max(event.total, 1)}"
        self._set_save_button_progress(button_text, event.message)
        dialog = self._save_progress_dialog()
        dialog.update_progress(
            key=event.key,
            phase_label=phase_label,
            status_label=status_label,
            completed=event.completed,
            total=event.total,
            message=event.message,
        )

    def on_save_completed(self, event: SaveCompleted) -> None:
        """Render save completion in desktop UI/logging."""
        for status in event.docdb_statuses:
            if status.error is not None:
                logger.error(
                    "Failed to write to DocDB with error %s. "
                    "Output saved to results folder",
                    status.error,
                )
            else:
                logger.info(
                    "Channels locations saved, and ccf coordinates saved for %s",
                    status.probe_name,
                )

        if event.active_choices is not None:
            self.callbacks.render_alignment_choices(list(event.active_choices))
        logger.info(
            "Channel locations saved to results folder for %d alignment output(s)",
            event.saved_count,
        )
        if self._progress_dialog is not None:
            self._progress_dialog.show_finished(
                f"Saved {event.saved_count} alignment output"
                f"{'' if event.saved_count == 1 else 's'}.",
                success=True,
            )
            self._progress_dialog.close_dialog()
        if self._save_ui_active():
            self._set_save_button_progress("Saved", "Saved successfully")
            self._restore_save_button_state()

    def on_save_failed(self, event: SaveFailed) -> None:
        """Log save failure reported by the app layer."""
        logger.error(event.message)
        if self._progress_dialog is not None:
            self._progress_dialog.show_finished(event.message, success=False)
        if self._save_ui_active():
            self._set_save_button_progress("Save failed", event.message)
            self._restore_save_button_state()

    def on_save_cancelled(self, event: SaveCancelled) -> None:
        """Render user-requested save cancellation."""
        logger.info(event.message)
        if self._progress_dialog is not None:
            self._progress_dialog.show_cancelled(event.message)
            self._progress_dialog.close_dialog()
        if self._save_ui_active():
            self._set_save_button_progress("Cancelled", event.message)
            self._restore_save_button_state()

    def save_alignment_outputs(self) -> bool:
        """Save alignment outputs, prompting for output if needed."""
        save_ready = self.commands.can_save_alignment_output()
        if isinstance(save_ready, Blocked):
            if not self.callbacks.ensure_output_directory(save_ready.first):
                return False
            save_ready = self.commands.can_save_alignment_output()

        if not isinstance(save_ready, Ok):
            if isinstance(save_ready, Blocked):
                self.callbacks.log_requirement(save_ready.first)
            return False

        if self.save_runner.is_running:
            logger.info("Save request ignored because save work is active")
            return False

        use_docdb = self.callbacks.use_docdb()
        return self._start_prepared_save(use_docdb=use_docdb)

    def _start_prepared_save(
        self,
        *,
        use_docdb: bool,
        open_context: bool = True,
    ) -> bool:
        """Prepare a save job and run output generation in the background."""
        self._save_cancel_requested_reason = None
        self._show_save_preparing()
        self._set_save_button_progress("Saving...", "Saving alignment outputs")
        if open_context and self._active_save_context_manager is None:
            self._open_save_context(
                "Saving...",
                "Saved successfully",
                disable_widgets=self._save_disable_widgets(),
            )
        elif self._active_save_context is not None:
            self._active_save_context.update_message("Preparing save...")

        try:
            prepared = self.commands.prepare_edited_alignment_save(
                use_docdb=use_docdb,
            )
        except Exception as exc:
            self._close_save_context(exc)
            logger.exception("Failed to prepare edited-alignment save")
            return False

        if isinstance(prepared, Blocked):
            self.callbacks.log_requirement(prepared.first)
            self._close_save_context(RuntimeError(prepared.first.message))
            return False
        if isinstance(prepared, Failed):
            self._close_save_context(RuntimeError(prepared.message))
            return False

        self._save_targets = prepared.target_keys
        if self._progress_dialog is not None:
            self._progress_dialog.set_cancel_callback(
                lambda: self.cancel_active_save("cancelled by user")
            )
            self._progress_dialog.set_cancel_enabled(True)

        try:
            self.save_runner.start(
                prepared=prepared,
                run_job=self.commands.run_prepared_alignment_save,
                on_progress=self._on_save_job_progress,
                on_finished=self._on_save_job_finished,
            )
        except Exception as exc:
            failed = Failed(f"Failed to start edited-alignment save: {exc}")
            self.commands.publish_prepared_alignment_save_result(prepared, failed)
            self._close_save_context(exc)
            logger.exception("Failed to start edited-alignment save")
            return False
        return True

    def _show_save_preparing(self) -> None:
        """Clear stale progress rows before save preparation emits events."""
        dialog = self._save_progress_dialog()
        dialog.set_cancel_callback(None)
        dialog.show_started(
            (),
            message="Preparing save...",
            cancel_enabled=False,
        )

    def shutdown_active_save(
        self,
        reason: str = "application closing",
        *,
        timeout_ms: int = 5000,
    ) -> bool:
        """Settle active save workers before desktop teardown."""
        stopped = True
        if self.save_runner.is_running:
            self._save_cancel_requested_reason = reason
            stopped = (
                self.save_runner.shutdown(reason, timeout_ms=timeout_ms) and stopped
            )
        if stopped and (
            self._progress_dialog is not None
            or self._active_save_context_manager is not None
        ):
            if self._progress_dialog is not None:
                self._progress_dialog.close_dialog()
            self._close_save_context(RuntimeError(f"Save cancelled: {reason}"))
        return stopped

    def has_active_work(self) -> bool:
        """Return whether save work or save UI state is still settling."""
        return (
            self.save_runner.is_running or self._active_save_context_manager is not None
        )

    def request_async_shutdown(self, reason: str = "application closing") -> bool:
        """Request cancellation for active save work without waiting."""
        return self.cancel_active_save(reason)

    def cancel_active_save(self, reason: str = "cancelled by user") -> bool:
        """Request cooperative cancellation for active save work."""
        if self.save_runner.is_running:
            self._save_cancel_requested_reason = reason
            self.save_runner.cancel(reason)
            if self._progress_dialog is not None:
                self._progress_dialog.set_cancel_enabled(False)
                self._progress_dialog.update_progress(
                    key=None,
                    phase_label="Saving output",
                    status_label="Cancelling",
                    completed=0,
                    total=max(len(self._save_targets), 1),
                    message="Cancelling alignment save...",
                )
            self._set_save_button_progress("Cancelling...", "Cancelling save")
            return True
        return False

    def _on_save_job_progress(self, event: Any) -> None:
        """Publish save-job progress from the GUI thread."""
        self.events.emit(event)

    def _on_save_job_finished(
        self,
        prepared: Any,
        result: AlignmentSaveJobResult,
    ) -> None:
        """Publish final save result and close desktop busy state."""
        if self._save_cancel_requested_reason is not None and isinstance(
            result,
            AlignmentSaveJobCompleted,
        ):
            result = AlignmentSaveJobCancelled(
                reason=self._save_cancel_requested_reason
            )
        try:
            published = self.commands.publish_prepared_alignment_save_result(
                prepared,
                result,
            )
            if isinstance(published, Failed):
                self._close_save_context(RuntimeError(published.message))
                return
            if isinstance(published, AlignmentSaveJobCancelled):
                self._close_save_context(
                    RuntimeError(f"Save cancelled: {published.reason}")
                )
                return
            assert isinstance(published, EditedAlignmentOutputsSaved)
            self._close_save_context()
        except Exception as exc:
            logger.exception("Failed to publish edited-alignment save result")
            self._close_save_context(exc)
        finally:
            self._save_cancel_requested_reason = None

    def _open_save_context(self, *args: Any, **kwargs: Any) -> None:
        """Enter and hold the desktop busy context for async save work."""
        manager = self.callbacks.busy_context(*args, **kwargs)
        self._active_save_context_manager = manager
        self._active_save_context = manager.__enter__()

    def _close_save_context(self, exc: BaseException | None = None) -> None:
        """Exit the active desktop busy context, if one is open."""
        manager = self._active_save_context_manager
        try:
            if manager is None:
                return
            if exc is None:
                manager.__exit__(None, None, None)
            else:
                manager.__exit__(type(exc), exc, exc.__traceback__)
        finally:
            self._active_save_context = None
            self._active_save_context_manager = None
            self._save_targets = ()
            self._restore_save_button_state()

    def _save_progress_dialog(self) -> Any:
        """Return the active save progress dialog, creating it if needed."""
        if self._progress_dialog is None:
            self._progress_dialog = self.callbacks.save_progress_dialog()
        return self._progress_dialog

    def _set_save_button_progress(self, text: str, tooltip: str) -> None:
        """Update the Save button label/tooltip while preserving originals."""
        button = self.callbacks.complete_button()
        if self._save_button_original_text is None and hasattr(button, "text"):
            self._save_button_original_text = button.text()
        if self._save_button_original_tooltip is None and hasattr(button, "toolTip"):
            self._save_button_original_tooltip = button.toolTip()
        if hasattr(button, "setText"):
            button.setText(text)
        if hasattr(button, "setToolTip"):
            button.setToolTip(tooltip)

    def _restore_save_button_state(self) -> None:
        """Restore the Save button label/tooltip after active save work ends."""
        button = self.callbacks.complete_button()
        if self._save_button_original_text is not None and hasattr(button, "setText"):
            button.setText(self._save_button_original_text)
        if self._save_button_original_tooltip is not None and hasattr(
            button, "setToolTip"
        ):
            button.setToolTip(self._save_button_original_tooltip)
        self._save_button_original_text = None
        self._save_button_original_tooltip = None

    def _save_disable_widgets(self) -> list[Any]:
        """Return widgets disabled while save work owns document state."""
        widgets: list[Any] = []
        button = self.callbacks.complete_button()
        if button is not None:
            widgets.append(button)
        widgets.extend(self.callbacks.save_blocking_widgets())
        return widgets

    def _save_ui_active(self) -> bool:
        """Return whether this coordinator has active save UI state to mutate."""
        return (
            self._progress_dialog is not None
            or self._active_save_context is not None
            or self._save_button_original_text is not None
            or self._save_button_original_tooltip is not None
        )

    def display_qc_options(self) -> bool:
        """Open the QC dialog if histology is available."""
        if not self.callbacks.histology_available():
            return False
        self.callbacks.open_qc_dialog()
        return True

    def qc_button_clicked(self) -> bool:
        """Validate QC fields and save local/DocDB alignment output."""
        if not self.callbacks.histology_available():
            return False

        ephys_qc = self.callbacks.ephys_qc()
        ephys_desc = self.callbacks.selected_qc_descriptions()
        if ephys_qc != "Pass" and not ephys_desc:
            self.callbacks.warning(
                "Status",
                "You must select a reason for qc choice",
            )
            self.display_qc_options()
            return False

        logger.warning(
            "Alyx QC upload is unavailable without ONE; saving local/DocDB "
            "alignment output instead."
        )
        return self.save_alignment_outputs()


def _phase_label(phase: str) -> str:
    return {
        "preparing": "Preparing",
        "building_outputs": "Transforming CCF",
        "writing_files": "Writing files",
    }.get(phase, str(phase).replace("_", " ").title())


def _status_label(status: str) -> str:
    return {
        "started": "Running",
        "running": "Running",
        "warning": "Warning",
        "completed": "Done",
        "cancelled": "Cancelled",
    }.get(status, str(status).title())


def _save_scope_message(event: SaveProgressStarted) -> str | None:
    total = len(event.targets)
    scoped_total = event.edited_count + event.unchanged_count
    if total == 0 or scoped_total == 0:
        return None
    shank_label = "shank" if total == 1 else "shanks"
    edited_label = (
        "edited alignment" if event.edited_count == 1 else "edited alignments"
    )
    unchanged_label = (
        "unchanged/original alignment"
        if event.unchanged_count == 1
        else "unchanged/original alignments"
    )
    return (
        f"Saving {total} visited {shank_label}: "
        f"{event.edited_count} {edited_label}, "
        f"{event.unchanged_count} {unchanged_label}."
    )
