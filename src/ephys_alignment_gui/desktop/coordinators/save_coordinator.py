"""Desktop coordination shell for save and QC commands."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from ephys_alignment_gui.application.results import EditedAlignmentOutputsSaved
from ephys_alignment_gui.application.save_runtime_rehydration import (
    SaveRuntimeRehydrationPlan,
)
from ephys_alignment_gui.core.alignment_events import (
    SaveCompleted,
    SaveFailed,
    SaveProgressStarted,
    SaveProgressUpdated,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.event_bus import EventSubscription
from ephys_alignment_gui.core.workflow import Blocked, Failed, Ok, Requirement
from ephys_alignment_gui.desktop.workers.save_rehydration_runner import (
    QtSaveRuntimeRehydrationRunner,
    SaveRuntimeRehydrationResult,
    SaveRuntimeRehydrationRunner,
)
from ephys_alignment_gui.io.load_data_job import LoadDataJobProgress

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


@dataclass
class DesktopSaveCoordinator:
    """Coordinate desktop save and QC button behavior."""

    commands: Any
    events: Any
    callbacks: DesktopSaveCallbacks
    rehydration_runner: SaveRuntimeRehydrationRunner = field(
        default_factory=QtSaveRuntimeRehydrationRunner
    )
    _active_save_context: Any | None = field(default=None, init=False, repr=False)
    _active_save_context_manager: Any | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _pending_use_docdb: bool | None = field(default=None, init=False, repr=False)
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
    _rehydration_targets: tuple[AlignmentKey, ...] = field(
        default=(),
        init=False,
        repr=False,
    )

    def connect_save_events(self) -> list[EventSubscription]:
        """Subscribe desktop save coordination to semantic save events."""
        return [
            self.events.subscribe(SaveProgressStarted, self.on_save_progress_started),
            self.events.subscribe(SaveProgressUpdated, self.on_save_progress_updated),
            self.events.subscribe(SaveCompleted, self.on_save_completed),
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
        )

    def on_save_progress_updated(self, event: SaveProgressUpdated) -> None:
        """Render one semantic save progress update."""
        if self._active_save_context is not None:
            self._active_save_context.update_message(event.message)
        self._set_save_button_progress(
            f"Saving {event.completed}/{max(event.total, 1)}",
            event.message,
        )
        dialog = self._save_progress_dialog()
        dialog.update_progress(
            key=event.key,
            phase_label=_phase_label(event.phase),
            status_label=_status_label(event.status),
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
            "Channel locations saved to results folder for %d edited alignment(s)",
            event.saved_count,
        )
        if self._progress_dialog is not None:
            self._progress_dialog.show_finished(
                f"Saved {event.saved_count} edited alignment"
                f"{'' if event.saved_count == 1 else 's'}.",
                success=True,
            )
        if self._save_ui_active():
            self._set_save_button_progress("Saved", "Saved successfully")

    def on_save_failed(self, event: SaveFailed) -> None:
        """Log save failure reported by the app layer."""
        logger.error(event.message)
        if self._progress_dialog is not None:
            self._progress_dialog.show_finished(event.message, success=False)
        if self._save_ui_active():
            self._set_save_button_progress("Save failed", event.message)

    def save_alignment_outputs(self) -> bool:
        """Save edited alignment outputs, prompting for output if needed."""
        save_ready = self.commands.can_save_alignment_output()
        if isinstance(save_ready, Blocked):
            if not self.callbacks.ensure_output_directory(save_ready.first):
                return False
            save_ready = self.commands.can_save_alignment_output()

        if not isinstance(save_ready, Ok):
            if isinstance(save_ready, Blocked):
                self.callbacks.log_requirement(save_ready.first)
            return False

        if self.rehydration_runner.is_running:
            logger.info("Save request ignored because runtime reload is active")
            return False

        use_docdb = self.callbacks.use_docdb()
        rehydration = self.commands.prepare_save_runtime_rehydration()
        if isinstance(rehydration, Failed):
            self.commands.publish_save_runtime_rehydration_result(rehydration)
            return False
        if isinstance(rehydration, SaveRuntimeRehydrationPlan):
            return self._start_rehydration_then_save(rehydration, use_docdb=use_docdb)

        assert isinstance(rehydration, Ok)
        return self._save_now(use_docdb=use_docdb)

    def _save_now(self, *, use_docdb: bool) -> bool:
        """Run the final save transaction on the GUI thread."""
        self._set_save_button_progress("Saving...", "Saving edited alignments")
        try:
            with self.callbacks.busy_context(
                "Saving...",
                "Saved successfully",
                disable_widgets=self.callbacks.complete_button(),
            ):
                result = self.commands.save_edited_alignment_outputs(
                    use_docdb=use_docdb,
                    rehydrate_missing=False,
                )
                if isinstance(result, Blocked):
                    self.callbacks.log_requirement(result.first)
                    return False
                if isinstance(result, Failed):
                    return False
                assert isinstance(result, EditedAlignmentOutputsSaved)
        finally:
            self._restore_save_button_state()
        return True

    def _start_rehydration_then_save(
        self,
        plan: SaveRuntimeRehydrationPlan,
        *,
        use_docdb: bool,
    ) -> bool:
        """Start background runtime reload, then save from the completion callback."""
        self._show_rehydration_started(plan)
        self._set_save_button_progress(
            f"Reloading 0/{max(len(plan.dependencies), 1)}",
            "Reloading runtime data needed for save",
        )
        self._open_save_context(
            "Reloading data needed for save...",
            "Saved successfully",
            disable_widgets=self.callbacks.complete_button(),
        )
        self._pending_use_docdb = use_docdb
        try:
            self.rehydration_runner.start(
                plan=plan,
                run_job=self.commands.run_save_runtime_rehydration,
                on_progress=self._on_rehydration_progress,
                on_finished=self._on_rehydration_finished,
            )
        except Exception as exc:
            self._pending_use_docdb = None
            failed = Failed(f"Failed to start save-runtime reload: {exc}")
            self.commands.publish_save_runtime_rehydration_result(failed)
            self._close_save_context(exc)
            logger.exception("Failed to start save-runtime reload")
            return False
        return True

    def shutdown_active_save(
        self,
        reason: str = "application closing",
        *,
        timeout_ms: int = 5000,
    ) -> bool:
        """Cancel and settle active save-runtime reload before desktop teardown."""
        if not self.rehydration_runner.is_running:
            return True
        stopped = self.rehydration_runner.shutdown(reason, timeout_ms=timeout_ms)
        if stopped:
            self._pending_use_docdb = None
            if self._progress_dialog is not None:
                self._progress_dialog.close_dialog()
            self._close_save_context(RuntimeError(f"Save cancelled: {reason}"))
        return stopped

    def cancel_active_save(self, reason: str = "cancelled by user") -> bool:
        """Request cooperative cancellation for the active save-runtime reload."""
        if not self.rehydration_runner.is_running:
            return False
        self.rehydration_runner.cancel(reason)
        if self._progress_dialog is not None:
            self._progress_dialog.update_progress(
                key=None,
                phase_label="Reloading runtime",
                status_label="Cancelling",
                completed=0,
                total=max(len(self._rehydration_targets), 1),
                message="Cancelling save runtime reload...",
            )
        self._set_save_button_progress("Cancelling...", "Cancelling save")
        return True

    def _on_rehydration_progress(self, event: LoadDataJobProgress) -> None:
        """Update save progress while missing runtimes are reloaded."""
        if self._active_save_context is not None:
            self._active_save_context.update_message(event.message)
        key = self._key_from_load_target(getattr(event, "target", None))
        total = len(self._rehydration_targets)
        index = _target_index(self._rehydration_targets, key)
        completed = index if event.status == "completed" else max(index - 1, 0)
        self._set_save_button_progress(
            f"Reloading {completed}/{max(total, 1)}",
            event.message,
        )
        if self._progress_dialog is not None:
            self._progress_dialog.update_progress(
                key=key,
                phase_label="Reloading runtime",
                status_label=("Loaded" if event.status == "completed" else "Loading"),
                completed=completed,
                total=total,
                message=event.message,
            )

    def _on_rehydration_finished(
        self,
        result: SaveRuntimeRehydrationResult,
    ) -> None:
        """Publish reload completion and run the final save transaction."""
        published = self.commands.publish_save_runtime_rehydration_result(result)
        if isinstance(published, Failed):
            self._pending_use_docdb = None
            self._close_save_context(RuntimeError(published.message))
            return

        use_docdb = bool(self._pending_use_docdb)
        self._pending_use_docdb = None
        if self._active_save_context is not None:
            self._active_save_context.update_message("Saving output files...")
        if self._progress_dialog is not None:
            self._progress_dialog.set_cancel_enabled(False)
        save_result = self.commands.save_edited_alignment_outputs(
            use_docdb=use_docdb,
            rehydrate_missing=False,
        )
        if isinstance(save_result, Blocked):
            self.callbacks.log_requirement(save_result.first)
            self._close_save_context(RuntimeError(save_result.first.message))
            return
        if isinstance(save_result, Failed):
            self._close_save_context(RuntimeError(save_result.message))
            return
        assert isinstance(save_result, EditedAlignmentOutputsSaved)
        self._close_save_context()

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
            self._rehydration_targets = ()
            self._restore_save_button_state()

    def _show_rehydration_started(self, plan: SaveRuntimeRehydrationPlan) -> None:
        """Show cancellable runtime reload progress before final saving."""
        targets = tuple(dependency.key for dependency in plan.dependencies)
        self._rehydration_targets = targets
        dialog = self._save_progress_dialog()
        dialog.set_cancel_callback(lambda: self.cancel_active_save("cancelled by user"))
        dialog.show_started(
            targets,
            message=(
                "Reloading runtime data for "
                f"{len(targets)} edited alignment"
                f"{'' if len(targets) == 1 else 's'} before saving..."
            ),
            cancel_enabled=True,
        )

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

    def _save_ui_active(self) -> bool:
        """Return whether this coordinator has active save UI state to mutate."""
        return (
            self._progress_dialog is not None
            or self._active_save_context is not None
            or self._save_button_original_text is not None
            or self._save_button_original_tooltip is not None
        )

    @staticmethod
    def _key_from_load_target(target: Any | None) -> AlignmentKey | None:
        if target is None:
            return None
        stream_key = getattr(target, "stream_key", None)
        shank_idx = getattr(target, "shank_idx", None)
        if not isinstance(stream_key, tuple) or len(stream_key) != 2:
            return None
        if shank_idx is None:
            return None
        return AlignmentKey(
            recording_id=str(stream_key[0]),
            ephys_collection=str(stream_key[1]),
            shank_idx=int(shank_idx),
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
        "rehydrating": "Reloading runtime",
        "building_outputs": "Transforming CCF",
        "writing_files": "Writing files",
    }.get(phase, str(phase).replace("_", " ").title())


def _status_label(status: str) -> str:
    return {
        "started": "Running",
        "running": "Running",
        "completed": "Done",
    }.get(status, str(status).title())


def _target_index(targets: tuple[AlignmentKey, ...], key: AlignmentKey | None) -> int:
    if key is None:
        return 1 if targets else 0
    try:
        return targets.index(key) + 1
    except ValueError:
        return 1 if targets else 0
