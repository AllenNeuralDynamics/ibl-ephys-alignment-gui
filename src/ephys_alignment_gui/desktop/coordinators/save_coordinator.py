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
from ephys_alignment_gui.application.workflow import Blocked, Failed, Ok, Requirement
from ephys_alignment_gui.core.alignment_events import SaveCompleted, SaveFailed
from ephys_alignment_gui.core.event_bus import EventSubscription
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

    def connect_save_events(self) -> list[EventSubscription]:
        """Subscribe desktop save coordination to semantic save events."""
        return [
            self.events.subscribe(SaveCompleted, self.on_save_completed),
            self.events.subscribe(SaveFailed, self.on_save_failed),
        ]

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

    def on_save_failed(self, event: SaveFailed) -> None:
        """Log save failure reported by the app layer."""
        logger.error(event.message)

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
        return True

    def _start_rehydration_then_save(
        self,
        plan: SaveRuntimeRehydrationPlan,
        *,
        use_docdb: bool,
    ) -> bool:
        """Start background runtime reload, then save from the completion callback."""
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
            self._close_save_context(RuntimeError(f"Save cancelled: {reason}"))
        return stopped

    def _on_rehydration_progress(self, event: LoadDataJobProgress) -> None:
        """Update save progress while missing runtimes are reloaded."""
        if self._active_save_context is not None:
            self._active_save_context.update_message(event.message)

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
