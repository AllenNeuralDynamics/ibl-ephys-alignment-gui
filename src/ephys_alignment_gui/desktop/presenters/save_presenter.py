"""Desktop presentation shell for save and QC commands."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.application.results import VisitedAlignmentOutputsSaved
from ephys_alignment_gui.application.workflow import Blocked, Failed, Ok, Requirement
from ephys_alignment_gui.core.alignment_events import SaveCompleted, SaveFailed
from ephys_alignment_gui.core.event_bus import EventSubscription

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
class DesktopSavePresenter:
    """Coordinate desktop save and QC button behavior."""

    commands: Any
    events: Any
    callbacks: DesktopSaveCallbacks

    def connect_save_events(self) -> list[EventSubscription]:
        """Subscribe desktop save presentation to semantic save events."""
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
            "Channel locations saved to results folder for %d visited alignment(s)",
            event.saved_count,
        )

    def on_save_failed(self, event: SaveFailed) -> None:
        """Log save failure reported by the app layer."""
        logger.error(event.message)

    def save_alignment_outputs(self) -> bool:
        """Save visited alignment outputs, prompting for output if needed."""
        save_ready = self.commands.can_save_alignment_output()
        if isinstance(save_ready, Blocked):
            if not self.callbacks.ensure_output_directory(save_ready.first):
                return False
            save_ready = self.commands.can_save_alignment_output()

        if not isinstance(save_ready, Ok):
            if isinstance(save_ready, Blocked):
                self.callbacks.log_requirement(save_ready.first)
            return False

        with self.callbacks.busy_context(
            "Saving...",
            "Saved successfully",
            disable_widgets=self.callbacks.complete_button(),
        ):
            result = self.commands.save_visited_alignment_outputs(
                use_docdb=self.callbacks.use_docdb(),
            )
            if isinstance(result, Blocked):
                self.callbacks.log_requirement(result.first)
                return False
            if isinstance(result, Failed):
                return False
            assert isinstance(result, VisitedAlignmentOutputsSaved)
        return True

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
