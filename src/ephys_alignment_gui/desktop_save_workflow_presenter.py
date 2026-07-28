"""Desktop presentation shell for save and QC workflows."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.app import VisitedAlignmentOutputsSaved
from ephys_alignment_gui.workflow import Blocked, Failed, Ok, Requirement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopSaveWorkflowCallbacks:
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
class DesktopSaveWorkflowPresenter:
    """Coordinate desktop save and QC button behavior."""

    commands: Any
    callbacks: DesktopSaveWorkflowCallbacks

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
                logger.error(result.message)
                return False
            assert isinstance(result, VisitedAlignmentOutputsSaved)
            self._render_saved_outputs(result)
        return True

    def _render_saved_outputs(self, result: VisitedAlignmentOutputsSaved) -> None:
        """Render save command results in desktop UI/logging."""
        for saved in result.saved_outputs.values():
            if saved.saved.docdb_probe_name is None:
                continue
            if saved.saved.docdb_error is not None:
                logger.error(
                    "Failed to write to DocDB with error %s. "
                    "Output saved to results folder",
                    saved.saved.docdb_error,
                )
            else:
                logger.info(
                    "Channels locations saved, and ccf coordinates saved for %s",
                    saved.saved.docdb_probe_name,
                )

        if result.active_choices is not None:
            self.callbacks.render_alignment_choices(result.active_choices)
        logger.info(
            "Channel locations saved to results folder for %d visited alignment(s)",
            result.saved_count,
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
