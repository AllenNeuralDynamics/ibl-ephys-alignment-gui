"""Desktop action adapter for shank dropdown selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.application.foreground_operations import (
    ForegroundOperation,
    ForegroundOperationConflict,
)
from ephys_alignment_gui.application.results import ShankSelected
from ephys_alignment_gui.core.timing import start_timing
from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.desktop.coordinators.foreground_operation import (
    acquire_foreground_operation,
)

logger = logging.getLogger(__name__)


@dataclass
class DesktopShankSelectionActions:
    """Adapt the shank dropdown selection into an app-level command."""

    app: Any
    selection_view: Any
    reference_line_display: Any

    def shank_selected(self) -> bool:
        """Select the shank currently shown by the desktop combobox."""
        shank_idx = self.selection_view.current_shank_index()
        if shank_idx is None:
            logger.error("Cannot select shank: invalid shank label")
            return False

        shank_id = shank_idx + 1
        selection = self.app.queries.workspace.active_shank_selection()
        if shank_idx == selection.shank_idx:
            logger.info("Shank %s already selected", shank_id)
            return True

        timer = start_timing(
            "shank_dropdown_switch",
            shank_idx=shank_idx,
            shank_id=shank_id,
        )
        lease = acquire_foreground_operation(
            getattr(self.app, "foreground_operations", None),
            ForegroundOperation.SELECTION_ACTIVATION,
        )
        if isinstance(lease, ForegroundOperationConflict):
            logger.error(lease.message)
            timer.finish("blocked", message=lease.message)
            return False
        with lease:
            with timer.activate():
                with timer.step("select_shank_command"):
                    result = self.app.commands.shanks.select_shank(
                        shank_idx,
                        outgoing_reference_lines=(
                            self.reference_line_display.positions()
                        ),
                        source="dropdown",
                    )
                if isinstance(result, Failed):
                    logger.error(result.message)
                    timer.finish("failed", message=result.message)
                    return False
                if not isinstance(result, ShankSelected):
                    timer.finish("failed", result_type=type(result).__name__)
                    return False

        logger.info("Shank %s selected (index %s)", shank_id, result.shank_idx)
        timer.finish("completed")
        return True
