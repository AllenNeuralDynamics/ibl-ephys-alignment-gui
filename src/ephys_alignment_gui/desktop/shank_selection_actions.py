"""Desktop action adapter for shank dropdown selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.app_results import ShankSelected
from ephys_alignment_gui.workflow import Failed

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

        result = self.app.commands.shanks.select_shank(
            shank_idx,
            outgoing_reference_lines=self.reference_line_display.positions(),
            source="dropdown",
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        if not isinstance(result, ShankSelected):
            return False

        logger.info("Shank %s selected (index %s)", shank_id, result.shank_idx)
        return True
