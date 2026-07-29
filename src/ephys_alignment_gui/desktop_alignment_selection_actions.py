"""Desktop action adapter for previous/original alignment dropdown selection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.controller import PreviousAlignmentSelected
from ephys_alignment_gui.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopAlignmentSelectionCallbacks:
    """Desktop side effects needed after an alignment choice changes."""

    render_loaded_shank_histology: Callable[[], bool]


@dataclass
class DesktopAlignmentSelectionActions:
    """Adapt the alignment dropdown selection into app-level commands."""

    app: Any
    callbacks: DesktopAlignmentSelectionCallbacks

    def alignment_selected(self, idx: int) -> bool:
        """Select a previous/original alignment and refresh loaded histology."""
        logger.info("Alignment index %s selected", idx)

        selected = self.app.commands.select_previous_alignment(idx)
        if isinstance(selected, Failed):
            logger.error(selected.message)
            return False
        if not isinstance(selected, PreviousAlignmentSelected):
            return False

        selection = self.app.queries.active_shank_selection()
        if not selection.data_loaded:
            logger.info("Data not loaded yet, alignment params updated")
            return True

        prepared = self.app.commands.prepare_loaded_shank(
            selection.shank_idx,
            select_default_alignment_if_empty=False,
        )
        if isinstance(prepared, Failed):
            logger.error(prepared.message)
            return False
        if not prepared.histology_available:
            return True

        rendered = self.callbacks.render_loaded_shank_histology()
        logger.info("Alignment change complete")
        return rendered
