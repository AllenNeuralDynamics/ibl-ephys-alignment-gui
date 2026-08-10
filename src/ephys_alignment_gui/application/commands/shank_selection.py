"""App-level shank selection and reference-line commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.alignment_events import ShankChanged
from ephys_alignment_gui.application.results import (
    PendingReferenceLinesUpdated,
    ShankSelected,
)
from ephys_alignment_gui.application.workflow import Failed, Ok
from ephys_alignment_gui.controller import AlignmentController
from ephys_alignment_gui.event_bus import EventBus
from ephys_alignment_gui.reference_line_capture import (
    REFERENCE_LINES_NOT_PROVIDED,
    ReferenceLineCapture,
    capture_active_reference_lines,
    capture_outgoing_reference_lines,
)

logger = logging.getLogger(__name__)


@dataclass
class ShankSelectionCommandHandler:
    """Coordinate app-level shank selection and semantic shank events."""

    controller: AlignmentController
    events: EventBus

    def select_shank(
        self,
        shank_idx: int,
        *,
        outgoing_reference_lines: ReferenceLineCapture = REFERENCE_LINES_NOT_PROVIDED,
        source: str = "command",
        preserve_plot_selection: bool | None = None,
    ) -> ShankSelected | Failed:
        """Select a shank as a complete app-level transaction."""
        if (
            self.controller.document.data_loaded
            and outgoing_reference_lines is not REFERENCE_LINES_NOT_PROVIDED
        ):
            capture_result = capture_outgoing_reference_lines(
                self.controller,
                outgoing_reference_lines,
            )
            if isinstance(capture_result, Failed):
                return capture_result

        result = self.controller.select_shank(shank_idx)
        if isinstance(result, ShankSelected):
            self.events.emit(
                ShankChanged(
                    source=source,
                    previous_shank_idx=result.previous_shank_idx,
                    shank_idx=result.shank_idx,
                    previous_key=result.previous_key,
                    active_key=result.selected_key,
                    data_loaded=result.data_loaded,
                    preserve_plot_selection=preserve_plot_selection,
                )
            )
        return result

    def capture_active_reference_lines(
        self,
        reference_lines: tuple[Any, Any] | None,
    ) -> PendingReferenceLinesUpdated | Ok | Failed:
        """Capture active reference-line coordinates as document state."""
        return capture_active_reference_lines(self.controller, reference_lines)
