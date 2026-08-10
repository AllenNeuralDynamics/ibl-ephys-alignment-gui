"""Desktop action adapter for alignment edit button workflows."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.app_results import AlignmentEditApplied
from ephys_alignment_gui.workflow import Failed

logger = logging.getLogger(__name__)

NUDGE_STEP_M = 50 / 1e6


@dataclass(frozen=True)
class DesktopAlignmentEditActionCallbacks:
    """Desktop-only state needed to start alignment edit commands."""

    histology_available: Callable[[], bool]
    capture_pending_reference_lines: Callable[[], bool]
    tip_position_um: Callable[[], float | None]


@dataclass
class DesktopAlignmentEditActions:
    """Adapt desktop button actions into app-level alignment edit commands."""

    commands: Any
    callbacks: DesktopAlignmentEditActionCallbacks

    def fit_button_pressed(self) -> bool:
        """Capture current reference lines and fit the active alignment."""
        if not self.callbacks.histology_available():
            return False
        if not self.callbacks.capture_pending_reference_lines():
            return False
        result = self.commands.fit_active_alignment_from_pending_reference_lines()
        return self._edit_applied(result)

    def offset_button_pressed(self, *, track_shift_m: float = 0.0) -> bool:
        """Offset the active alignment from the rendered probe-tip line."""
        if not self.callbacks.histology_available():
            return False
        tip_position_um = self.callbacks.tip_position_um()
        if tip_position_um is None:
            logger.error("Cannot offset alignment: probe tip line is not rendered")
            return False
        result = self.commands.offset_active_alignment_from_tip(
            tip_position_um=tip_position_um,
            track_shift_m=track_shift_m,
        )
        return self._edit_applied(result)

    def movedown_button_pressed(self) -> bool:
        """Nudge the active alignment down by one fixed step."""
        return self._nudge_active_alignment(track_shift_m=-NUDGE_STEP_M)

    def moveup_button_pressed(self) -> bool:
        """Nudge the active alignment up by one fixed step."""
        return self._nudge_active_alignment(track_shift_m=NUDGE_STEP_M)

    def next_button_pressed(self) -> bool:
        """Move the active edit cursor forward."""
        if not self.callbacks.histology_available():
            return False
        return self._edit_applied(self.commands.go_next_alignment())

    def prev_button_pressed(self) -> bool:
        """Move the active edit cursor backward."""
        if not self.callbacks.histology_available():
            return False
        return self._edit_applied(self.commands.go_previous_alignment())

    def reset_button_pressed(self) -> bool:
        """Reset the active alignment to its initialized geometry."""
        if not self.callbacks.histology_available():
            return False
        return self._edit_applied(self.commands.reset_active_alignment_to_initial())

    def _nudge_active_alignment(self, *, track_shift_m: float) -> bool:
        if not self.callbacks.histology_available():
            return False
        tip_position_um = self.callbacks.tip_position_um()
        if tip_position_um is None:
            logger.error("Cannot offset alignment: probe tip line is not rendered")
            return False
        result = self.commands.nudge_active_alignment_from_tip(
            tip_position_um=tip_position_um,
            track_shift_m=track_shift_m,
        )
        return self._edit_applied(result)

    @staticmethod
    def _edit_applied(result: Any) -> bool:
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        return isinstance(result, AlignmentEditApplied)
