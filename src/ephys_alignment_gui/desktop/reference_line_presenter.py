"""Desktop presentation for reference-line document capture."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass
class DesktopReferenceLinePresenter:
    """Translate reference-line display positions into app commands."""

    app: Any
    reference_line_display: Any

    def capture_pending_reference_lines(self) -> bool:
        """Capture current reference-line positions into document state."""
        result = self.app.commands.shanks.capture_active_reference_lines(
            self.reference_line_display.positions()
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        return True
