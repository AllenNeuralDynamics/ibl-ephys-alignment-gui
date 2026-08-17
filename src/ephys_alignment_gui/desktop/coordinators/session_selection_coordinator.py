"""Desktop coordination shell for recording/session selection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.application.results.metadata import RecordingSelected
from ephys_alignment_gui.core.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopSessionSelectionCallbacks:
    """Non-widget side effects for session selection."""

    capture_pending_reference_lines: Callable[[], None]
    select_first_probe: Callable[[], None]


@dataclass
class DesktopSessionSelectionCoordinator:
    """Coordinate desktop behavior for selecting a recording/session."""

    app: Any
    selection_view: Any
    callbacks: DesktopSessionSelectionCallbacks

    def session_selected(self, idx: int | None = None) -> bool:
        """Select the current recording and render its probe choices."""
        callbacks = self.callbacks
        if not self.app.queries.workspace.mouse_root_loaded():
            return False

        session_name = self._selected_session_name(idx)
        if not session_name:
            return False

        active_probe = self.app.queries.workspace.active_probe_selection_state()
        if active_probe is not None and active_probe.recording_id == session_name:
            logger.info("Session %s already selected", session_name)
            return True

        callbacks.capture_pending_reference_lines()
        result = self.app.commands.metadata.select_recording_metadata(session_name)
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        assert isinstance(result, RecordingSelected)

        self.selection_view.populate_probes(result.probes)
        self.selection_view.clear_shanks()
        if result.probes:
            self.selection_view.select_probe_index(0)
            callbacks.select_first_probe()
        return True

    def _selected_session_name(self, idx: int | None) -> str:
        if idx is not None:
            session_name = self.selection_view.session_at_index(idx)
            if session_name:
                return session_name
        return self.selection_view.current_session()
