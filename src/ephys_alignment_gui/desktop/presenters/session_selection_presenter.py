"""Desktop presentation shell for recording/session selection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.application.results.metadata import RecordingSelected
from ephys_alignment_gui.application.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopSessionSelectionCallbacks:
    """Non-widget side effects for session selection."""

    capture_pending_reference_lines: Callable[[], None]
    show_empty_state: Callable[[], None]
    select_first_probe: Callable[[], None]


@dataclass
class DesktopSessionSelectionPresenter:
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

        callbacks.capture_pending_reference_lines()
        evicted = self.app.commands.load.evict_stream_cache()
        if isinstance(evicted, Failed):
            logger.error(evicted.message)
            return False
        result = self.app.commands.metadata.select_recording_metadata(session_name)
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        assert isinstance(result, RecordingSelected)

        callbacks.show_empty_state()
        self.selection_view.populate_probes(result.probes)
        self.selection_view.clear_shanks()
        self.selection_view.set_load_data_enabled(False)
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
