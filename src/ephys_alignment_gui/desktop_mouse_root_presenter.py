"""Desktop presentation shell for loading a mouse-root datapackage."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.controller import MouseRootLoaded
from ephys_alignment_gui.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopMouseRootCallbacks:
    """Non-widget side effects for mouse-root loading."""

    busy_context: Callable[..., AbstractContextManager[Any]]
    select_first_session: Callable[[], None]


@dataclass
class DesktopMouseRootPresenter:
    """Coordinate desktop behavior for loading a mouse root."""

    commands: Any
    path_view: Any
    selection_view: Any
    callbacks: DesktopMouseRootCallbacks

    def set_mouse_root(self, mouse_root: Path) -> bool:
        """Load a mouse root and prepare session/probe selection widgets."""
        with self.callbacks.busy_context(
            "Loading datapackage...",
            "Mouse root loaded",
            disable_widgets=self.path_view.mouse_root_widgets(),
        ):
            result = self.commands.set_mouse_root(mouse_root)
            if isinstance(result, Failed):
                logger.error(result.message)
                return False
            assert isinstance(result, MouseRootLoaded)
            if result.root_changed:
                self.commands.clear_histology_context()
            loaded_root = result.mouse_root

            self.path_view.set_mouse_root(loaded_root.root)

            sessions = loaded_root.sessions
            self.selection_view.populate_sessions(sessions)
            self.selection_view.clear_probes()
            self.selection_view.clear_shanks()
            self.selection_view.set_load_data_enabled(False)
            n_probes = sum(
                len(rec_probes) for rec_probes in loaded_root.probes.values()
            )
            logger.info(
                "Loaded mouse %r with %d session(s), %d probe(s)",
                loaded_root.mouse_id,
                len(sessions),
                n_probes,
            )
            if sessions:
                self.selection_view.select_session_index(0)
                self.callbacks.select_first_session()
        return True

    def mouse_root_edited(self) -> bool:
        """Handle direct text edits to the mouse-root line edit."""
        text = self.path_view.mouse_root_text().strip()
        if not text:
            self.selection_view.set_load_data_enabled(False)
            return False
        try:
            path = Path(text)
        except Exception as exc:
            logger.error("Invalid mouse-root path: %s", exc)
            self.selection_view.set_load_data_enabled(False)
            return False
        if not self.set_mouse_root(path):
            self.selection_view.set_load_data_enabled(False)
            return False
        return True
