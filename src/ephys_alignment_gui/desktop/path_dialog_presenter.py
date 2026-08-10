"""Desktop presentation shell for path-selection dialogs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.settings import (
    INPUT_ROOT_ENV_VAR,
    input_root_from_environment,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopPathDialogCallbacks:
    """State queries and path commands used by desktop folder dialogs."""

    active_mouse_root: Callable[[], Path | None]
    set_mouse_root: Callable[[Path], bool]
    active_output_root: Callable[[], Path | None]
    set_save_root: Callable[[Path], bool]


@dataclass
class DesktopPathDialogPresenter:
    """Coordinate mouse-root and output-root desktop folder dialogs."""

    folder_dialog: Any
    callbacks: DesktopPathDialogCallbacks
    input_root_provider: Callable[[], Path | None] = input_root_from_environment

    def select_mouse_root(self) -> bool:
        """Prompt for a mouse-root directory and load it if selected."""
        folder = self.folder_dialog.select_existing_directory(
            "Select Mouse Root",
            directory=self.mouse_root_start_dir(),
        )
        if folder is None:
            return False
        return self.callbacks.set_mouse_root(folder)

    def mouse_root_start_dir(self) -> Path | None:
        """Return the preferred start directory for mouse-root browsing."""
        active_root = self.callbacks.active_mouse_root()
        if active_root is not None:
            return active_root

        input_root = self.input_root_provider()
        if input_root is None:
            return None
        if input_root.is_dir():
            return input_root
        logger.warning(
            "Ignoring %s because it is not a directory: %s",
            INPUT_ROOT_ENV_VAR,
            input_root,
        )
        return None

    def select_output_root(self) -> bool:
        """Prompt for an output-root directory and set it if selected."""
        folder = self.folder_dialog.select_existing_directory(
            "Select Save Root",
            directory=self.callbacks.active_output_root(),
        )
        if folder is None:
            return False
        return self.callbacks.set_save_root(folder)
