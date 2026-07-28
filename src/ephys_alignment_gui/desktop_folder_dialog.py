"""Desktop wrapper for existing-directory dialogs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt5 import QtWidgets


@dataclass
class DesktopFolderDialog:
    """Own Qt folder-dialog calls for desktop presenters."""

    parent: Any = None
    get_existing_directory: Callable[..., str] = (
        QtWidgets.QFileDialog.getExistingDirectory
    )

    def select_existing_directory(
        self,
        title: str,
        *,
        directory: Path | str | None = None,
    ) -> Path | None:
        """Prompt for an existing directory and return a selected path."""
        selected = self.get_existing_directory(
            self.parent,
            title,
            directory="" if directory is None else str(directory),
        )
        if not selected:
            return None
        return Path(selected)

    def select_existing_directory_text(
        self,
        title: str,
        *,
        directory: Path | str | None = None,
    ) -> str:
        """Prompt for an existing directory and return Qt-style text."""
        selected = self.select_existing_directory(title, directory=directory)
        if selected is None:
            return ""
        return str(selected)
