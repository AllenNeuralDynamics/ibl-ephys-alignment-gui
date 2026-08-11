"""Desktop presentation shell for output-root path workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results.path import (
    OutputDirectoryDerived,
    OutputRootSet,
)
from ephys_alignment_gui.application.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass
class DesktopOutputPathPresenter:
    """Coordinate desktop behavior for save/output root paths."""

    commands: Any
    path_view: Any

    def derive_output_directory_from_save_root(self) -> bool:
        """Derive and display the probe output directory if possible."""
        result = self.commands.derive_output_directory()
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        assert isinstance(result, OutputDirectoryDerived)
        if result.output_directory is None:
            return False
        self.display_output_directory(result.output_directory)
        return True

    def set_save_root(self, save_root: Path) -> bool:
        """Set the save root and render the active output path."""
        result = self.commands.set_output_root(save_root)
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        assert isinstance(result, OutputRootSet)
        logger.info("Save root set to: %s", result.output_root)
        if result.output_directory is not None:
            self.display_output_directory(result.output_directory)
        else:
            self.path_view.set_output_root(result.output_root)
        return True

    def output_folder_edited(self) -> bool:
        """Handle direct text edits to the output-root line edit."""
        text = self.path_view.output_root_text().strip()
        if not text:
            return False
        try:
            path = Path(text)
        except Exception as exc:
            logger.error("Invalid output path: %s", exc)
            return False
        return self.set_save_root(path)

    def display_output_directory(self, output_directory: Path | None) -> None:
        """Render the currently derived per-probe output directory."""
        self.path_view.set_output_directory(output_directory)
        if output_directory is not None:
            logger.info("Output dir: %s", output_directory)
