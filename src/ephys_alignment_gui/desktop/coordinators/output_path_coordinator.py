"""Desktop coordination shell for output-root path workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.foreground_operations import (
    ForegroundOperation,
    ForegroundOperationConflict,
)
from ephys_alignment_gui.application.results.path import (
    OutputDirectoryDerived,
    OutputRootSet,
)
from ephys_alignment_gui.core.alignment_events import (
    OutputDirectoryChanged,
    OutputRootChanged,
)
from ephys_alignment_gui.core.event_bus import EventSubscription
from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.desktop.coordinators.foreground_operation import (
    acquire_foreground_operation,
)

logger = logging.getLogger(__name__)


@dataclass
class DesktopOutputPathCoordinator:
    """Coordinate desktop behavior for save/output root paths."""

    commands: Any
    events: Any
    path_view: Any
    foreground_operations: Any | None = None

    def connect_path_events(self) -> list[EventSubscription]:
        """Subscribe desktop path coordination to semantic path events."""
        return [
            self.events.subscribe(OutputRootChanged, self.on_output_root_changed),
            self.events.subscribe(
                OutputDirectoryChanged,
                self.on_output_directory_changed,
            ),
        ]

    def on_output_root_changed(self, event: OutputRootChanged) -> None:
        """Render path state after the output root changes."""
        logger.info("Save root set to: %s", event.output_root)
        self.render_output_paths(event.output_root, event.output_directory)

    def on_output_directory_changed(self, event: OutputDirectoryChanged) -> None:
        """Render path state after the per-probe output directory changes."""
        self.render_output_paths(event.output_root, event.output_directory)

    def derive_output_directory_from_save_root(self) -> bool:
        """Derive the probe output directory if possible."""
        result = self.commands.derive_output_directory()
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        assert isinstance(result, OutputDirectoryDerived)
        return result.output_directory is not None

    def set_save_root(self, save_root: Path) -> bool:
        """Set the save root and let path events render the active output path."""
        lease = acquire_foreground_operation(
            self.foreground_operations,
            ForegroundOperation.OUTPUT_PACKAGE_CHANGE,
        )
        if isinstance(lease, ForegroundOperationConflict):
            logger.error(lease.message)
            return False
        with lease:
            result = self.commands.set_output_root(save_root)
            if isinstance(result, Failed):
                logger.error(result.message)
                return False
            assert isinstance(result, OutputRootSet)
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

    def render_output_paths(
        self,
        output_root: Path | None,
        output_directory: Path | None,
    ) -> None:
        """Render frontend output path state from app-owned path values."""
        if output_root is not None:
            self.path_view.set_output_root(output_root)
            self.display_output_directory(output_directory)
        elif output_directory is not None:
            self.display_output_directory(output_directory)
        else:
            self.display_output_directory(None)
