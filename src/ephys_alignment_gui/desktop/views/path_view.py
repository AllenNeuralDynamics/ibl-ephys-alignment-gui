"""Desktop view wrapper for input/output path widgets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DesktopPathView:
    """Own Qt widget operations for path text fields."""

    mouse_root_button: Any
    mouse_root_line: Any
    output_folder_line: Any

    def mouse_root_text(self) -> str:
        """Return the edited mouse-root path text."""
        return self.mouse_root_line.text().strip()

    def output_root_text(self) -> str:
        """Return the edited output-root path text."""
        return self.output_folder_line.text().strip()

    def set_mouse_root(self, mouse_root: Path) -> None:
        """Render the selected mouse-root path."""
        self.mouse_root_line.setText(str(mouse_root))

    def set_output_directory(self, output_directory: Path | None) -> None:
        """Render secondary state for the active per-probe output directory."""
        if output_directory is None:
            self.output_folder_line.setToolTip("")
            return
        self.output_folder_line.setToolTip(
            f"Active probe output directory: {output_directory}"
        )

    def set_output_root(self, output_root: Path) -> None:
        """Render the editable save root."""
        self.output_folder_line.setText(str(output_root))

    def mouse_root_widgets(self) -> list[Any]:
        """Widgets disabled while the mouse-root datapackage is loading."""
        return [self.mouse_root_button, self.mouse_root_line]
