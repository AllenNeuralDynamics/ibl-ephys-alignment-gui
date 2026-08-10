"""Desktop presentation for plot export commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DesktopPlotExportPresenter:
    """Resolve desktop export paths and delegate plot rendering/export."""

    app: Any
    plot_exporter: Any
    output_folder_prompt: Any

    def save_plots(self, save_path: Any = None, *, sess_info: str = "") -> bool:
        """Export all plots to an explicit or app-derived output directory."""
        if save_path:
            output_dir = Path(save_path)
        else:
            if not self.app.queries.workspace.has_output_directory():
                if not self.output_folder_prompt.ensure_for_save():
                    return False
            image_path = self.app.queries.workspace.active_plot_export_directory()
            if image_path is None:
                return False
            output_dir = Path(image_path)

        output_dir.mkdir(exist_ok=True)
        self.plot_exporter.export(output_dir, sess_info=sess_info)
        return True
