"""Desktop presentation for loaded-shank histology refresh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DesktopHistologyRefreshPresenter:
    """Coordinate histology, perpendicular slice, and line-overlay refresh."""

    app: Any
    histology_presenter: Any
    slice_panel_presenter: Any
    slice_menu_coordinator: Any
    reference_line_display: Any

    def render_loaded_shank_histology(self, shank_idx: int | None = None) -> bool:
        """Render active histology panels and restore shank reference overlays."""
        if shank_idx is None:
            shank_idx = self.app.queries.workspace.active_shank_selection().shank_idx

        if not self.histology_presenter.render_active_panels():
            return False
        self.slice_panel_presenter.refresh_perpendicular_histology(
            self.slice_menu_coordinator.current_selection()
        )

        line_state = self.app.queries.workspace.active_reference_line_state(shank_idx)
        if line_state is not None:
            warped_positions_um = getattr(
                line_state,
                "warped_positions_um",
                getattr(line_state, "track_positions_um", None),
            )
            self.reference_line_display.create_lines(
                line_state.feature_positions_um,
                warped_positions_um,
            )
            if not self.app.queries.workspace.reference_lines_visible():
                self.reference_line_display.remove_from_plots()
        return True
