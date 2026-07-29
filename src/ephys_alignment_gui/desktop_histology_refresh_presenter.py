"""Desktop presentation for loaded-shank histology refresh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DesktopHistologyRefreshPresenter:
    """Coordinate histology, perpendicular slice, and line-overlay refresh."""

    app: Any
    histology_display: Any
    slice_display: Any
    reference_line_display: Any

    def render_loaded_shank_histology(self, shank_idx: int | None = None) -> bool:
        """Render active histology panels and restore shank reference overlays."""
        if shank_idx is None:
            shank_idx = self.app.queries.active_shank_selection().shank_idx

        if not self.histology_display.render_active_panels():
            return False
        self.slice_display.refresh_perpendicular_histology()

        line_state = self.app.queries.active_reference_line_state(shank_idx)
        if line_state is not None:
            self.reference_line_display.create_lines(
                line_state.feature_positions_um,
                line_state.track_positions_um,
            )
        return True
