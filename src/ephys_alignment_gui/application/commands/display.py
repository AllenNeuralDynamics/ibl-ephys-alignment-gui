"""App command handler for frontend-agnostic display/edit settings."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState


@dataclass(frozen=True)
class DisplayCommandHandler:
    """Mutate app-owned display/edit settings from user commands."""

    display_state: AlignmentDisplayState

    def toggle_reference_lines_visible(self) -> bool:
        """Toggle whether reference lines are visible in rendered panels."""
        return self.display_state.toggle_reference_lines_visible()

    def toggle_histology_boundaries_visible(self) -> bool:
        """Toggle whether nearby/reference histology boundaries are visible."""
        return self.display_state.toggle_histology_boundaries_visible()

    def toggle_region_annotation_source(self) -> str:
        """Toggle between available region annotation label sources."""
        return self.display_state.toggle_region_annotation_source()

    def set_linear_fit_enabled(self, enabled: bool) -> bool:
        """Set whether fit commands should use linear fitting."""
        return self.display_state.edit_settings.set_lin_fit(enabled)

    def set_probe_limits(self, min_um: float, max_um: float) -> None:
        """Set the active probe depth limits used by desktop render queries."""
        self.display_state.depth_view.set_probe_limits(min_um, max_um)
