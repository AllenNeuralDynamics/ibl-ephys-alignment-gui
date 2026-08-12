"""App command handler for frontend-agnostic display/edit settings."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.core.alignment_events import (
    HistologyBoundariesVisibilityChanged,
    ReferenceLineVisibilityChanged,
    RegionAnnotationSourceChanged,
)
from ephys_alignment_gui.core.event_bus import EventBus


@dataclass(frozen=True)
class DisplayCommandHandler:
    """Mutate app-owned display/edit settings from user commands."""

    display_state: AlignmentDisplayState
    events: EventBus

    def toggle_reference_lines_visible(self) -> bool:
        """Toggle whether reference lines are visible in rendered panels."""
        visible = self.display_state.toggle_reference_lines_visible()
        self.events.emit(ReferenceLineVisibilityChanged(visible=visible))
        return visible

    def toggle_histology_boundaries_visible(self) -> bool:
        """Toggle whether nearby/reference histology boundaries are visible."""
        visible = self.display_state.toggle_histology_boundaries_visible()
        self.events.emit(HistologyBoundariesVisibilityChanged(visible=visible))
        return visible

    def toggle_region_annotation_source(self) -> str:
        """Toggle between available region annotation label sources."""
        source = self.display_state.toggle_region_annotation_source()
        self.events.emit(RegionAnnotationSourceChanged(source=source))
        return source

    def set_linear_fit_enabled(self, enabled: bool) -> bool:
        """Set whether fit commands should use linear fitting."""
        return self.display_state.edit_settings.set_lin_fit(enabled)

    def set_probe_limits(self, min_um: float, max_um: float) -> None:
        """Set the active probe depth limits used by desktop render queries."""
        self.display_state.depth_view.set_probe_limits(min_um, max_um)
