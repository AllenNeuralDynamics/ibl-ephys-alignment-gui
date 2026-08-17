"""Desktop display action choreography."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.core.alignment_events import (
    HistologyBoundariesVisibilityChanged,
    ReferenceLineVisibilityChanged,
    RegionAnnotationSourceChanged,
)
from ephys_alignment_gui.core.event_bus import EventSubscription


@dataclass
class DesktopDisplayActions:
    """Coordinate desktop display commands and concrete display updates."""

    app: Any
    displays: Any
    histology_presenter: Any
    slice_panel_presenter: Any
    alignment_screen: Any
    fit_alignment: Callable[[], bool]
    histology_available: Callable[[], bool]

    def connect_display_events(self) -> list[EventSubscription]:
        """Subscribe desktop display updates to semantic display-state events."""
        return [
            self.app.events.subscribe(
                HistologyBoundariesVisibilityChanged,
                self.on_histology_boundaries_visibility_changed,
            ),
            self.app.events.subscribe(
                RegionAnnotationSourceChanged,
                self.on_region_annotation_source_changed,
            ),
            self.app.events.subscribe(
                ReferenceLineVisibilityChanged,
                self.on_reference_line_visibility_changed,
            ),
        ]

    def on_histology_boundaries_visibility_changed(
        self,
        event: HistologyBoundariesVisibilityChanged,
    ) -> None:
        """Render the selected reference/nearby histology boundary display."""
        if not event.visible:
            self.histology_presenter.render_active_nearby()
            return
        self.histology_presenter.render_active_reference()

    def on_region_annotation_source_changed(
        self,
        _event: RegionAnnotationSourceChanged,
    ) -> None:
        """Refresh histology panels after the annotation source changes."""
        self.histology_presenter.render_active_aligned()
        self.histology_presenter.render_active_reference()
        self.histology_presenter.render_active_scale_factor()
        if self.app.queries.workspace.reference_lines_visible():
            self.displays.reference_lines.reattach()
        else:
            self.displays.reference_lines.remove_from_plots()

    def on_reference_line_visibility_changed(
        self,
        event: ReferenceLineVisibilityChanged,
    ) -> None:
        """Apply reference-line visibility to desktop plot handles."""
        if not event.visible:
            self.displays.reference_lines.remove_from_plots()
            return
        self.displays.reference_lines.add_to_plots()

    def toggle_histology_boundaries(self) -> bool:
        """Toggle reference/nearby histology boundary display."""
        self.app.commands.display.toggle_histology_boundaries_visible()
        return True

    def toggle_region_annotation_source(self) -> None:
        """Toggle region annotation source and refresh histology panels."""
        self.app.commands.display.toggle_region_annotation_source()

    def toggle_labels(self) -> None:
        """Toggle atlas label visibility on histology panels."""
        self.displays.histology.toggle_labels()

    def toggle_reference_lines(self) -> None:
        """Toggle reference-line visibility on desktop plots."""
        self.app.commands.display.toggle_reference_lines_visible()

    def toggle_channels(self) -> None:
        """Toggle channel overlays on slice panels."""
        self.slice_panel_presenter.toggle_channel_visibility()

    def delete_selected_reference_line(self) -> None:
        """Delete the currently selected reference line, if any."""
        self.displays.reference_lines.delete_selected()

    def reset_axis(self) -> None:
        """Reset feature-depth y-range and feature image x-range."""
        self.alignment_screen.set_default_feature_y_range(
            depth_view=self.app.queries.workspace.depth_view_settings(),
            in_brain_depths_um=self.app.queries.ephys.active_in_brain_depths_um(),
        )
        self.displays.ephys.reset_feature_image_x_range()

    def set_linear_fit_enabled(self, enabled: bool) -> bool:
        """Set linear-fit option and recompute when reference lines exist."""
        self.app.commands.display.set_linear_fit_enabled(enabled)
        if (
            not self.histology_available()
            or not self.displays.reference_lines.has_lines()
        ):
            return False
        return self.fit_alignment()

    def sync_histology_top_to_tip(self) -> None:
        """Keep histology top line synchronized to the current tip line."""
        self.displays.histology.sync_top_to_tip()

    def sync_histology_tip_to_top(self) -> None:
        """Keep histology tip line synchronized to the current top line."""
        self.displays.histology.sync_tip_to_top()
