"""Desktop presenter for slice-panel query and render choreography."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.core.alignment_read_models import (
    ActiveSliceRenderState,
    PerpendicularSliceRenderState,
)
from ephys_alignment_gui.core.slice_display_policy import SliceSelection
from ephys_alignment_gui.desktop.displays.slice_panel_view import SlicePanelView

logger = logging.getLogger(__name__)


@dataclass
class SlicePanelPresenter:
    """Query app slice read models and render them through a slice view."""

    app: Any
    view: SlicePanelView

    def clear(self) -> None:
        """Clear slice-panel plot items and forget desktop handles."""
        self.view.clear()

    def slice_render_state(
        self,
        selection: SliceSelection | None,
    ) -> ActiveSliceRenderState | None:
        """Return the active render DTO for a slice selection."""
        if selection is None:
            return None
        return self.app.queries.slices.active_slice_render_state(selection)

    def scalar_channel_for_selection(
        self,
        selection: SliceSelection | None,
    ) -> str | None:
        """Return the selected scalar channel for a slice menu selection."""
        render_state = self.slice_render_state(selection)
        if render_state is None:
            return None
        return render_state.scalar_channel

    def render_slice_selection(self, selection: SliceSelection) -> None:
        """Render a coronal slice selection from the application read model."""
        if not self.view.histology_exists():
            return
        render_state = self.slice_render_state(selection)
        if render_state is None:
            logger.warning("No active slice render state for %s", selection)
            return
        self.render_slice(render_state)

    def render_slice(self, render_state: ActiveSliceRenderState) -> None:
        """Render a coronal slice payload and matching perpendicular slice."""
        self.view.render_slice(render_state)
        if render_state.scalar_channel is not None:
            self.plot_perpendicular_histology(render_state.scalar_channel)

    def plot_perpendicular_histology(self, channel_name: str = "ccf") -> None:
        """Plot the perpendicular histology slice for the current alignment."""
        if not self.view.histology_exists():
            return

        self.view.clear_perpendicular()
        render_state = self.app.queries.slices.active_perpendicular_slice_state(
            channel_name
        )
        if render_state is None:
            return

        self.render_perpendicular_histology(render_state)

    def render_perpendicular_histology(
        self,
        render_state: PerpendicularSliceRenderState,
    ) -> None:
        """Render a perpendicular slice payload with desktop plot items."""
        self.view.render_perpendicular_histology(render_state)

    def update_perpendicular_levels(self) -> None:
        """Sync perpendicular plot levels with main slice histogram levels."""
        self.view.update_perpendicular_levels()

    def refresh_perpendicular_histology(
        self,
        selection: SliceSelection | None,
    ) -> None:
        """Refresh perpendicular slice for the selected scalar slice."""
        channel_name = self.scalar_channel_for_selection(selection)
        if channel_name is None:
            return
        self.plot_perpendicular_histology(channel_name)

    def plot_channels(
        self,
        projection: Any = None,
        *,
        selection: SliceSelection | None = None,
    ) -> None:
        """Render or update channel/tip overlays on the coronal slice."""
        if projection is None:
            render_state = self.slice_render_state(selection)
            if render_state is None:
                return
            projection = render_state.projection
        self.view.plot_channels(projection)

    def toggle_channel_visibility(self) -> None:
        """Toggle channel, tip, trajectory, and perpendicular overlays."""
        self.view.toggle_channel_visibility()

    def render_export_trajectory_overlay(
        self,
        pen: Any,
        *,
        selection: SliceSelection | None = None,
    ) -> None:
        """Render the coronal trajectory overlay used by overview exports."""
        channel_locations_ras = self.current_channel_locations_ras(selection)
        self.view.render_export_trajectory_overlay(
            pen,
            channel_locations_ras=channel_locations_ras,
        )

    def current_channel_locations_ras(
        self,
        selection: SliceSelection | None = None,
    ) -> Any | None:
        """Return channel locations for the current slice overlay."""
        channel_locations_ras = self.view.current_channel_locations_ras()
        if channel_locations_ras is not None:
            return channel_locations_ras

        render_state = self.slice_render_state(selection)
        if render_state is None:
            return None
        self.view.set_channel_projection(render_state.projection)
        return render_state.projection.channel_locations_ras
