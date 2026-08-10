"""Desktop presenter for histology, scale, and fit plot rendering."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.alignment_read_models import (
    ActiveAlignmentRenderState,
    HistologyPanelRenderState,
    ScaleFactorRenderState,
)
from ephys_alignment_gui.desktop.histology_panel_view import HistologyPanelView

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopHistologyRenderCallbacks:
    """Desktop callbacks needed to build histology render state."""

    probe_extent_query_kwargs: Callable[[], dict[str, float]]
    fit_depth_um: Callable[[], Any]
    lin_fit_enabled: Callable[[], bool]
    scale_factor_y_range: Callable[[], tuple[float, float]]


@dataclass
class DesktopHistologyPresenter:
    """Coordinate desktop histology/scale/fit rendering from app read models."""

    app: Any
    panel: HistologyPanelView
    callbacks: DesktopHistologyRenderCallbacks

    def render_active_aligned(
        self,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> bool:
        """Render the active aligned histology panel."""
        histology_state = self.active_histology_panel_state()
        if histology_state is None:
            return False
        self.panel.render_aligned(histology_state, fig, movable=movable)
        return True

    def render_active_reference(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render the active reference histology panel."""
        histology_state = self.active_histology_panel_state()
        if histology_state is None:
            return False
        self.panel.render_reference(histology_state, fig, movable=movable)
        return True

    def render_active_scale_factor(self) -> bool:
        """Render the active scale-factor panel."""
        histology_state = self.active_histology_panel_state(
            unavailable_message=(
                "Cannot render scale factor: active alignment data is not loaded"
            )
        )
        if histology_state is None:
            return False
        self.render_scale_factor(histology_state)
        return True

    def render_active_fit(self) -> bool:
        """Render the active feature/track fit panel."""
        return self.render_fit()

    def render_active_nearby(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render nearby histology boundary distances in the reference panel."""
        brain_atlas = self.app.queries.workspace.active_brain_atlas()
        if brain_atlas is None:
            logger.error("Cannot render nearby boundaries: brain atlas is not loaded")
            return False
        state = self.app.queries.alignment_render.active_nearby_boundary_state(
            **self.callbacks.probe_extent_query_kwargs(),
            allen=self.app.queries.workspace.allen_structure_tree(),
            brain_atlas=brain_atlas,
        )
        if state is None:
            logger.error(
                "Cannot render nearby boundaries: active alignment data is not loaded"
            )
            return False
        self.panel.render_nearby(state, fig, movable=movable)
        return True

    def render_active_panels(self, *, labels_visible: bool = True) -> bool:
        """Render reference histology, aligned histology, scale, and fit panels."""
        histology_state = self.active_histology_panel_state()
        if histology_state is None:
            return False

        self.panel.render_reference(histology_state)
        self.panel.render_aligned(histology_state)
        self.panel.set_labels_visible(labels_visible)
        self.render_scale_factor(histology_state)
        self.render_fit()
        return True

    def render_alignment_edit(self, render_state: ActiveAlignmentRenderState) -> bool:
        """Render the histology/scale/fit cluster after an alignment edit."""
        histology_state = self.histology_panel_state(render_state)
        if histology_state is None:
            return False

        self.panel.render_aligned(histology_state)
        self.render_scale_factor(histology_state)
        self.render_fit()
        return True

    def render_aligned(
        self,
        render_state: ActiveAlignmentRenderState,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> bool:
        """Render aligned histology from a shared active-alignment DTO."""
        histology_state = self.histology_panel_state(render_state)
        if histology_state is None:
            return False
        self.panel.render_aligned(histology_state, fig, movable=movable)
        return True

    def render_reference(
        self,
        render_state: ActiveAlignmentRenderState,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render reference histology from a shared active-alignment DTO."""
        histology_state = self.histology_panel_state(render_state)
        if histology_state is None:
            return False
        self.panel.render_reference(histology_state, fig, movable=movable)
        return True

    def active_histology_panel_state(
        self,
        *,
        unavailable_message: str = (
            "Cannot render histology: active alignment data is not loaded"
        ),
    ) -> HistologyPanelRenderState | None:
        """Return active histology-panel state from the narrow histology query."""
        state = self.app.queries.alignment_render.active_histology_panel_state(
            **self.callbacks.probe_extent_query_kwargs()
        )
        if state is None:
            logger.error(unavailable_message)
        return state

    def histology_panel_state(
        self,
        render_state: ActiveAlignmentRenderState,
    ) -> HistologyPanelRenderState | None:
        """Build histology-panel state for an active alignment render DTO."""
        probe_extent = self.app.queries.alignment_render.probe_extent_render_state(
            render_state.active_alignment,
            **self.callbacks.probe_extent_query_kwargs(),
        )
        if probe_extent is None:
            logger.error("Cannot render histology: active probe extent is not loaded")
            return None
        return HistologyPanelRenderState(
            key=render_state.key,
            histology=render_state.histology,
            probe_extent=probe_extent,
        )

    def render_scale_factor(self, histology_state: HistologyPanelRenderState) -> None:
        """Render scale-factor data associated with an active alignment render DTO."""
        self.panel.render_scale_factor(
            ScaleFactorRenderState(
                key=histology_state.key,
                region=histology_state.histology.scale.region,
                scale=histology_state.histology.scale.scale,
                probe_extent=histology_state.probe_extent,
            ),
            y_range=self.callbacks.scale_factor_y_range(),
        )

    def render_fit(self) -> bool:
        """Render the active feature/track fit curve."""
        state = self.app.queries.alignment_render.active_fit_plot_state(
            depth_um=self.callbacks.fit_depth_um(),
            lin_fit=self.callbacks.lin_fit_enabled(),
        )
        if state is None:
            logger.error("Cannot render fit: active alignment data is not loaded")
            return False
        self.panel.render_fit(state)
        return True
