"""Desktop presenter for histology, scale, and fit plot rendering."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.core.alignment_read_models import (
    ActiveAlignmentRenderState,
    ActiveHistologyScreenState,
)
from ephys_alignment_gui.desktop.displays.histology_panel_view import HistologyPanelView

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
        screen_state = self.active_histology_screen_state()
        if screen_state is None:
            return False
        self.panel.render_aligned(screen_state.histology, fig, movable=movable)
        return True

    def render_active_reference(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render the active reference histology panel."""
        screen_state = self.active_histology_screen_state()
        if screen_state is None:
            return False
        self.panel.render_reference(screen_state.histology, fig, movable=movable)
        return True

    def render_active_scale_factor(self) -> bool:
        """Render the active scale-factor panel."""
        screen_state = self.active_histology_screen_state(
            unavailable_message=(
                "Cannot render scale factor: active alignment data is not loaded"
            )
        )
        if screen_state is None:
            return False
        self.render_scale_factor(screen_state)
        return True

    def render_active_fit(self) -> bool:
        """Render the active feature/track fit panel."""
        screen_state = self.active_histology_screen_state(
            unavailable_message="Cannot render fit: active alignment data is not loaded"
        )
        if screen_state is None:
            return False
        self.render_fit(screen_state)
        return True

    def render_active_nearby(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render nearby histology boundary distances in the reference panel."""
        screen_state = self.active_nearby_boundary_screen_state(
            unavailable_message=(
                "Cannot render nearby boundaries: active alignment data is not loaded"
            )
        )
        if screen_state is None:
            return False
        if screen_state.nearby is None:
            logger.error("Cannot render nearby boundaries: brain atlas is not loaded")
            return False
        self.panel.render_nearby(screen_state.nearby, fig, movable=movable)
        return True

    def render_active_panels(self, *, labels_visible: bool = True) -> bool:
        """Render reference histology, aligned histology, scale, and fit panels."""
        screen_state = self.active_histology_screen_state()
        if screen_state is None:
            return False

        self.panel.render_reference(screen_state.histology)
        self.panel.render_aligned(screen_state.histology)
        self.panel.set_labels_visible(labels_visible)
        self.render_scale_factor(screen_state)
        self.render_fit(screen_state)
        return True

    def render_alignment_edit(self, render_state: ActiveAlignmentRenderState) -> bool:
        """Render the histology/scale/fit cluster after an alignment edit."""
        screen_state = self.histology_screen_state(render_state)
        if screen_state is None:
            return False

        self.panel.render_aligned(screen_state.histology)
        self.render_scale_factor(screen_state)
        self.render_fit(screen_state)
        return True

    def render_aligned(
        self,
        render_state: ActiveAlignmentRenderState,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> bool:
        """Render aligned histology from a shared active-alignment DTO."""
        screen_state = self.histology_screen_state(render_state)
        if screen_state is None:
            return False
        self.panel.render_aligned(screen_state.histology, fig, movable=movable)
        return True

    def render_reference(
        self,
        render_state: ActiveAlignmentRenderState,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render reference histology from a shared active-alignment DTO."""
        screen_state = self.histology_screen_state(render_state)
        if screen_state is None:
            return False
        self.panel.render_reference(screen_state.histology, fig, movable=movable)
        return True

    def active_histology_screen_state(
        self,
        *,
        unavailable_message: str = (
            "Cannot render histology: active alignment data is not loaded"
        ),
    ) -> ActiveHistologyScreenState | None:
        """Return active histology-screen state from the app query layer."""
        state = self.app.queries.alignment_render.active_histology_screen_state(
            **self.callbacks.probe_extent_query_kwargs(),
            depth_um=self.callbacks.fit_depth_um(),
            lin_fit=self.callbacks.lin_fit_enabled(),
        )
        if state is None:
            logger.error(unavailable_message)
        return state

    def active_nearby_boundary_screen_state(
        self,
        *,
        unavailable_message: str,
    ) -> ActiveHistologyScreenState | None:
        """Return active histology-screen state including nearby boundaries."""
        state = self.app.queries.alignment_render.active_nearby_boundary_screen_state(
            **self.callbacks.probe_extent_query_kwargs(),
            depth_um=self.callbacks.fit_depth_um(),
            lin_fit=self.callbacks.lin_fit_enabled(),
        )
        if state is None:
            logger.error(unavailable_message)
        return state

    def histology_screen_state(
        self,
        render_state: ActiveAlignmentRenderState,
    ) -> ActiveHistologyScreenState | None:
        """Build histology-screen state for an active alignment render DTO."""
        state = self.app.queries.alignment_render.histology_screen_state_for_alignment(
            render_state,
            **self.callbacks.probe_extent_query_kwargs(),
            depth_um=self.callbacks.fit_depth_um(),
            lin_fit=self.callbacks.lin_fit_enabled(),
        )
        if state is None:
            logger.error("Cannot render histology: active probe extent is not loaded")
        return state

    def render_scale_factor(self, screen_state: ActiveHistologyScreenState) -> None:
        """Render scale-factor data associated with an active alignment render DTO."""
        self.panel.render_scale_factor(
            screen_state.scale_factor,
            y_range=self.callbacks.scale_factor_y_range(),
        )

    def render_fit(self, screen_state: ActiveHistologyScreenState) -> None:
        """Render the active feature/track fit curve."""
        self.panel.render_fit(screen_state.fit)
