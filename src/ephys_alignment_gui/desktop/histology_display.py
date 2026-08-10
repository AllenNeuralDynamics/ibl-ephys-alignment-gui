"""Desktop histology display composition."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.alignment_read_models import ActiveAlignmentRenderState
from ephys_alignment_gui.desktop.histology_panel_view import (
    FitPanelItems,
    HistologyPanelAxes,
    HistologyPanelPlots,
    HistologyPanelStyle,
    HistologyPanelView,
)
from ephys_alignment_gui.desktop.histology_presenter import (
    DesktopHistologyPresenter,
    DesktopHistologyRenderCallbacks,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopHistologyDisplayPorts:
    """Desktop handles and callbacks needed to build the histology display."""

    aligned_plot: Any
    reference_plot: Any
    scale_plot: Any
    scale_colorbar: Any
    aligned_axis: Any
    reference_axis: Any
    layout: Any
    extra_y_axis: Any
    dotted_pen: Any
    fit_curve: Any
    fit_scatter: Any
    linear_fit_curve: Any
    set_axis: Callable[..., Any]
    padding_provider: Callable[[], float]
    scale_factor_y_range: Callable[[], tuple[float, float]]
    histology_available: Callable[[], bool]


@dataclass(frozen=True)
class DesktopHistologyDisplay:
    """Own the histology panel and app-querying histology presenter."""

    panel: HistologyPanelView
    presenter: DesktopHistologyPresenter
    ports: DesktopHistologyDisplayPorts

    @classmethod
    def create(
        cls,
        *,
        app: Any,
        ports: DesktopHistologyDisplayPorts,
    ) -> DesktopHistologyDisplay:
        """Build the histology display cluster from desktop ports."""
        panel = HistologyPanelView(
            plots=HistologyPanelPlots(
                aligned=ports.aligned_plot,
                reference=ports.reference_plot,
                scale=ports.scale_plot,
                scale_colorbar=ports.scale_colorbar,
            ),
            axes=HistologyPanelAxes(
                aligned=ports.aligned_axis,
                reference=ports.reference_axis,
            ),
            style=HistologyPanelStyle(dotted_pen=ports.dotted_pen),
            set_axis=ports.set_axis,
            padding_provider=ports.padding_provider,
            fit_items=FitPanelItems(
                fit_curve=ports.fit_curve,
                fit_scatter=ports.fit_scatter,
                linear_fit_curve=ports.linear_fit_curve,
            ),
        )
        presenter = DesktopHistologyPresenter(
            app=app,
            panel=panel,
            callbacks=DesktopHistologyRenderCallbacks(
                probe_extent_query_kwargs=lambda: _probe_extent_query_kwargs(app),
                fit_depth_um=app.queries.workspace.fit_depth_um,
                lin_fit_enabled=app.queries.workspace.linear_fit_enabled,
                scale_factor_y_range=ports.scale_factor_y_range,
            ),
        )
        return cls(panel=panel, presenter=presenter, ports=ports)

    @property
    def extra_y_axis(self) -> Any:
        """Return the extra histology y-axis used during plot export."""
        return self.ports.extra_y_axis

    @property
    def aligned_plot(self) -> Any:
        """Return the aligned histology plot handle."""
        return self.panel.plots.aligned

    @property
    def reference_plot(self) -> Any:
        """Return the reference histology plot handle."""
        return self.panel.plots.reference

    def export_scene(self) -> Any:
        """Return the scene that contains the histology export layout."""
        return self.ports.layout.scene()

    def clear(self) -> None:
        """Clear histology-panel plot items and forget desktop handles."""
        self.panel.clear()

    def render_active_aligned(
        self,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> bool:
        """Render the active aligned histology panel."""
        return self.presenter.render_active_aligned(fig, movable=movable)

    def render_active_reference(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render the active reference histology panel."""
        return self.presenter.render_active_reference(fig, movable=movable)

    def render_active_scale_factor(self) -> bool:
        """Render the active scale-factor panel."""
        return self.presenter.render_active_scale_factor()

    def render_active_fit(self) -> bool:
        """Render the active feature/track fit panel."""
        return self.presenter.render_active_fit()

    def render_active_panels(self, *, labels_visible: bool = True) -> bool:
        """Render reference histology, aligned histology, scale, and fit panels."""
        return self.presenter.render_active_panels(labels_visible=labels_visible)

    def render_alignment_edit(self, render_state: ActiveAlignmentRenderState) -> bool:
        """Render the histology/scale/fit cluster after an alignment edit."""
        return self.presenter.render_alignment_edit(render_state)

    def render_active_nearby(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render nearby histology boundary distances."""
        if not self.ports.histology_available():
            return False
        return self.presenter.render_active_nearby(fig, movable=movable)

    def toggle_labels(self) -> None:
        """Toggle atlas label axis visibility for both histology panels."""
        self.panel.toggle_labels()

    def tip_position_um(self) -> float | None:
        """Return the current editable tip-line position."""
        return self.panel.tip_position_um()

    def sync_top_to_tip(self) -> None:
        """Keep the top line synchronized to the current tip line."""
        self.panel.sync_top_to_tip()

    def sync_tip_to_top(self) -> None:
        """Keep the tip line synchronized to the current top line."""
        self.panel.sync_tip_to_top()

    def select_region(self, item: Any) -> None:
        """Record the currently hovered/selected histology region item."""
        self.panel.select_region(item)

    def selected_region_index(self) -> int | None:
        """Return the index of the selected histology/ref region."""
        return self.panel.selected_region_index()

    def scale_factor_for_region_item(self, item: Any) -> float | None:
        """Return the scale factor associated with a rendered scale-region item."""
        return self.panel.scale_factor_for_region_item(item)


def _probe_extent_query_kwargs(app: Any) -> dict[str, float]:
    """Return probe extent settings needed for histology-panel queries."""
    depth_view = app.queries.workspace.depth_view_settings()
    return {
        "probe_tip_um": depth_view.probe_tip_um,
        "probe_top_um": depth_view.probe_top_um,
        "probe_extra_um": depth_view.probe_extra_um,
    }
