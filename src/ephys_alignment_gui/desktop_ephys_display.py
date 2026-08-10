"""Desktop ephys display composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.alignment_read_models import ActiveShankScreenState
from ephys_alignment_gui.desktop_ephys_panel_layout import (
    DesktopEphysPanelLayout,
    EphysPanelLayoutCallbacks,
    EphysPanelLayoutSizes,
)
from ephys_alignment_gui.desktop_ephys_panel_view import (
    DesktopEphysPanelView,
    EphysPanelPlots,
    EphysPanelStyle,
)
from ephys_alignment_gui.desktop_ephys_plot_presenter import (
    DesktopEphysPlotPresenter,
    EphysPlotRenderCallbacks,
)
from ephys_alignment_gui.plot_menu_state import PlotMenuState
from ephys_alignment_gui.plot_registry import PlotMenu


@dataclass(frozen=True)
class DesktopEphysDisplayPorts:
    """Desktop handles and callbacks needed to build the ephys display."""

    image_plot: Any
    image_colorbar: Any
    line_plot: Any
    probe_plot: Any
    probe_colorbar: Any
    graphics_layout: Any
    line_pen: Any
    raw_image_payloads: Callable[[], Mapping[Any, Any]]
    set_axis: Callable[..., Any]
    reset_axis: Callable[[], None]
    cluster_clicked: Callable[..., Any]


@dataclass(frozen=True)
class DesktopEphysDisplay:
    """Own the ephys panel view, layout, and plot presenter."""

    panel: DesktopEphysPanelView
    layout: DesktopEphysPanelLayout
    plot_presenter: DesktopEphysPlotPresenter

    @classmethod
    def create(
        cls,
        *,
        app: Any,
        ports: DesktopEphysDisplayPorts,
    ) -> DesktopEphysDisplay:
        """Build the ephys display cluster from desktop ports."""
        panel = DesktopEphysPanelView(
            plots=EphysPanelPlots(
                image=ports.image_plot,
                image_colorbar=ports.image_colorbar,
                line=ports.line_plot,
                probe=ports.probe_plot,
                probe_colorbar=ports.probe_colorbar,
            ),
            style=EphysPanelStyle(line_pen=ports.line_pen),
            set_axis=ports.set_axis,
            cluster_clicked=ports.cluster_clicked,
        )
        plot_presenter = DesktopEphysPlotPresenter(
            app=app,
            callbacks=EphysPlotRenderCallbacks(
                raw_image_payloads=ports.raw_image_payloads,
                render_image=panel.render_image,
                render_scatter=panel.render_scatter,
                render_line=panel.render_line,
                render_probe=panel.render_probe,
            ),
        )
        layout = DesktopEphysPanelLayout(
            panel=panel,
            graphics_layout=ports.graphics_layout,
            callbacks=EphysPanelLayoutCallbacks(
                set_axis=ports.set_axis,
                reset_axis=ports.reset_axis,
            ),
        )
        return cls(
            panel=panel,
            layout=layout,
            plot_presenter=plot_presenter,
        )

    @property
    def feature_xrange(self) -> Any:
        """Return the active feature-plot x-range, if one is known."""
        return self.panel.feature_xrange

    def clear(self) -> None:
        """Clear ephys panel plots."""
        self.panel.clear()

    def attach_plot_menus(self, menu_bar: Any) -> None:
        """Attach ephys plot menus to a desktop menu bar."""
        self.plot_presenter.attach_plot_menus(menu_bar)

    def attach_unit_filter_menu(self, menu_bar: Any, parent: Any) -> None:
        """Attach the ephys unit-filter menu to a desktop menu bar."""
        self.plot_presenter.attach_unit_filter_menu(menu_bar, parent)

    def toggle_plot(self, menu: PlotMenu, *, reverse: bool = False) -> None:
        """Toggle to the next available plot in one ephys plot menu."""
        self.plot_presenter.toggle_plot(menu, reverse=reverse)

    def apply_view(
        self,
        *,
        view: int,
        axis_width: float,
        image_width: float,
        line_width: float,
        probe_width: float,
    ) -> None:
        """Apply one of the desktop ephys panel layouts."""
        self.layout.apply_view(
            view,
            EphysPanelLayoutSizes(
                axis_width=axis_width,
                image_width=image_width,
                line_width=line_width,
                probe_width=probe_width,
            ),
        )

    def current_plot_keys(self) -> dict[PlotMenu, str | None]:
        """Return selected plot-spec keys for each ephys plot menu."""
        return self.plot_presenter.current_plot_keys()

    def has_plot_menus(self) -> bool:
        """Return whether the desktop ephys plot menus are attached."""
        return self.plot_presenter.has_plot_menus()

    def render_menus(self, plot_menu_state: PlotMenuState) -> None:
        """Render ephys plot menus from a Qt-free menu read model."""
        self.plot_presenter.render_menus(plot_menu_state)

    def render_shank_ephys_plots(self, state: ActiveShankScreenState) -> None:
        """Render selected ephys plots for the active shank."""
        self.plot_presenter.render_shank_ephys_plots(state)

    def reset_feature_image_x_range(self) -> None:
        """Reset the feature image plot x-range to the active payload range."""
        feature_xrange = self.feature_xrange
        if feature_xrange is None:
            return
        self.panel.plots.image.setXRange(
            min=feature_xrange[0],
            max=feature_xrange[1],
            padding=0,
        )
