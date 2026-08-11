"""Desktop ephys display composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.core.alignment_read_models import ActiveShankScreenState
from ephys_alignment_gui.desktop.ephys_panel_layout import (
    DesktopEphysPanelLayout,
    EphysPanelLayoutCallbacks,
)
from ephys_alignment_gui.desktop.ephys_panel_view import (
    DesktopEphysPanelView,
)
from ephys_alignment_gui.desktop.ephys_plot_presenter import (
    DesktopEphysPlotPresenter,
    EphysPlotRenderCallbacks,
)
from ephys_alignment_gui.plotting.menu_state import PlotMenuState
from ephys_alignment_gui.plotting.registry import PlotMenu


@dataclass(frozen=True)
class DesktopEphysDisplayConfig:
    """External style/callback dependencies needed to build the ephys display."""

    line_pen: Any
    depth_guide_pen: Any
    padding_provider: Callable[[], float]
    raw_image_payloads: Callable[[], Mapping[Any, Any]]
    set_axis: Callable[..., Any]
    reset_axis: Callable[[], None]
    cluster_clicked: Callable[..., Any]
    on_mouse_double_clicked: Callable[..., Any]
    on_mouse_hover: Callable[..., Any]


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
        config: DesktopEphysDisplayConfig,
    ) -> DesktopEphysDisplay:
        """Build the ephys display cluster from desktop dependencies."""
        panel = DesktopEphysPanelView.create(
            depth_view=app.queries.workspace.depth_view_settings(),
            padding=config.padding_provider(),
            line_pen=config.line_pen,
            depth_guide_pen=config.depth_guide_pen,
            set_axis=config.set_axis,
            cluster_clicked=config.cluster_clicked,
            on_mouse_double_clicked=config.on_mouse_double_clicked,
            on_mouse_hover=config.on_mouse_hover,
        )
        plot_presenter = DesktopEphysPlotPresenter(
            app=app,
            callbacks=EphysPlotRenderCallbacks(
                raw_image_payloads=config.raw_image_payloads,
                render_image=panel.render_image,
                render_scatter=panel.render_scatter,
                render_line=panel.render_line,
                render_probe=panel.render_probe,
            ),
        )
        layout = DesktopEphysPanelLayout(
            panel=panel,
            graphics_layout=panel.widgets.graphics_layout,
            callbacks=EphysPanelLayoutCallbacks(
                set_axis=config.set_axis,
                reset_axis=config.reset_axis,
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

    @property
    def area(self) -> Any:
        """Return the top-level ephys panel widget."""
        return self.panel.widgets.area

    @property
    def graphics_layout(self) -> Any:
        """Return the ephys panel graphics layout."""
        return self.panel.widgets.graphics_layout

    @property
    def image_plot(self) -> Any:
        """Return the feature image/scatter plot handle."""
        return self.panel.plots.image

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
        configure: bool = False,
    ) -> None:
        """Apply one of the desktop ephys panel layouts."""
        if configure:
            self.layout.capture_sizes()
        self.layout.apply_view(view)

    def export_sizes(self) -> tuple[float, float]:
        """Return ephys sizes needed by plot export."""
        sizes = self.layout.sizes
        if sizes is None:
            sizes = self.layout.capture_sizes()
        return sizes.probe_width, sizes.axis_width

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
