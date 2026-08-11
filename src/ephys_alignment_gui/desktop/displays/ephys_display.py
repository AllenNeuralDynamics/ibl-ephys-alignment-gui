"""Desktop ephys display composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.displays.ephys_panel_layout import (
    DesktopEphysPanelLayout,
    EphysPanelLayoutCallbacks,
)
from ephys_alignment_gui.desktop.displays.ephys_panel_view import (
    DesktopEphysPanelView,
)


@dataclass(frozen=True)
class DesktopEphysDisplayConfig:
    """External style/callback dependencies needed to build the ephys display."""

    depth_view: Any
    line_pen: Any
    depth_guide_pen: Any
    padding_provider: Callable[[], float]
    set_axis: Callable[..., Any]
    reset_axis: Callable[[], None]
    cluster_clicked: Callable[..., Any]
    on_mouse_double_clicked: Callable[..., Any]
    on_mouse_hover: Callable[..., Any]


@dataclass(frozen=True)
class DesktopEphysDisplay:
    """Own the ephys panel view and layout handles."""

    panel: DesktopEphysPanelView
    layout: DesktopEphysPanelLayout

    @classmethod
    def create(
        cls,
        *,
        config: DesktopEphysDisplayConfig,
    ) -> DesktopEphysDisplay:
        """Build the ephys display cluster from desktop dependencies."""
        panel = DesktopEphysPanelView.create(
            depth_view=config.depth_view,
            padding=config.padding_provider(),
            line_pen=config.line_pen,
            depth_guide_pen=config.depth_guide_pen,
            set_axis=config.set_axis,
            cluster_clicked=config.cluster_clicked,
            on_mouse_double_clicked=config.on_mouse_double_clicked,
            on_mouse_hover=config.on_mouse_hover,
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

    def show_empty_state(self) -> None:
        """Show the unloaded placeholder on the feature image plot."""
        self.panel.show_empty_state()

    def clear_empty_state(self) -> None:
        """Clear the unloaded placeholder from the feature image plot."""
        self.panel.clear_empty_state()

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
