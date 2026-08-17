"""Desktop histology display composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.displays.histology_panel_view import (
    HistologyPanelView,
)


@dataclass(frozen=True)
class DesktopHistologyDisplayConfig:
    """External style/callback dependencies needed to build histology displays."""

    depth_view: Any
    dotted_pen: Any
    fit_pen: Any
    linear_fit_pen: Any
    baseline_pen: Any
    set_axis: Callable[..., Any]
    padding_provider: Callable[[], float]
    on_linear_fit_changed: Callable[..., Any]
    on_mouse_double_clicked: Callable[..., Any]
    on_mouse_hover: Callable[..., Any]
    linear_fit_enabled: Callable[[], bool]


@dataclass(frozen=True)
class DesktopHistologyDisplay:
    """Own histology-panel pyqtgraph handles and desktop-only helpers."""

    panel: HistologyPanelView

    @classmethod
    def create(
        cls,
        *,
        config: DesktopHistologyDisplayConfig,
        perpendicular_plot: Any,
        view_factory: Callable[..., HistologyPanelView] = HistologyPanelView.create,
    ) -> DesktopHistologyDisplay:
        """Build the histology display cluster from desktop dependencies."""
        panel = view_factory(
            depth_view=config.depth_view,
            padding=config.padding_provider(),
            set_axis=config.set_axis,
            dotted_pen=config.dotted_pen,
            fit_pen=config.fit_pen,
            linear_fit_pen=config.linear_fit_pen,
            baseline_pen=config.baseline_pen,
            perpendicular_plot=perpendicular_plot,
            linear_fit_enabled=config.linear_fit_enabled,
            on_linear_fit_changed=config.on_linear_fit_changed,
            on_mouse_double_clicked=config.on_mouse_double_clicked,
            on_mouse_hover=config.on_mouse_hover,
        )
        return cls(panel=panel)

    @property
    def area(self) -> Any:
        """Return the top-level histology panel widget."""
        return self.panel.plots.area

    @property
    def layout(self) -> Any:
        """Return the histology graphics layout."""
        return self.panel.plots.layout

    @property
    def depth_ruler(self) -> Any:
        """Return the shared histology depth-ruler plot handle."""
        return self.panel.plots.depth_ruler

    @property
    def scale_plot(self) -> Any:
        """Return the scale-factor strip plot handle."""
        return self.panel.plots.scale

    @property
    def scale_axis(self) -> Any:
        """Return the scale-factor colourbar axis."""
        return self.panel.plots.scale_axis

    @property
    def fit_plot(self) -> Any:
        """Return the fit plot widget."""
        return self.panel.fit_plot

    @property
    def linear_fit_checkbox(self) -> Any:
        """Return the linear-fit checkbox."""
        return self.panel.linear_fit_checkbox

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
        return self.layout.scene()

    def clear(self) -> None:
        """Clear histology-panel plot items and forget desktop handles."""
        self.panel.clear()

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

    def warped_feature_y_from_scene(self, scene_pos: Any) -> float | None:
        """Map a warped histology scene position to displayed feature depth."""
        return self.panel.warped_feature_y_from_scene(scene_pos)

    def select_region(self, item: Any) -> None:
        """Record the currently hovered/selected histology region item."""
        self.panel.select_region(item)

    def selected_region_index(self) -> int | None:
        """Return the index of the selected histology/ref region."""
        return self.panel.selected_region_index()

    def scale_factor_for_region_item(self, item: Any) -> float | None:
        """Return the scale factor associated with a rendered scale-region item."""
        return self.panel.scale_factor_for_region_item(item)
