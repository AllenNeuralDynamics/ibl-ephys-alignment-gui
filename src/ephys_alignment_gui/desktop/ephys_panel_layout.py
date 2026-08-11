"""Desktop layout switching for ephys plot panels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.ephys_panel_view import DesktopEphysPanelView


@dataclass(frozen=True)
class EphysPanelLayoutSizes:
    """Captured ephys panel dimensions used for stable layout switching."""

    axis_width: float
    image_width: float
    line_width: float
    probe_width: float


@dataclass(frozen=True)
class EphysPanelLayoutCallbacks:
    """Desktop callbacks needed after ephys panel layout changes."""

    set_axis: Callable[..., Any]
    reset_axis: Callable[[], None]


@dataclass
class DesktopEphysPanelLayout:
    """Apply desktop ephys panel plot arrangements."""

    panel: DesktopEphysPanelView
    graphics_layout: Any
    callbacks: EphysPanelLayoutCallbacks
    sizes: EphysPanelLayoutSizes | None = None

    def capture_sizes(self) -> EphysPanelLayoutSizes:
        """Capture current panel dimensions for stable layout switching."""
        values = self.panel.capture_layout_sizes()
        self.sizes = EphysPanelLayoutSizes(
            axis_width=values["axis_width"],
            image_width=values["image_width"],
            line_width=values["line_width"],
            probe_width=values["probe_width"],
        )
        return self.sizes

    def apply_view(self, view: int, sizes: EphysPanelLayoutSizes | None = None) -> None:
        """Apply one of the three ephys data-panel layouts."""
        if sizes is None:
            sizes = self.sizes
        if sizes is None:
            sizes = self.capture_sizes()
        if view == 1:
            self._apply_image_line_probe(sizes)
        elif view == 2:
            self._apply_image_probe_line(sizes)
        elif view == 3:
            self._apply_probe_line_image(sizes)
        else:
            return
        self._refresh_plots()

    def _apply_image_line_probe(self, sizes: EphysPanelLayoutSizes) -> None:
        plots = self.panel.plots
        self._clear_layout()
        self.graphics_layout.addItem(plots.image_colorbar, 0, 0)
        self.graphics_layout.addItem(plots.probe_colorbar, 0, 1, 1, 2)
        self.graphics_layout.addItem(plots.image, 1, 0)
        self.graphics_layout.addItem(plots.line, 1, 1)
        self.graphics_layout.addItem(plots.probe, 1, 2)
        self._hide_probe_and_line_depth_axes()
        self._set_image_left_axis()
        plots.image.setPreferredWidth(sizes.image_width + sizes.axis_width)
        plots.line.setPreferredWidth(sizes.line_width)
        plots.probe.setFixedWidth(sizes.probe_width)
        self._set_stretch_factors(6, 1, 1)

    def _apply_image_probe_line(self, sizes: EphysPanelLayoutSizes) -> None:
        plots = self.panel.plots
        self._clear_layout()
        self.graphics_layout.addItem(plots.image_colorbar, 0, 0)
        self.graphics_layout.addItem(plots.probe_colorbar, 0, 1, 1, 2)
        self.graphics_layout.addItem(plots.image, 1, 0)
        self.graphics_layout.addItem(plots.probe, 1, 1)
        self.graphics_layout.addItem(plots.line, 1, 2)
        self._hide_probe_and_line_depth_axes()
        self._set_image_left_axis()
        plots.image.setPreferredWidth(sizes.image_width + sizes.axis_width)
        plots.line.setPreferredWidth(sizes.line_width)
        plots.probe.setFixedWidth(sizes.probe_width)
        self._set_stretch_factors(6, 1, 1)

    def _apply_probe_line_image(self, sizes: EphysPanelLayoutSizes) -> None:
        plots = self.panel.plots
        self._clear_layout()
        self.graphics_layout.addItem(plots.probe_colorbar, 0, 0, 1, 2)
        self.graphics_layout.addItem(plots.image_colorbar, 0, 2)
        self.graphics_layout.addItem(plots.probe, 1, 0)
        self.graphics_layout.addItem(plots.line, 1, 1)
        self.graphics_layout.addItem(plots.image, 1, 2)
        self.callbacks.set_axis(plots.probe_colorbar, "left", pen="w")
        self.callbacks.set_axis(plots.image_colorbar, "left", show=False)
        self.callbacks.set_axis(plots.line, "left", show=False)
        self.callbacks.set_axis(plots.image, "left", pen="w")
        self.callbacks.set_axis(plots.image, "left", show=False)
        self.callbacks.set_axis(
            plots.probe,
            "left",
            label="Distance from probe tip (um)",
        )
        plots.probe.setFixedWidth(sizes.probe_width + sizes.axis_width)
        plots.image.setPreferredWidth(sizes.image_width)
        plots.line.setPreferredWidth(sizes.line_width)
        self._set_stretch_factors(1, 1, 6)

    def _clear_layout(self) -> None:
        plots = self.panel.plots
        self.graphics_layout.removeItem(plots.image_colorbar)
        self.graphics_layout.removeItem(plots.probe_colorbar)
        self.graphics_layout.removeItem(plots.image)
        self.graphics_layout.removeItem(plots.line)
        self.graphics_layout.removeItem(plots.probe)

    def _hide_probe_and_line_depth_axes(self) -> None:
        plots = self.panel.plots
        self.callbacks.set_axis(plots.image_colorbar, "left", pen="w")
        self.callbacks.set_axis(plots.probe_colorbar, "left", show=False)
        self.callbacks.set_axis(plots.probe, "left", show=False)
        self.callbacks.set_axis(plots.line, "left", show=False)

    def _set_image_left_axis(self) -> None:
        self.callbacks.set_axis(
            self.panel.plots.image,
            "left",
            label="Distance from probe tip (um)",
        )

    def _set_stretch_factors(self, col0: int, col1: int, col2: int) -> None:
        self.graphics_layout.layout.setColumnStretchFactor(0, col0)
        self.graphics_layout.layout.setColumnStretchFactor(1, col1)
        self.graphics_layout.layout.setColumnStretchFactor(2, col2)
        self.graphics_layout.layout.setRowStretchFactor(0, 1)
        self.graphics_layout.layout.setRowStretchFactor(1, 10)

    def _refresh_plots(self) -> None:
        plots = self.panel.plots
        plots.image.update()
        feature_xrange = self.panel.feature_xrange
        if feature_xrange is not None:
            plots.image.setXRange(
                min=feature_xrange[0] - 10,
                max=feature_xrange[1] + 10,
                padding=0,
            )
        self.callbacks.reset_axis()
        plots.line.update()
        plots.probe.update()
