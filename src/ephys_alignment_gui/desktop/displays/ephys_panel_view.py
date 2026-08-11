"""Desktop pyqtgraph rendering for ephys data panels."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui

from ephys_alignment_gui.desktop.displays.ephys_plot_items import EphysPlotItems
from ephys_alignment_gui.desktop.displays.feature_plot_view import FeaturePlotView
from ephys_alignment_gui.desktop.displays.plot_elements import ColorBar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EphysPanelPlots:
    """Pyqtgraph plots that make up the desktop ephys panel."""

    image: Any
    image_colorbar: Any
    line: Any
    probe: Any
    probe_colorbar: Any


@dataclass(frozen=True)
class EphysPanelWidgets:
    """Top-level widgets/layout owned by the desktop ephys panel."""

    area: Any
    graphics_layout: Any
    image_axis: Any


@dataclass(frozen=True)
class EphysPanelStyle:
    """Desktop styling handles for ephys panel rendering."""

    line_pen: Any
    depth_guide_pen: Any


@dataclass
class DesktopEphysPanelView:
    """Own ephys pyqtgraph items and render ephys plot payloads."""

    plots: EphysPanelPlots
    widgets: EphysPanelWidgets
    style: EphysPanelStyle
    set_axis: Callable[..., Any]
    cluster_clicked: Callable[..., Any]
    probe_tip_lines: list[Any] = field(default_factory=list)
    probe_top_lines: list[Any] = field(default_factory=list)
    items: EphysPlotItems = field(default_factory=EphysPlotItems)
    feature_plot: FeaturePlotView = field(default_factory=FeaturePlotView)

    @classmethod
    def create(
        cls,
        *,
        depth_view: Any,
        padding: float,
        line_pen: Any,
        depth_guide_pen: Any,
        set_axis: Callable[..., Any],
        cluster_clicked: Callable[..., Any],
        on_mouse_double_clicked: Callable[..., Any],
        on_mouse_hover: Callable[..., Any],
    ) -> DesktopEphysPanelView:
        """Create the desktop ephys panel and all of its plot handles."""
        probe_tip_lines: list[Any] = []
        probe_top_lines: list[Any] = []

        image = pg.PlotItem()
        _set_depth_range(image, depth_view, padding)
        image.setMouseEnabled(x=False, y=True)
        _add_depth_guides(
            image,
            depth_view,
            depth_guide_pen=depth_guide_pen,
            probe_tip_lines=probe_tip_lines,
            probe_top_lines=probe_top_lines,
        )
        set_axis(image, "bottom")
        image_axis = set_axis(image, "left", label="Distance from probe tip (uV)")

        image_colorbar = pg.PlotItem()
        image_colorbar.setMaximumHeight(70)
        image_colorbar.setMouseEnabled(x=False, y=False)
        set_axis(image_colorbar, "bottom", show=False)
        set_axis(image_colorbar, "left", pen="w")
        set_axis(image_colorbar, "top", pen="w")

        line = pg.PlotItem()
        line.setMouseEnabled(x=False, y=True)
        _set_depth_range(line, depth_view, padding)
        _add_depth_guides(
            line,
            depth_view,
            depth_guide_pen=depth_guide_pen,
            probe_tip_lines=probe_tip_lines,
            probe_top_lines=probe_top_lines,
        )
        set_axis(line, "bottom")
        set_axis(line, "left", show=False)

        probe = pg.PlotItem()
        probe.setMouseEnabled(x=False, y=False)
        probe.setMaximumWidth(50)
        _set_depth_range(probe, depth_view, padding)
        _add_depth_guides(
            probe,
            depth_view,
            depth_guide_pen=depth_guide_pen,
            probe_tip_lines=probe_tip_lines,
            probe_top_lines=probe_top_lines,
        )
        set_axis(probe, "bottom", pen="w")
        set_axis(probe, "left", show=False)

        probe_colorbar = pg.PlotItem()
        probe_colorbar.setMouseEnabled(x=False, y=False)
        probe_colorbar.setMaximumHeight(70)
        set_axis(probe_colorbar, "bottom", show=False)
        set_axis(probe_colorbar, "left", show=False)
        set_axis(probe_colorbar, "top", pen="w")

        area = pg.GraphicsLayoutWidget()
        area.scene().sigMouseClicked.connect(on_mouse_double_clicked)
        area.scene().sigMouseHover.connect(on_mouse_hover)
        graphics_layout = pg.GraphicsLayout()
        graphics_layout.addItem(image_colorbar, 0, 0)
        graphics_layout.addItem(probe_colorbar, 0, 1, 1, 2)
        graphics_layout.addItem(image, 1, 0)
        graphics_layout.addItem(line, 1, 1)
        graphics_layout.addItem(probe, 1, 2)
        graphics_layout.layout.setColumnStretchFactor(0, 6)
        graphics_layout.layout.setColumnStretchFactor(1, 1)
        graphics_layout.layout.setColumnStretchFactor(2, 1)
        graphics_layout.layout.setRowStretchFactor(0, 1)
        graphics_layout.layout.setRowStretchFactor(1, 10)
        area.addItem(graphics_layout)

        return cls(
            plots=EphysPanelPlots(
                image=image,
                image_colorbar=image_colorbar,
                line=line,
                probe=probe,
                probe_colorbar=probe_colorbar,
            ),
            widgets=EphysPanelWidgets(
                area=area,
                graphics_layout=graphics_layout,
                image_axis=image_axis,
            ),
            style=EphysPanelStyle(
                line_pen=line_pen,
                depth_guide_pen=depth_guide_pen,
            ),
            set_axis=set_axis,
            cluster_clicked=cluster_clicked,
            probe_tip_lines=probe_tip_lines,
            probe_top_lines=probe_top_lines,
        )

    @property
    def feature_xrange(self) -> Any:
        """Return the active feature-plot x-range, if one is known."""
        return self.feature_plot.xrange

    @property
    def probe_colorbars(self) -> list[Any]:
        """Return currently rendered probe colorbar items."""
        return self.items.probe_colorbars

    def capture_layout_sizes(self) -> dict[str, float]:
        """Capture stable dimensions used when reordering ephys panels."""
        axis_width = self.widgets.image_axis.width()
        return {
            "axis_width": axis_width,
            "image_width": self.plots.image.width() - axis_width,
            "line_width": self.plots.line.width(),
            "probe_width": self.plots.probe.width(),
        }

    def feature_y_range(self) -> tuple[float, float]:
        """Return the current feature-depth y-range."""
        y_min, y_max = self.plots.image.viewRange()[1]
        return float(y_min), float(y_max)

    def clear(self) -> None:
        """Clear all ephys plot items and feature-plot interaction metadata."""
        self.feature_plot.clear()
        self.items.detach(self._figures())

    def feature_y_from_scene(self, scene_pos: Any) -> float | None:
        """Map a scene position to feature-space y in micrometres."""
        return self.feature_plot.feature_y_from_scene(scene_pos)

    def cluster_index_for_plot_x(self, x_value: float) -> int | None:
        """Return the cluster index represented by a plotted x coordinate."""
        return self.feature_plot.cluster_index_for_plot_x(x_value)

    def render_scatter(self, data: Any) -> None:
        """Render a 2D scatter plot with electrophysiology data."""
        if not data:
            logger.warning("data for this plot not available")
            return

        self.items.clear_image(self.plots.image, self.plots.image_colorbar)

        color_bar = ColorBar(data["cmap"])
        cbar = color_bar.makeColourBar(
            20,
            5,
            self.plots.image_colorbar,
            min=np.min(data["levels"][0]),
            max=np.max(data["levels"][1]),
            label=data["title"],
        )
        self.plots.image_colorbar.addItem(cbar)
        self.items.image_colorbars.append(cbar)

        plot = pg.ScatterPlotItem()
        plot.setData(
            x=data["x"],
            y=data["y"],
            symbol=data["symbol"].tolist(),
            size=data["size"].tolist(),
            brush=self._qt_colours(data["colours"]),
            pen=data["pen"],
        )

        self.plots.image.addItem(plot)
        self.plots.image.setXRange(
            min=data["xrange"][0],
            max=data["xrange"][1],
            padding=0,
        )
        self.set_axis(self.plots.image, "bottom", label=data["xaxis"])
        self.items.image_plots.append(plot)
        self.feature_plot.set_data_plot(
            plot,
            x_scale=1,
            y_scale=1,
            xrange=data["xrange"],
            cluster_x_values=data["x"] if data["cluster"] else None,
        )

        if data["cluster"]:
            self.feature_plot.connect_clicked(self.cluster_clicked)

    def render_line(self, data: Any) -> None:
        """Render a 1D line plot with electrophysiology data."""
        if not data:
            logger.warning("data for this plot not available")
            return

        self.items.clear_line(self.plots.line)
        line = pg.PlotCurveItem()
        line.setData(x=data["x"], y=data["y"])
        line.setPen(self.style.line_pen)
        self.plots.line.addItem(line)
        self.plots.line.setXRange(
            min=data["xrange"][0],
            max=data["xrange"][1],
            padding=0,
        )
        self.set_axis(self.plots.line, "bottom", label=data["xaxis"])
        self.items.line_plots.append(line)

    def render_probe(self, data: Any, bounds: Any = None) -> None:
        """Render a 2D image using the probe geometry layout."""
        if not data:
            logger.warning("data for this plot not available")
            return

        self.items.clear_probe(self.plots.probe, self.plots.probe_colorbar)
        self.set_axis(self.plots.probe_colorbar, "top", pen="w")
        color_bar = ColorBar(data["cmap"])
        lut = color_bar.getColourMap()
        for img, scale, offset in zip(data["img"], data["scale"], data["offset"]):
            image = pg.ImageItem()
            image.setImage(img)
            image.setTransform(QtGui.QTransform(*self._transform(scale, offset)))
            image.setLookupTable(lut)
            image.setLevels((data["levels"][0], data["levels"][1]))
            self.plots.probe.addItem(image)
            self.items.probe_plots.append(image)

        cbar = color_bar.makeColourBar(
            20,
            5,
            self.plots.probe_colorbar,
            min=data["levels"][0],
            max=data["levels"][1],
            label=data["title"],
            lim=True,
        )
        self.plots.probe_colorbar.addItem(cbar)
        self.items.probe_colorbars.append(cbar)

        self.plots.probe.setXRange(
            min=data["xrange"][0],
            max=data["xrange"][1],
            padding=0,
        )
        self.set_axis(self.plots.probe, "bottom", pen="w", label="blank")
        if bounds is not None:
            for bound in bounds:
                line = pg.InfiniteLine(pos=bound, angle=0, pen="w")
                self.plots.probe.addItem(line)
                self.items.probe_bounds.append(line)

    def render_image(self, data: Any) -> None:
        """Render a 2D image with electrophysiology data."""
        if not data:
            logger.warning("data for this plot not available")
            return

        self.items.clear_image(self.plots.image, self.plots.image_colorbar)
        self.set_axis(self.plots.image_colorbar, "top", pen="w")

        image = pg.ImageItem()
        img_data = data["img"]
        if img_data.ndim == 3:
            image.setImage(img_data, autoLevels=False)
        else:
            image.setImage(img_data)
        image.setTransform(
            QtGui.QTransform(*self._transform(data["scale"], data["offset"]))
        )
        cmap = data.get("cmap")
        if cmap:
            color_bar = ColorBar(data["cmap"])
            lut = color_bar.getColourMap()
            image.setLookupTable(lut)
            image.setLevels((data["levels"][0], data["levels"][1]))
            cbar = color_bar.makeColourBar(
                20,
                5,
                self.plots.image_colorbar,
                min=data["levels"][0],
                max=data["levels"][1],
                label=data["title"],
            )
            self.plots.image_colorbar.addItem(cbar)
            self.items.image_colorbars.append(cbar)
        elif img_data.ndim == 3:
            cbar_img = self._phase_legend_item()
            self.plots.image_colorbar.addItem(cbar_img)
            self.items.image_colorbars.append(cbar_img)
            self.set_axis(
                self.plots.image_colorbar,
                "top",
                pen="w",
                label="phase ↑  coherence ↓",
            )
        else:
            image.setLevels((1, 0))

        self.plots.image.addItem(image)
        self.items.image_plots.append(image)
        self.plots.image.setXRange(
            min=data["xrange"][0],
            max=data["xrange"][1],
            padding=0,
        )
        self.set_axis(self.plots.image, "bottom", label=data["xaxis"])
        self.feature_plot.set_data_plot(
            image,
            x_scale=data["scale"][0],
            y_scale=data["scale"][1],
            xrange=data["xrange"],
        )

    def _figures(self) -> dict[str, Any]:
        return {
            "img": self.plots.image,
            "img_cb": self.plots.image_colorbar,
            "line": self.plots.line,
            "probe": self.plots.probe,
            "probe_cb": self.plots.probe_colorbar,
        }

    @staticmethod
    def _transform(scale: Any, offset: Any) -> list[float]:
        return [
            scale[0],
            0.0,
            0.0,
            0.0,
            scale[1],
            0.0,
            offset[0],
            offset[1],
            1.0,
        ]

    @staticmethod
    def _qt_colours(colours: Any) -> list[Any]:
        """Convert plain color payloads into Qt colors for pyqtgraph brushes."""
        values = np.asarray(colours, dtype=object).tolist()
        converted = []
        for value in values:
            if isinstance(value, QtGui.QColor):
                converted.append(value)
            elif isinstance(value, str):
                converted.append(QtGui.QColor(value))
            elif isinstance(value, (tuple, list)):
                converted.append(QtGui.QColor(*[int(channel) for channel in value]))
            else:
                converted.append(value)
        return converted

    @staticmethod
    def _phase_legend_item() -> Any:
        from matplotlib.colors import hsv_to_rgb

        n = 256
        bar_h = 10

        hsv_phase = np.zeros((bar_h, n, 3))
        hsv_phase[:, :, 0] = np.linspace(0, 1, n)[None, :]
        hsv_phase[:, :, 1] = 1.0
        hsv_phase[:, :, 2] = 1.0
        rgb_phase = (hsv_to_rgb(hsv_phase) * 255).astype(np.uint8)

        hsv_sat = np.zeros((bar_h, n, 3))
        hsv_sat[:, :, 0] = 0.0
        hsv_sat[:, :, 1] = np.linspace(0, 1, n)[None, :]
        hsv_sat[:, :, 2] = 1.0
        rgb_sat = (hsv_to_rgb(hsv_sat) * 255).astype(np.uint8)

        combined = np.concatenate([rgb_phase, rgb_sat], axis=0).transpose(1, 0, 2)
        cbar_img = pg.ImageItem()
        cbar_img.setImage(combined, autoLevels=False)
        return cbar_img


def _set_depth_range(plot: Any, depth_view: Any, padding: float) -> None:
    y_min, y_max = depth_view.plot_y_range_um
    plot.setYRange(min=y_min, max=y_max, padding=padding)


def _add_depth_guides(
    plot: Any,
    depth_view: Any,
    *,
    depth_guide_pen: Any,
    probe_tip_lines: list[Any],
    probe_top_lines: list[Any],
) -> None:
    probe_tip_lines.append(
        plot.addLine(y=depth_view.probe_tip_um, pen=depth_guide_pen, z=50)
    )
    probe_top_lines.append(
        plot.addLine(y=depth_view.probe_top_um, pen=depth_guide_pen, z=50)
    )
