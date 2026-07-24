"""Desktop pyqtgraph rendering for ephys data panels."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui

from ephys_alignment_gui.ephys_plot_items import EphysPlotItems
from ephys_alignment_gui.feature_plot_view import FeaturePlotView
from ephys_alignment_gui.plot_elements import ColorBar

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
class EphysPanelStyle:
    """Desktop styling handles for ephys panel rendering."""

    line_pen: Any


@dataclass
class DesktopEphysPanelView:
    """Own ephys pyqtgraph items and render ephys plot payloads."""

    plots: EphysPanelPlots
    style: EphysPanelStyle
    set_axis: Callable[..., Any]
    cluster_clicked: Callable[..., Any]
    items: EphysPlotItems = field(default_factory=EphysPlotItems)
    feature_plot: FeaturePlotView = field(default_factory=FeaturePlotView)

    @property
    def feature_xrange(self) -> Any:
        """Return the active feature-plot x-range, if one is known."""
        return self.feature_plot.xrange

    @property
    def probe_colorbars(self) -> list[Any]:
        """Return currently rendered probe colorbar items."""
        return self.items.probe_colorbars

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
            brush=data["colours"].tolist(),
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
