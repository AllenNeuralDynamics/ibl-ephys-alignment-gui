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
from ephys_alignment_gui.plotting.raster_request import ImageRasterRequest

logger = logging.getLogger(__name__)


_EPHYS_COLORBAR_MAX_HEIGHT = 90
_SCALAR_COLORBAR_AXIS_HEIGHT = 42
_SCALAR_COLORBAR_WIDTH = 20
_SCALAR_COLORBAR_HEIGHT = 5
_PHASE_LEGEND_AXIS_HEIGHT = 52
_PHASE_LEGEND_WIDTH = 1.0
_PHASE_LEGEND_HEIGHT = 2.0


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
    empty_state_item: tuple[Any, Callable[..., Any]] | None = None

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
        image_colorbar.setMaximumHeight(_EPHYS_COLORBAR_MAX_HEIGHT)
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
        probe_colorbar.setMaximumHeight(_EPHYS_COLORBAR_MAX_HEIGHT)
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

    def image_raster_request(self) -> ImageRasterRequest:
        """Return the current feature image plot raster target."""
        width_px = (
            _dimension(self.plots.image, "width")
            - _dimension(self.widgets.image_axis, "width")
        )
        return ImageRasterRequest.from_plot_size(
            width_px=max(1.0, width_px),
            height_px=max(1.0, _dimension(self.plots.image, "height")),
        )

    def feature_y_range(self) -> tuple[float, float]:
        """Return the current feature-depth y-range."""
        y_min, y_max = self.plots.image.viewRange()[1]
        return float(y_min), float(y_max)

    def clear(self) -> None:
        """Clear all ephys plot items and feature-plot interaction metadata."""
        self.feature_plot.clear()
        self.items.detach(self._figures())

    def show_empty_state(self, text: str = "Select and load data") -> None:
        """Show a centered placeholder in the feature image plot."""
        if self.empty_state_item is not None:
            return
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=(160, 160, 160))
        view_box = self.plots.image.getViewBox()
        view_box.addItem(item, ignoreBounds=True)

        def _center(*_args: Any) -> None:
            (x0, x1), (y0, y1) = view_box.viewRange()
            item.setPos((x0 + x1) / 2.0, (y0 + y1) / 2.0)

        _center()
        view_box.sigRangeChanged.connect(_center)
        self.empty_state_item = (item, _center)

    def clear_empty_state(self) -> None:
        """Remove the feature image placeholder if it is visible."""
        if self.empty_state_item is None:
            return
        item, center = self.empty_state_item
        view_box = self.plots.image.getViewBox()
        try:
            view_box.sigRangeChanged.disconnect(center)
        except (TypeError, RuntimeError):
            pass
        view_box.removeItem(item)
        self.empty_state_item = None

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
            _SCALAR_COLORBAR_WIDTH,
            _SCALAR_COLORBAR_HEIGHT,
            self.plots.image_colorbar,
            min=np.min(data["levels"][0]),
            max=np.max(data["levels"][1]),
            label=data["title"],
            axis_height=_SCALAR_COLORBAR_AXIS_HEIGHT,
        )
        self.plots.image_colorbar.addItem(cbar)
        self.items.image_colorbars.append(cbar)

        plot = pg.ScatterPlotItem()
        plot.setData(
            x=data["x"],
            y=data["y"],
            symbol=data["symbol"].tolist(),
            size=data["size"].tolist(),
            brush=self._scatter_brushes(
                color_bar=color_bar,
                colours=data["colours"],
                levels=data["levels"],
            ),
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
            _SCALAR_COLORBAR_WIDTH,
            _SCALAR_COLORBAR_HEIGHT,
            self.plots.probe_colorbar,
            min=data["levels"][0],
            max=data["levels"][1],
            label=data["title"],
            lim=True,
            axis_height=_SCALAR_COLORBAR_AXIS_HEIGHT,
            edge_tick_padding=1.0,
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

        # A payload may carry one image or several (one per recording block,
        # each with its own depth pitch). Normalise to lists.
        imgs = data["img"]
        scales = data["scale"]
        offsets = data["offset"]
        no_data_masks = data.get("no_data_mask")
        if not isinstance(imgs, list):
            imgs, scales, offsets = [imgs], [scales], [offsets]
            no_data_masks = [no_data_masks] if no_data_masks is not None else [None]
        elif no_data_masks is None:
            no_data_masks = [None] * len(imgs)
        elif not isinstance(no_data_masks, list):
            no_data_masks = [no_data_masks]

        first_image = None
        for img_data, scale, offset, no_data_mask in zip(
            imgs,
            scales,
            offsets,
            no_data_masks,
        ):
            image = pg.ImageItem()
            if img_data.ndim == 3:
                image.setImage(img_data, autoLevels=False)
            else:
                image.setImage(img_data)
            image.setTransform(QtGui.QTransform(*self._transform(scale, offset)))
            if data.get("cmap"):
                color_bar = ColorBar(data["cmap"])
                image.setLookupTable(color_bar.getColourMap())
                image.setLevels((data["levels"][0], data["levels"][1]))
            elif img_data.ndim != 3:
                image.setLevels((1, 0))

            self.plots.image.addItem(image)
            self.items.image_plots.append(image)
            overlay = self._no_data_overlay(
                no_data_mask,
                data.get("no_data_color", (145, 158, 170, 210)),
            )
            if overlay is not None:
                overlay_item = pg.ImageItem()
                overlay_item.setImage(overlay, autoLevels=False)
                overlay_item.setTransform(QtGui.QTransform(*self._transform(scale, offset)))
                self.plots.image.addItem(overlay_item)
                self.items.image_plots.append(overlay_item)
            if first_image is None:
                first_image = image
                first_scale = scale

        # Colour bar is shared across blocks: levels are common by construction.
        if data.get("cmap"):
            cbar = ColorBar(data["cmap"]).makeColourBar(
                _SCALAR_COLORBAR_WIDTH,
                _SCALAR_COLORBAR_HEIGHT,
                self.plots.image_colorbar,
                min=data["levels"][0],
                max=data["levels"][1],
                label=data["title"],
                axis_height=_SCALAR_COLORBAR_AXIS_HEIGHT,
            )
            self.plots.image_colorbar.addItem(cbar)
            self.items.image_colorbars.append(cbar)
        elif imgs[0].ndim == 3:
            cbar_img = self._phase_legend_item()
            self.plots.image_colorbar.addItem(cbar_img)
            self.items.image_colorbars.append(cbar_img)
            self._configure_phase_legend_axis()

        self.plots.image.setXRange(
            min=data["xrange"][0],
            max=data["xrange"][1],
            padding=0,
        )
        self.set_axis(self.plots.image, "bottom", label=data["xaxis"])
        self.feature_plot.set_data_plot(
            first_image,
            x_scale=first_scale[0],
            y_scale=first_scale[1],
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
    def _no_data_overlay(mask: Any, color: Any) -> np.ndarray | None:
        """Return an RGBA overlay for image bins outside the sampled channel support."""
        if mask is None:
            return None

        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.size == 0 or not mask_array.any():
            return None

        rgba = np.asarray(color, dtype=np.uint8).ravel()
        if rgba.size == 3:
            rgba = np.concatenate([rgba, np.array([220], dtype=np.uint8)])
        if rgba.size != 4:
            rgba = np.array([145, 158, 170, 210], dtype=np.uint8)

        overlay = np.zeros(mask_array.shape + (4,), dtype=np.uint8)
        overlay[mask_array] = rgba
        return overlay

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
    def _scatter_brushes(*, color_bar: ColorBar, colours: Any, levels: Any) -> list[Any]:
        """Return brushes for literal colors or scalar values mapped through a LUT."""
        colour_values = np.asarray(colours)
        if colour_values.dtype.kind in "fiu" and colour_values.ndim <= 1:
            values = colour_values.astype(float, copy=False).ravel()
            brush_levels = DesktopEphysPanelView._numeric_colour_levels(
                levels,
                values,
            )
            safe_values = np.where(np.isfinite(values), values, brush_levels[0])
            return color_bar.getBrush(safe_values, levels=brush_levels)

        return DesktopEphysPanelView._qt_colours(colours)

    @staticmethod
    def _numeric_colour_levels(levels: Any, values: np.ndarray) -> list[float]:
        """Return a finite, non-degenerate color range for scalar scatter values."""
        level_values = np.asarray(levels, dtype=float).ravel()
        if level_values.size >= 2:
            lo, hi = float(level_values[0]), float(level_values[1])
        else:
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                return [0.0, 1.0]
            lo = float(np.min(finite_values))
            hi = float(np.max(finite_values))

        if not np.isfinite(lo) or not np.isfinite(hi):
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                return [0.0, 1.0]
            lo = float(np.min(finite_values))
            hi = float(np.max(finite_values))

        if lo == hi:
            hi = lo + 1.0

        return [lo, hi]

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
        cbar_img.setTransform(
            QtGui.QTransform(
                _PHASE_LEGEND_WIDTH / n,
                0.0,
                0.0,
                0.0,
                _PHASE_LEGEND_HEIGHT / (bar_h * 2),
                0.0,
                0.0,
                0.0,
                1.0,
            )
        )
        return cbar_img

    def _configure_phase_legend_axis(self) -> None:
        self.plots.image_colorbar.setXRange(
            min=0.0,
            max=_PHASE_LEGEND_WIDTH,
            padding=0,
        )
        self.plots.image_colorbar.setYRange(
            min=0.0,
            max=_PHASE_LEGEND_HEIGHT,
            padding=0,
        )
        axis = self.set_axis(
            self.plots.image_colorbar,
            "top",
            pen="k",
            label="phase (rad) / coherence",
        )
        if axis is None:
            return
        axis.setHeight(_PHASE_LEGEND_AXIS_HEIGHT)
        axis.setTicks(
            [
                [
                    (0.0, "0"),
                    (_PHASE_LEGEND_WIDTH / 2, "pi"),
                    (_PHASE_LEGEND_WIDTH, "2pi"),
                ],
                [
                    (0.0, "coh 0"),
                    (_PHASE_LEGEND_WIDTH, "coh 1"),
                ],
            ]
        )


def _set_depth_range(plot: Any, depth_view: Any, padding: float) -> None:
    y_min, y_max = depth_view.plot_y_range_um
    plot.setYRange(min=y_min, max=y_max, padding=padding)


def _dimension(item: Any, method_name: str) -> float:
    method = getattr(item, method_name, None)
    if not callable(method):
        return 0.0
    try:
        return float(method())
    except Exception:
        return 0.0


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
