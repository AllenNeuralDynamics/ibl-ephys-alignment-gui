"""Desktop pyqtgraph view/layer for histology region panels."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets

from ephys_alignment_gui.core.alignment_read_models import (
    FitPlotRenderState,
    HistologyPanelRenderState,
    NearbyBoundaryRenderState,
    ProbeExtentRenderState,
    ScaleFactorRenderState,
)
from ephys_alignment_gui.desktop.displays.depth_panel_layout import (
    DEPTH_PANEL_HEADER_HEIGHT_PX,
    set_depth_panel_bottom_axis,
    set_depth_panel_header_height,
)
from ephys_alignment_gui.desktop.displays.plot_elements import ColorBar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistologyPanelPlots:
    """Pyqtgraph plot handles owned by the desktop histology panel."""

    aligned: Any
    reference: Any
    perpendicular: Any | None = None
    scale: Any | None = None
    scale_colorbar: Any | None = None
    area: Any | None = None
    layout: Any | None = None
    depth_ruler: Any | None = None
    scale_axis: Any | None = None


@dataclass(frozen=True)
class FitPanelItems:
    """Pyqtgraph items owned by the desktop fit panel."""

    fit_curve: Any
    fit_scatter: Any
    linear_fit_curve: Any
    plot_widget: Any | None = None
    linear_fit_checkbox: Any | None = None


@dataclass(frozen=True)
class HistologyPanelAxes:
    """Axis handles controlled by the desktop histology panel."""

    aligned: Any
    reference: Any


@dataclass(frozen=True)
class HistologyPanelStyle:
    """Pens and style objects used by the desktop histology panel."""

    dotted_pen: Any


@dataclass
class HistologyPanelView:
    """Render histology region state into desktop pyqtgraph panels."""

    plots: HistologyPanelPlots
    axes: HistologyPanelAxes
    style: HistologyPanelStyle
    set_axis: Callable[..., Any]
    padding_provider: Callable[[], float]
    fit_items: FitPanelItems | None = None
    label_status: bool = True
    tip_pos: Any = None
    top_pos: Any = None
    hist_regions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 1), dtype=object)
    )
    hist_ref_regions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 1), dtype=object)
    )
    scale_regions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 1), dtype=object)
    )
    scale_factor: Any = None
    selected_region: Any = None
    hist_label_items: list[Any] = field(default_factory=list)
    hist_ref_label_items: list[Any] = field(default_factory=list)
    _probe_extent: ProbeExtentRenderState | None = None

    @classmethod
    def create(
        cls,
        *,
        depth_view: Any,
        padding: float,
        set_axis: Callable[..., Any],
        dotted_pen: Any,
        fit_pen: Any,
        linear_fit_pen: Any,
        baseline_pen: Any,
        perpendicular_plot: Any,
        linear_fit_enabled: Callable[[], bool],
        on_linear_fit_changed: Callable[..., Any],
        on_mouse_double_clicked: Callable[..., Any],
        on_mouse_hover: Callable[..., Any],
    ) -> HistologyPanelView:
        """Create the histology/scale/fit panel and all of its plot handles."""
        aligned = pg.PlotItem()
        aligned.setContentsMargins(0, 0, 0, 0)
        aligned.setMouseEnabled(x=False)
        _set_depth_range(aligned, depth_view, padding)
        set_depth_panel_bottom_axis(
            aligned,
            set_axis,
            label="Warped annotations",
            ticks=False,
        )
        aligned_axis = set_axis(aligned, "left", show=False)

        scale = pg.PlotItem()
        scale.setMaximumWidth(50)
        scale.setMouseEnabled(x=False)
        set_depth_panel_bottom_axis(scale, set_axis, pen="w", ticks=False)
        set_axis(scale, "left", show=False)
        scale.setYLink(aligned)

        scale_colorbar = pg.PlotItem()
        scale_colorbar.setMouseEnabled(x=False, y=False)
        scale_colorbar.setMaximumHeight(DEPTH_PANEL_HEADER_HEIGHT_PX)
        set_axis(scale_colorbar, "bottom", show=False)
        set_axis(scale_colorbar, "left", show=False)
        scale_axis = set_axis(scale_colorbar, "top", pen="w")
        set_axis(scale_colorbar, "right", show=False)

        reference = pg.PlotItem()
        reference.setMouseEnabled(x=False)
        _set_depth_range(reference, depth_view, padding)
        reference.setYLink(aligned)
        set_depth_panel_bottom_axis(
            reference,
            set_axis,
            label="Original annotations",
            ticks=False,
        )
        set_axis(reference, "left", show=False)
        reference_axis = set_axis(reference, "right", show=False)

        perpendicular_plot.setYLink(aligned)

        area = pg.GraphicsLayoutWidget()
        area.setMouseTracking(True)
        area.scene().sigMouseClicked.connect(on_mouse_double_clicked)
        area.scene().sigMouseHover.connect(on_mouse_hover)

        depth_ruler = pg.PlotItem()
        depth_ruler.setMouseEnabled(x=False, y=False)
        depth_ruler.setMaximumWidth(48)
        _set_depth_range(depth_ruler, depth_view, padding)
        set_depth_panel_bottom_axis(depth_ruler, set_axis, pen="w", ticks=False)
        set_axis(depth_ruler, "right", show=False)
        depth_axis = set_axis(depth_ruler, "left", pen="k")
        depth_axis.setWidth(44)

        layout = pg.GraphicsLayout()
        layout.addItem(scale_colorbar, 0, 0, 1, 5)
        layout.addItem(depth_ruler, 1, 0)
        layout.addItem(aligned, 1, 1)
        layout.addItem(perpendicular_plot, 1, 2)
        layout.addItem(scale, 1, 3)
        layout.addItem(reference, 1, 4)
        layout.layout.setColumnStretchFactor(0, 1)
        layout.layout.setColumnStretchFactor(1, 4)
        layout.layout.setColumnStretchFactor(2, 5)
        layout.layout.setColumnStretchFactor(3, 1)
        layout.layout.setColumnStretchFactor(4, 4)
        layout.layout.setRowStretchFactor(0, 1)
        layout.layout.setRowStretchFactor(1, 10)
        set_depth_panel_header_height(layout)
        area.addItem(layout)

        fit_plot = pg.PlotWidget(background="w")
        fit_plot.setMouseEnabled(x=False, y=False)
        view_min, view_max = depth_view.view_range_um
        fit_plot.setXRange(min=view_min, max=view_max)
        fit_plot.setYRange(min=view_min, max=view_max)
        set_axis(fit_plot, "bottom", label="Ephys reference depth (μm)")
        set_axis(fit_plot, "left", label="Atlas reference depth (μm)")
        baseline = pg.PlotCurveItem()
        baseline.setData(
            x=depth_view.fit_depth_um,
            y=depth_view.fit_depth_um,
            pen=baseline_pen,
        )
        fit_curve = pg.PlotCurveItem(pen=fit_pen)
        fit_scatter = pg.ScatterPlotItem(size=7, symbol="o", brush="w", pen="b")
        linear_fit_curve = pg.PlotCurveItem(pen=linear_fit_pen)
        fit_plot.addItem(baseline)
        fit_plot.addItem(fit_curve)
        fit_plot.addItem(linear_fit_curve)
        fit_plot.addItem(fit_scatter)

        linear_fit_checkbox = QtWidgets.QCheckBox("Linear fit", fit_plot)
        linear_fit_checkbox.setChecked(linear_fit_enabled())
        linear_fit_checkbox.stateChanged.connect(on_linear_fit_changed)
        fit_items = FitPanelItems(
            fit_curve=fit_curve,
            fit_scatter=fit_scatter,
            linear_fit_curve=linear_fit_curve,
            plot_widget=fit_plot,
            linear_fit_checkbox=linear_fit_checkbox,
        )
        fit_plot.sigDeviceRangeChanged.connect(
            lambda *args: position_linear_fit_checkbox(fit_items)
        )
        position_linear_fit_checkbox(fit_items)

        return cls(
            plots=HistologyPanelPlots(
                aligned=aligned,
                reference=reference,
                perpendicular=perpendicular_plot,
                scale=scale,
                scale_colorbar=scale_colorbar,
                area=area,
                layout=layout,
                depth_ruler=depth_ruler,
                scale_axis=scale_axis,
            ),
            axes=HistologyPanelAxes(
                aligned=aligned_axis,
                reference=reference_axis,
            ),
            style=HistologyPanelStyle(dotted_pen=dotted_pen),
            set_axis=set_axis,
            padding_provider=lambda: padding,
            fit_items=fit_items,
        )

    @property
    def fit_plot(self) -> Any:
        """Return the fit plot widget owned by this panel."""
        return None if self.fit_items is None else self.fit_items.plot_widget

    @property
    def linear_fit_checkbox(self) -> Any:
        """Return the linear-fit checkbox owned by this panel."""
        if self.fit_items is None:
            return None
        return self.fit_items.linear_fit_checkbox

    def warped_feature_y_from_scene(self, scene_pos: Any) -> float | None:
        """Map a warped-panel scene position to displayed feature depth in um."""
        for plot in (self.plots.aligned, self.plots.perpendicular):
            y_pos = self._plot_y_from_scene(plot, scene_pos)
            if y_pos is not None:
                return y_pos
        return None

    @staticmethod
    def _plot_y_from_scene(plot: Any, scene_pos: Any) -> float | None:
        if plot is None:
            return None
        view_box = getattr(plot, "vb", None)
        if view_box is None:
            get_view_box = getattr(plot, "getViewBox", None)
            view_box = get_view_box() if callable(get_view_box) else None
        if view_box is None:
            return None

        scene_rect = getattr(view_box, "sceneBoundingRect", None)
        if callable(scene_rect):
            try:
                rect = scene_rect()
                contains = getattr(rect, "contains", None)
                if callable(contains) and not contains(scene_pos):
                    return None
            except (AttributeError, RuntimeError, TypeError):
                pass

        map_scene_to_view = getattr(view_box, "mapSceneToView", None)
        if not callable(map_scene_to_view):
            return None
        pos = map_scene_to_view(scene_pos)
        y = getattr(pos, "y", None)
        return float(y()) if callable(y) else None

    def clear(self) -> None:
        """Clear histology-panel plot items and forget desktop handles."""
        self._disconnect_tip_top()
        self.plots.aligned.clear()
        self.plots.reference.clear()
        if self.plots.scale is not None:
            self.plots.scale.clear()
        if self.plots.scale_colorbar is not None:
            self.plots.scale_colorbar.clear()

        self.clear_fit()
        self.tip_pos = None
        self.top_pos = None
        self.hist_regions = np.empty((0, 1), dtype=object)
        self.hist_ref_regions = np.empty((0, 1), dtype=object)
        self.scale_regions = np.empty((0, 1), dtype=object)
        self.scale_factor = None
        self.selected_region = None
        self.hist_label_items = []
        self.hist_ref_label_items = []
        self._probe_extent = None

    def clear_fit(self) -> None:
        """Clear fit-panel data while preserving the persistent plot items."""
        if self.fit_items is None:
            return
        self.fit_items.fit_curve.setData()
        self.fit_items.fit_scatter.setData()
        self.fit_items.linear_fit_curve.setData()

    def render_aligned(
        self,
        state: HistologyPanelRenderState,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> None:
        """Plot aligned histology regions and editable probe extent lines."""
        fig = self.plots.aligned if fig is None else fig
        fig.clear()
        self.hist_label_items = []
        set_depth_panel_bottom_axis(
            self.plots.aligned,
            self.set_axis,
            label="Warped annotations",
            ticks=False,
        )

        self.hist_regions = self._plot_region_bands(
            fig,
            state.histology.histology,
            self.hist_label_items,
        )
        self.selected_region = self._default_selected_region(self.hist_regions)
        self._add_probe_extent_lines(
            state.probe_extent,
            fig,
            movable=movable,
            connect_tip_top=True,
        )

    def render_reference(
        self,
        state: HistologyPanelRenderState,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> None:
        """Plot original/reference histology regions and probe extent lines."""
        fig = self.plots.reference if fig is None else fig
        fig.clear()
        self.hist_ref_label_items = []
        set_depth_panel_bottom_axis(
            self.plots.reference,
            self.set_axis,
            label="Original annotations",
            ticks=False,
        )

        self.hist_ref_regions = self._plot_region_bands(
            fig,
            state.histology.reference_histology,
            self.hist_ref_label_items,
        )
        self._add_probe_extent_lines(
            state.probe_extent,
            fig,
            movable=movable,
            connect_tip_top=False,
        )

    def render_nearby(
        self,
        state: NearbyBoundaryRenderState,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> None:
        """Plot nearby-region boundary distances in the reference panel."""
        fig = self.plots.reference if fig is None else fig
        fig.clear()
        self.hist_ref_regions = np.empty((0, 1), dtype=object)

        set_depth_panel_bottom_axis(
            fig,
            self.set_axis,
            label="dist to boundary (um)",
        )
        fig.setXRange(min=0, max=100)

        self._plot_nearby_region_curves(
            fig,
            state.x,
            state.y,
            state.colours,
            alpha=None,
        )
        self._plot_nearby_region_curves(
            fig,
            state.parent_x,
            state.parent_y,
            state.parent_colours,
            alpha=70,
        )
        self._add_probe_extent_lines(
            state.probe_extent,
            fig,
            movable=movable,
            connect_tip_top=False,
        )

    def render_scale_factor(
        self,
        state: ScaleFactorRenderState,
        *,
        y_range: tuple[float, float],
    ) -> None:
        """Render the scale-factor strip beside the histology panel."""
        if self.plots.scale is None or self.plots.scale_colorbar is None:
            return

        self.plots.scale.clear()
        self.scale_regions = np.empty((0, 1), dtype=object)
        self.scale_factor = state.scale
        scale = np.asarray(state.scale)
        regions = state.region
        scale_factor = scale - 0.5
        color_bar = ColorBar("seismic")
        cbar = color_bar.makeColourBar(
            20,
            5,
            self.plots.scale_colorbar,
            min=0.5,
            max=1.5,
            label="Scale Factor",
        )
        colours = color_bar.map.mapToQColor(scale_factor)
        y_min, y_max = y_range

        for ir, reg in enumerate(regions):
            region = pg.LinearRegionItem(
                values=(reg[0], reg[1]),
                orientation=pg.LinearRegionItem.Horizontal,
                brush=colours[ir],
                movable=False,
            )
            bound = pg.InfiniteLine(pos=reg[0], angle=0, pen=colours[ir])

            self.plots.scale.addItem(region)
            self.plots.scale.addItem(bound)
            self.scale_regions = np.vstack(
                [self.scale_regions, np.array([[region]], dtype=object)]
            )

            text_y = (max(y_min, reg[0]) + min(y_max, reg[1])) / 2
            text_item = pg.TextItem(
                text=f"{scale[ir]:.2f}",
                anchor=(0.5, 0.5),
                color="black",
            )
            text_item.setPos(-0.05, text_y)
            self.plots.scale.addItem(text_item)

        if len(regions) > 0:
            bound = pg.InfiniteLine(pos=regions[-1][1], angle=0, pen=colours[-1])
            self.plots.scale.addItem(bound)

        set_depth_panel_bottom_axis(
            self.plots.scale,
            self.set_axis,
            label="blank",
            pen="w",
            ticks=False,
        )
        self.plots.scale_colorbar.addItem(cbar)

    def render_fit(self, state: FitPlotRenderState) -> None:
        """Render feature/track fit curves."""
        if self.fit_items is None:
            return

        self.fit_items.fit_curve.setData(
            x=state.feature_um,
            y=state.track_um,
        )
        self.fit_items.fit_scatter.setData(
            x=state.feature_um,
            y=state.track_um,
        )
        if state.linear_feature_um is not None and state.linear_track_um is not None:
            self.fit_items.linear_fit_curve.setData(
                x=state.linear_feature_um,
                y=state.linear_track_um,
            )
        else:
            self.fit_items.linear_fit_curve.setData()

    def toggle_labels(self) -> None:
        """Toggle atlas label axis visibility for both histology panels."""
        self.set_labels_visible(not self.label_status)

    def set_labels_visible(self, visible: bool) -> None:
        """Set atlas label axis visibility for both histology panels."""
        self.label_status = visible
        pen = "k" if visible else None
        text_pen = "k" if visible else None

        for axis in (self.axes.reference, self.axes.aligned):
            axis.setPen(pen)
            axis.setTextPen(text_pen)
        self.plots.reference.update()
        self.plots.aligned.update()

    def _plot_region_bands(
        self,
        fig: Any,
        hist_data: Any,
        label_items: list[Any],
    ) -> np.ndarray:
        regions = np.empty((0, 1), dtype=object)
        for ir, reg in enumerate(hist_data.region):
            colour = QtGui.QColor(*hist_data.colour[ir])
            region = pg.LinearRegionItem(
                values=(reg[0], reg[1]),
                orientation=pg.LinearRegionItem.Horizontal,
                brush=colour,
                movable=False,
            )
            bound = pg.InfiniteLine(pos=reg[0], angle=0, pen="w")
            fig.addItem(region)
            fig.addItem(bound)
            regions = np.vstack([regions, np.array([[region]], dtype=object)])

            region_center_y = (reg[0] + reg[1]) / 2
            label_text = hist_data.axis_label[ir][1]
            text_item = pg.TextItem(
                text=label_text,
                anchor=(0.5, 0.5),
                color="white",
            )
            text_item.setPos(0, region_center_y)
            fig.addItem(text_item)
            label_items.append(text_item)

        if len(hist_data.region) > 0:
            final_boundary = pg.InfiniteLine(
                pos=hist_data.region[-1][1],
                angle=0,
                pen="w",
            )
            fig.addItem(final_boundary)
        return regions

    def _plot_nearby_region_curves(
        self,
        fig: Any,
        xs: Any,
        ys: Any,
        colours: Any,
        *,
        alpha: int | None,
    ) -> None:
        if xs is None or ys is None or colours is None:
            return

        for x, y, colour_value in zip(xs, ys, colours):
            colour = QtGui.QColor(colour_value)
            if alpha is not None:
                colour.setAlpha(alpha)
            plot = pg.PlotCurveItem()
            plot.setData(x=x, y=y * 1e6, fillLevel=10, fillOutline=True)
            plot.setBrush(colour)
            plot.setPen(colour)
            fig.addItem(plot)

    def _add_probe_extent_lines(
        self,
        probe_extent: ProbeExtentRenderState,
        fig: Any,
        *,
        movable: bool,
        connect_tip_top: bool,
    ) -> None:
        if connect_tip_top:
            self._disconnect_tip_top()

        tip_pos = pg.InfiniteLine(
            pos=probe_extent.probe_tip_um,
            angle=0,
            pen=self.style.dotted_pen,
            movable=movable,
        )
        top_pos = pg.InfiniteLine(
            pos=probe_extent.probe_top_um,
            angle=0,
            pen=self.style.dotted_pen,
            movable=movable,
        )

        if connect_tip_top:
            self._probe_extent = probe_extent
            self.tip_pos = tip_pos
            self.top_pos = top_pos
            self._set_aligned_probe_extent_bounds(probe_extent)
            self.tip_pos.sigPositionChanged.connect(self.sync_top_to_tip)
            self.top_pos.sigPositionChanged.connect(self.sync_tip_to_top)

        fig.addItem(tip_pos)
        fig.addItem(top_pos)

    def _set_aligned_probe_extent_bounds(
        self,
        probe_extent: ProbeExtentRenderState,
    ) -> None:
        feature_top_um = probe_extent.feature_max_um - 1.0
        if probe_extent.probe_top_um > feature_top_um:
            logger.warning(
                "Probe span (%.0f um) exceeds feature range (%.0f um). "
                "Using safe fallback bounds. Consider recording with larger "
                "channel span or adjusting initialization range.",
                probe_extent.probe_top_um,
                feature_top_um,
            )
        self.tip_pos.setBounds(probe_extent.tip_bounds_um)
        self.top_pos.setBounds(probe_extent.top_bounds_um)

    def sync_top_to_tip(self) -> None:
        """Keep the top line at the configured probe span above the tip line."""
        if self.tip_pos is None or self.top_pos is None or self._probe_extent is None:
            return
        self.top_pos.setPos(self.tip_pos.value() + self._probe_extent.probe_top_um)

    def sync_tip_to_top(self) -> None:
        """Keep the tip line at the configured probe span below the top line."""
        if self.tip_pos is None or self.top_pos is None or self._probe_extent is None:
            return
        self.tip_pos.setPos(self.top_pos.value() - self._probe_extent.probe_top_um)

    def tip_position_um(self) -> float | None:
        """Return the current editable tip-line position."""
        if self.tip_pos is None:
            return None
        return float(self.tip_pos.value())

    def select_region(self, item: Any) -> None:
        """Record the currently hovered/selected histology region item."""
        self.selected_region = item

    def selected_region_index(self) -> int | None:
        """Return the index of the selected histology/ref region."""
        if self.selected_region is None:
            return None
        idx = np.where(self.hist_regions == self.selected_region)[0]
        if idx.size == 0:
            idx = np.where(self.hist_ref_regions == self.selected_region)[0]
        if idx.size == 0:
            return None
        return int(idx[0])

    def scale_factor_for_region_item(self, item: Any) -> float | None:
        """Return the scale factor associated with a rendered scale-region item."""
        idx = np.where(self.scale_regions == item)[0]
        if idx.size == 0 or self.scale_factor is None:
            return None
        return float(self.scale_factor[int(idx[0])])

    def _disconnect_tip_top(self) -> None:
        for item in (self.tip_pos, self.top_pos):
            if item is None:
                continue
            try:
                item.sigPositionChanged.disconnect()
            except (TypeError, RuntimeError):
                pass

    @staticmethod
    def _default_selected_region(regions: np.ndarray) -> Any:
        if regions.size == 0:
            return None
        if regions.shape[0] >= 2:
            return regions[-2, 0]
        return regions[-1, 0]


def _set_depth_range(plot: Any, depth_view: Any, padding: float) -> None:
    y_min, y_max = depth_view.plot_y_range_um
    plot.setYRange(min=y_min, max=y_max, padding=padding)


def position_linear_fit_checkbox(fit_items: FitPanelItems) -> None:
    """Position the linear-fit checkbox inside the fit plot."""
    if fit_items.linear_fit_checkbox is None:
        return
    fit_items.linear_fit_checkbox.move(70, 10)
