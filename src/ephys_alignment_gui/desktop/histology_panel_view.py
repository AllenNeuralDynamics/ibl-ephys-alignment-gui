"""Desktop pyqtgraph view/layer for histology region panels."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui

from ephys_alignment_gui.alignment_read_models import (
    FitPlotRenderState,
    HistologyPanelRenderState,
    NearbyBoundaryRenderState,
    ProbeExtentRenderState,
    ScaleFactorRenderState,
)
from ephys_alignment_gui.desktop.plot_elements import ColorBar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistologyPanelPlots:
    """Pyqtgraph plot handles owned by the desktop histology panel."""

    aligned: Any
    reference: Any
    scale: Any | None = None
    scale_colorbar: Any | None = None


@dataclass(frozen=True)
class FitPanelItems:
    """Pyqtgraph items owned by the desktop fit panel."""

    fit_curve: Any
    fit_scatter: Any
    linear_fit_curve: Any


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
        self.set_axis(self.plots.aligned, "bottom", pen="w", label="blank")

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
        self.set_axis(self.plots.reference, "bottom", pen="w", label="blank")

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

        self.set_axis(fig, "bottom", label="dist to boundary (um)")
        fig.setXRange(min=0, max=100)
        fig.setYRange(
            min=state.probe_extent.probe_tip_um - state.probe_extent.probe_extra_um,
            max=state.probe_extent.probe_top_um + state.probe_extent.probe_extra_um,
            padding=self.padding_provider(),
        )

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

        self.plots.scale.setYRange(
            min=state.probe_extent.probe_tip_um - state.probe_extent.probe_extra_um,
            max=state.probe_extent.probe_top_um + state.probe_extent.probe_extra_um,
            padding=self.padding_provider(),
        )
        self.set_axis(self.plots.scale, "bottom", pen="w", label="blank")
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
