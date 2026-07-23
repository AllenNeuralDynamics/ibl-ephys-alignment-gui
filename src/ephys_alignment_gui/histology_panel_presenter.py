"""Desktop presenter/view adapter for histology region panels."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistologyPanelPlots:
    """Pyqtgraph plot handles owned by the desktop histology panel."""

    aligned: Any
    reference: Any


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
class HistologyPanelPresenter:
    """Render histology region state into the desktop pyqtgraph panels."""

    plots: HistologyPanelPlots
    axes: HistologyPanelAxes
    style: HistologyPanelStyle
    session_provider: Callable[[], Any]
    histology_exists: Callable[[], bool]
    set_axis: Callable[..., Any]
    tip_line_moved: Callable[[], None]
    top_line_moved: Callable[[], None]
    padding_provider: Callable[[], float]

    def plot_aligned(self, fig: Any | None = None, *, movable: bool = True) -> None:
        """Plot aligned histology regions and editable probe extent lines."""
        if not self.histology_exists():
            return

        session = self._session()
        fig = self.plots.aligned if fig is None else fig
        fig.clear()
        session.hist_label_items = []
        self.set_axis(self.plots.aligned, "bottom", pen="w", label="blank")

        session.hist_regions = self._plot_region_bands(
            fig,
            session.hist_data,
            session.hist_label_items,
        )
        session.selected_region = self._default_selected_region(session.hist_regions)
        self._add_probe_extent_lines(
            session,
            fig,
            movable=movable,
            connect_tip_top=True,
        )

    def plot_reference(self, fig: Any | None = None, *, movable: bool = False) -> None:
        """Plot original/reference histology regions and probe extent lines."""
        if not self.histology_exists():
            return

        session = self._session()
        fig = self.plots.reference if fig is None else fig
        fig.clear()
        session.hist_ref_label_items = []
        self.set_axis(self.plots.reference, "bottom", pen="w", label="blank")

        session.hist_ref_regions = self._plot_region_bands(
            fig,
            session.hist_data_ref,
            session.hist_ref_label_items,
        )
        self._add_probe_extent_lines(
            session,
            fig,
            movable=movable,
            connect_tip_top=False,
        )

    def plot_nearby(self, fig: Any | None = None, *, movable: bool = False) -> None:
        """Plot nearby-region boundary distances in the reference panel."""
        if not self.histology_exists():
            return

        session = self._session()
        fig = self.plots.reference if fig is None else fig
        fig.clear()
        session.hist_ref_regions = np.empty((0, 1), dtype=object)

        self.set_axis(fig, "bottom", label="dist to boundary (um)")
        fig.setXRange(min=0, max=100)
        fig.setYRange(
            min=session.probe_tip - session.probe_extra,
            max=session.probe_top + session.probe_extra,
            padding=self.padding_provider(),
        )

        self._plot_nearby_region_curves(
            fig,
            session.hist_nearby_x,
            session.hist_nearby_y,
            session.hist_nearby_col,
            alpha=None,
        )
        self._plot_nearby_region_curves(
            fig,
            session.hist_nearby_parent_x,
            session.hist_nearby_parent_y,
            session.hist_nearby_parent_col,
            alpha=70,
        )
        self._add_probe_extent_lines(
            session,
            fig,
            movable=movable,
            connect_tip_top=False,
        )

    def toggle_labels(self) -> None:
        """Toggle atlas label axis visibility for both histology panels."""
        session = self._session()
        session.label_status = not session.label_status
        if not session.label_status:
            pen = None
            text_pen = None
        else:
            pen = "k"
            text_pen = "k"

        for axis in (self.axes.reference, self.axes.aligned):
            axis.setPen(pen)
            axis.setTextPen(text_pen)
        self.plots.reference.update()
        self.plots.aligned.update()

    def _plot_region_bands(
        self,
        fig: Any,
        hist_data: dict[str, list[Any]],
        label_items: list[Any],
    ) -> np.ndarray:
        regions = np.empty((0, 1), dtype=object)
        for ir, reg in enumerate(hist_data["region"]):
            colour = QtGui.QColor(*hist_data["colour"][ir])
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
            label_text = hist_data["axis_label"][ir][1]
            text_item = pg.TextItem(
                text=label_text,
                anchor=(0.5, 0.5),
                color="white",
            )
            text_item.setPos(0, region_center_y)
            fig.addItem(text_item)
            label_items.append(text_item)

        if len(hist_data["region"]) > 0:
            final_boundary = pg.InfiniteLine(
                pos=hist_data["region"][-1][1],
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
        session: Any,
        fig: Any,
        *,
        movable: bool,
        connect_tip_top: bool,
    ) -> None:
        if connect_tip_top:
            self._disconnect_tip_top(session)

        session.tip_pos = pg.InfiniteLine(
            pos=session.probe_tip,
            angle=0,
            pen=self.style.dotted_pen,
            movable=movable,
        )
        session.top_pos = pg.InfiniteLine(
            pos=session.probe_top,
            angle=0,
            pen=self.style.dotted_pen,
            movable=movable,
        )

        if connect_tip_top:
            self._set_aligned_probe_extent_bounds(session)
            session.tip_pos.sigPositionChanged.connect(self.tip_line_moved)
            session.top_pos.sigPositionChanged.connect(self.top_line_moved)

        fig.addItem(session.tip_pos)
        fig.addItem(session.top_pos)

    def _set_aligned_probe_extent_bounds(self, session: Any) -> None:
        offset = 1
        feature_min_um = session.features[session.idx][0] * 1e6
        feature_max_um = session.features[session.idx][-1] * 1e6
        feature_top_um = feature_max_um - offset

        if session.probe_top > feature_top_um:
            logger.warning(
                "Probe span (%.0f um) exceeds feature range (%.0f um). "
                "Using safe fallback bounds. Consider recording with larger "
                "channel span or adjusting initialization range.",
                session.probe_top,
                feature_top_um,
            )
            fallback_bounds = (feature_min_um + offset, feature_max_um - offset)
            session.tip_pos.setBounds(fallback_bounds)
            session.top_pos.setBounds(fallback_bounds)
            return

        session.tip_pos.setBounds(
            (
                feature_min_um + offset,
                feature_max_um - (session.probe_top + offset),
            )
        )
        session.top_pos.setBounds(
            (
                feature_min_um + (session.probe_top + offset),
                feature_max_um - offset,
            )
        )

    @staticmethod
    def _disconnect_tip_top(session: Any) -> None:
        for item in (session.tip_pos, session.top_pos):
            if item is None:
                continue
            try:
                item.sigPositionChanged.disconnect()
            except TypeError:
                pass

    @staticmethod
    def _default_selected_region(regions: np.ndarray) -> Any:
        if regions.size == 0:
            return None
        if regions.shape[0] >= 2:
            return regions[-2, 0]
        return regions[-1, 0]

    def _session(self) -> Any:
        session = self.session_provider()
        if session is None:
            raise RuntimeError("Histology panel session is not available")
        return session
