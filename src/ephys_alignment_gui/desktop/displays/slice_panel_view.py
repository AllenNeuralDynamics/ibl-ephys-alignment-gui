"""Desktop pyqtgraph view for coronal and perpendicular slice panels."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui

from ephys_alignment_gui.core.alignment_read_models import (
    ActiveSliceRenderState,
    PerpendicularSliceRenderState,
)
from ephys_alignment_gui.core.slice_display_policy import SliceImageKind
from ephys_alignment_gui.desktop.displays.depth_panel_layout import (
    set_depth_panel_bottom_axis,
)
from ephys_alignment_gui.desktop.displays.plot_elements import ColorBar
from ephys_alignment_gui.geometry.ephys_alignment import TIP_SIZE_UM

logger = logging.getLogger(__name__)


def _show_histogram_log_counts(histogram_item: Any) -> None:
    """Display LUT histogram counts on a log1p scale without changing levels."""
    plots = getattr(histogram_item, "plots", None)
    if plots is None:
        plot = getattr(histogram_item, "plot", None)
        plots = () if plot is None else (plot,)
    for plot in plots:
        x_data = getattr(plot, "xData", None)
        y_data = getattr(plot, "yData", None)
        if x_data is None or y_data is None:
            continue
        log_counts = np.log1p(np.maximum(np.asarray(y_data, dtype=float), 0.0))
        plot.setData(x_data, log_counts)


@dataclass(frozen=True)
class SlicePanelPlots:
    """Pyqtgraph plot handles owned by the desktop slice panel."""

    coronal: Any
    coronal_layout: Any
    histogram_alt: Any
    perpendicular: Any
    area: Any | None = None


@dataclass(frozen=True)
class SlicePanelStyle:
    """Pens and style objects used by the desktop slice panel."""

    dotted_pen: Any
    solid_pen: Any
    reference_line_pen: Any


@dataclass
class SlicePanelViewState:
    """Desktop-only pyqtgraph handles and slice-panel UI state."""

    channel_status: bool = True
    channel_projection: Any = None
    slice_lines: list[Any] = field(default_factory=list)
    slice_chns: Any = None
    slice_tip: Any = None
    traj_line: Any = None
    perp_image_item: Any = None
    perp_probe_line: Any = None
    perp_channel_dots: Any = None
    perp_tip_marker: Any = None
    slice_color_bar: Any = None
    slice_hist_levels: Any = None
    active_slice_selection: Any = None
    slice_levels_by_selection: dict[Any, tuple[float, float]] = field(
        default_factory=dict
    )
    slice_item: Any = None
    histogram_item: Any = None

    def reset_coronal_overlays(self) -> None:
        """Forget coronal overlay handles after the plot is cleared."""
        self.channel_projection = None
        self.slice_lines = []
        self.slice_chns = None
        self.slice_tip = None
        self.traj_line = None

    def reset_perpendicular_overlays(self) -> None:
        """Forget perpendicular plot handles after the plot is cleared."""
        self.perp_image_item = None
        self.perp_probe_line = None
        self.perp_channel_dots = None
        self.perp_tip_marker = None


@dataclass
class SlicePanelView:
    """Own desktop pyqtgraph items and rendering for slice panels."""

    plots: SlicePanelPlots
    style: SlicePanelStyle
    histology_exists: Callable[[], bool]
    view_state: SlicePanelViewState = field(default_factory=SlicePanelViewState)
    slice_item: Any = None

    @classmethod
    def create(
        cls,
        *,
        depth_view: Any,
        padding: float,
        set_axis: Callable[..., Any],
        dotted_pen: Any,
        solid_pen: Any,
        reference_line_pen: Any,
        histology_exists: Callable[[], bool],
    ) -> SlicePanelView:
        """Create the desktop slice panel and all of its plot handles."""
        area = pg.GraphicsLayoutWidget()
        coronal_layout = pg.GraphicsLayout()
        histogram_alt = pg.ViewBox()
        coronal = pg.ViewBox()
        coronal_layout.addItem(coronal, 0, 0)
        coronal_layout.addItem(histogram_alt, 0, 1)
        coronal_layout.layout.setColumnStretchFactor(0, 3)
        coronal_layout.layout.setColumnStretchFactor(1, 1)
        area.addItem(coronal_layout)

        perpendicular = pg.PlotItem()
        perpendicular.setContentsMargins(0, 0, 0, 0)
        perpendicular.setMouseEnabled(x=False)
        y_min, y_max = depth_view.plot_y_range_um
        perpendicular.setYRange(min=y_min, max=y_max, padding=padding)
        set_depth_panel_bottom_axis(
            perpendicular,
            set_axis,
            label="Perpendicular distance (µm)",
        )
        set_axis(perpendicular, "left", show=False)
        return cls(
            plots=SlicePanelPlots(
                coronal=coronal,
                coronal_layout=coronal_layout,
                histogram_alt=histogram_alt,
                perpendicular=perpendicular,
                area=area,
            ),
            style=SlicePanelStyle(
                dotted_pen=dotted_pen,
                solid_pen=solid_pen,
                reference_line_pen=reference_line_pen,
            ),
            histology_exists=histology_exists,
            slice_item=histogram_alt,
        )

    def __post_init__(self) -> None:
        if self.view_state.slice_item is None:
            self.view_state.slice_item = (
                self.slice_item
                if self.slice_item is not None
                else self.plots.histogram_alt
            )

    def clear(self) -> None:
        """Clear slice-panel plot items and forget desktop handles."""
        self._remove_histogram_item()
        self.plots.coronal.clear()
        self.clear_perpendicular()
        self.view_state.reset_coronal_overlays()
        self.view_state.slice_color_bar = None
        self.view_state.slice_hist_levels = None
        self.view_state.active_slice_selection = None
        self.view_state.slice_levels_by_selection.clear()
        self.view_state.histogram_item = None

    def clear_perpendicular(self) -> None:
        """Clear perpendicular plot items and forget perpendicular handles."""
        view_state = self.view_state
        self._remove_item(self.plots.perpendicular, view_state.perp_image_item)
        self._remove_item(self.plots.perpendicular, view_state.perp_probe_line)
        self._remove_item(self.plots.perpendicular, view_state.perp_channel_dots)
        self._remove_item(self.plots.perpendicular, view_state.perp_tip_marker)
        self.view_state.reset_perpendicular_overlays()

    def render_slice(self, render_state: ActiveSliceRenderState) -> None:
        """Render a coronal slice payload with desktop plot items."""
        if not self.histology_exists():
            return

        view_state = self.view_state
        decision = render_state.decision
        self.plots.coronal.clear()
        view_state.reset_coronal_overlays()

        img = pg.ImageItem()
        img.setImage(render_state.image)
        img.setTransform(
            QtGui.QTransform(
                render_state.scale[0],
                0.0,
                0.0,
                0.0,
                render_state.scale[1],
                0.0,
                render_state.offset[0],
                render_state.offset[1],
                1.0,
            )
        )

        self._remove_histogram_item()
        if decision.kind is SliceImageKind.LABEL:
            view_state.slice_hist_levels = None
            view_state.active_slice_selection = None
            self.clear_perpendicular()
            self.plots.coronal_layout.addItem(self.plots.histogram_alt, 0, 1)
            view_state.slice_item = self.plots.histogram_alt
        elif decision.kind is SliceImageKind.RGB:
            view_state.slice_hist_levels = None
            view_state.active_slice_selection = None
            self.clear_perpendicular()
        else:
            view_state.active_slice_selection = render_state.selection
            self._render_scalar_slice_controls(img, render_state)

        self.plots.coronal.addItem(img)
        view_state.traj_line = pg.PlotCurveItem()
        view_state.traj_line.setData(
            x=render_state.track_annos_and_ends_ras[:, 0],
            y=render_state.track_annos_and_ends_ras[:, 2],
            pen=self.style.solid_pen,
        )
        self.plots.coronal.addItem(view_state.traj_line)
        self.plot_channels(render_state.projection)

    def render_perpendicular_histology(
        self,
        render_state: PerpendicularSliceRenderState,
    ) -> None:
        """Render a perpendicular slice payload with desktop plot items."""
        view_state = self.view_state
        view_state.reset_perpendicular_overlays()
        view_state.perp_image_item = pg.ImageItem()
        view_state.perp_image_item.setImage(render_state.image)
        view_state.perp_image_item.setTransform(
            QtGui.QTransform(
                render_state.scale_x_um,
                0.0,
                0.0,
                0.0,
                render_state.scale_y_um,
                0.0,
                -render_state.extent_um,
                render_state.feature_min_um,
                1.0,
            )
        )

        if view_state.slice_color_bar is None:
            view_state.slice_color_bar = ColorBar("cividis")
        view_state.perp_image_item.setLookupTable(
            view_state.slice_color_bar.getColourMap()
        )

        if view_state.slice_hist_levels is not None:
            view_state.perp_image_item.setLevels(view_state.slice_hist_levels)

        self.plots.perpendicular.addItem(view_state.perp_image_item)
        self.plots.perpendicular.setXRange(
            min=-render_state.extent_um,
            max=render_state.extent_um,
            padding=0,
        )

        if view_state.channel_status:
            self._render_perpendicular_channel_overlay(render_state)

    def update_perpendicular_levels(self) -> None:
        """Sync perpendicular plot levels with main slice histogram levels."""
        view_state = self.view_state
        if view_state.perp_image_item is None or view_state.histogram_item is None:
            return
        levels = view_state.histogram_item.getLevels()
        view_state.perp_image_item.setLevels(levels)
        view_state.slice_hist_levels = levels
        self._remember_slice_levels(levels)

    def plot_channels(self, projection: Any) -> None:
        """Render or update channel/tip overlays on the coronal slice."""
        if not self.histology_exists():
            return

        view_state = self.view_state
        view_state.channel_status = True
        view_state.channel_projection = projection

        if view_state.slice_chns is None:
            self._create_channel_overlay(projection)
            return
        self._update_channel_overlay(projection)

    def set_channel_projection(self, projection: Any) -> None:
        """Store channel projection data without changing visible overlays."""
        self.view_state.channel_projection = projection

    def toggle_channel_visibility(self) -> None:
        """Toggle channel, tip, trajectory, and perpendicular overlays."""
        if not self.histology_exists():
            return

        view_state = self.view_state
        view_state.channel_status = not view_state.channel_status
        if not view_state.channel_status:
            self._remove_slice_overlays()
            return
        self._add_slice_overlays()

    def render_export_trajectory_overlay(
        self,
        pen: Any,
        *,
        channel_locations_ras: Any | None = None,
    ) -> None:
        """Render the coronal trajectory overlay used by overview exports."""
        if not self.histology_exists():
            return
        if channel_locations_ras is None:
            channel_locations_ras = self.current_channel_locations_ras()
        if channel_locations_ras is None:
            return
        view_state = self.view_state
        if view_state.traj_line is None:
            view_state.traj_line = pg.PlotCurveItem()
        view_state.traj_line.setData(
            x=channel_locations_ras[:, 0],
            y=channel_locations_ras[:, 2],
            pen=pen,
        )
        self.plots.coronal.addItem(view_state.traj_line)

    def current_channel_locations_ras(self) -> Any | None:
        """Return channel locations for the current slice overlay."""
        projection = self.view_state.channel_projection
        if projection is None:
            return None
        return projection.channel_locations_ras

    def set_perpendicular_depth_link(self, linked_plot: Any) -> None:
        """Link the perpendicular slice y-axis to the histology depth plot."""
        self.plots.perpendicular.setYLink(linked_plot)

    def set_perpendicular_depth_range(self, depth_view: Any, padding: float) -> None:
        """Set the starting y-range for the perpendicular slice plot."""
        y_min, y_max = depth_view.plot_y_range_um
        self.plots.perpendicular.setYRange(min=y_min, max=y_max, padding=padding)

    def capture_export_geometry(self) -> tuple[float, float, Any]:
        """Capture slice plot geometry for zoomed plot export."""
        return (
            self.plots.coronal.width(),
            self.plots.coronal.height(),
            self.plots.coronal.viewRect(),
        )

    def _render_scalar_slice_controls(
        self,
        img: Any,
        render_state: ActiveSliceRenderState,
    ) -> None:
        decision = render_state.decision
        view_state = self.view_state
        view_state.slice_color_bar = ColorBar("cividis")
        img.setLookupTable(view_state.slice_color_bar.getColourMap())
        view_state.histogram_item = pg.HistogramLUTItem()
        self._configure_scalar_histogram_axis(view_state.histogram_item)
        view_state.histogram_item.setImageItem(img)
        view_state.histogram_item.gradient.setColorMap(view_state.slice_color_bar.map)
        view_state.histogram_item.autoHistogramRange()
        self.plots.coronal_layout.addItem(view_state.histogram_item, 0, 1)
        remembered_levels = view_state.slice_levels_by_selection.get(
            render_state.selection
        )
        if remembered_levels is not None:
            view_state.histogram_item.setLevels(
                min=remembered_levels[0],
                max=remembered_levels[1],
            )
        elif decision.initial_levels is not None:
            view_state.histogram_item.setLevels(
                min=decision.initial_levels[0],
                max=decision.initial_levels[1],
            )
        else:
            hist_levels = view_state.histogram_item.getLevels()
            hist_val, hist_count = img.getHistogram()
            populated = np.where(hist_count > 10)[0]
            if populated.size and hist_levels[0] != 0:
                upper_val = hist_val[populated[-1]]
                view_state.histogram_item.setLevels(
                    min=hist_levels[0],
                    max=upper_val,
                )
        _show_histogram_log_counts(view_state.histogram_item)

        view_state.slice_hist_levels = view_state.histogram_item.getLevels()
        self._remember_slice_levels(view_state.slice_hist_levels)
        view_state.histogram_item.sigLevelsChanged.connect(
            self.update_perpendicular_levels
        )
        view_state.slice_item = view_state.histogram_item

    @staticmethod
    def _configure_scalar_histogram_axis(histogram_item: Any) -> None:
        axis = getattr(histogram_item, "axis", None)
        if axis is None:
            return
        show = getattr(axis, "show", None)
        if callable(show):
            show()
        set_pen = getattr(axis, "setPen", None)
        if callable(set_pen):
            set_pen("k")
        set_text_pen = getattr(axis, "setTextPen", None)
        if callable(set_text_pen):
            set_text_pen("k")
        set_label = getattr(axis, "setLabel", None)
        if callable(set_label):
            set_label("intensity (a.u.)")

    def _render_perpendicular_channel_overlay(
        self,
        render_state: PerpendicularSliceRenderState,
    ) -> None:
        view_state = self.view_state
        view_state.perp_probe_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=self.style.dotted_pen,
        )
        self.plots.perpendicular.addItem(view_state.perp_probe_line)

        view_state.perp_channel_dots = pg.ScatterPlotItem()
        view_state.perp_channel_dots.setData(
            x=np.zeros(len(render_state.channel_depths_um)),
            y=render_state.channel_depths_um,
            pen="r",
            brush="r",
            size=4,
        )
        self.plots.perpendicular.addItem(view_state.perp_channel_dots)

        view_state.perp_tip_marker = pg.ScatterPlotItem()
        view_state.perp_tip_marker.setData(
            x=[0],
            y=[-TIP_SIZE_UM],
            pen="m",
            brush="m",
            size=5,
        )
        self.plots.perpendicular.addItem(view_state.perp_tip_marker)

    def _create_channel_overlay(self, projection: Any) -> None:
        view_state = self.view_state
        view_state.slice_lines = []
        view_state.slice_chns = pg.ScatterPlotItem()
        view_state.slice_chns.setData(
            x=projection.channel_locations_ras[:, 0],
            y=projection.channel_locations_ras[:, 2],
            pen="r",
            brush="r",
            size=4,
        )
        self.plots.coronal.addItem(view_state.slice_chns)

        view_state.slice_tip = pg.ScatterPlotItem()
        view_state.slice_tip.setData(
            x=[projection.tip_location_ras[0]],
            y=[projection.tip_location_ras[2]],
            pen="m",
            brush="m",
            size=5,
        )
        self.plots.coronal.addItem(view_state.slice_tip)

        self._add_perpendicular_vectors(projection)

    def _update_channel_overlay(self, projection: Any) -> None:
        view_state = self.view_state
        for line in view_state.slice_lines:
            self.plots.coronal.removeItem(line)
        view_state.slice_lines = []
        self._add_perpendicular_vectors(projection)
        view_state.slice_chns.setData(
            x=projection.channel_locations_ras[:, 0],
            y=projection.channel_locations_ras[:, 2],
            pen="r",
            brush="r",
        )
        view_state.slice_tip.setData(
            x=[projection.tip_location_ras[0]],
            y=[projection.tip_location_ras[2]],
            pen="m",
            brush="m",
            size=10,
        )

    def _add_perpendicular_vectors(self, projection: Any) -> None:
        logger.debug("Reference lines: %s", projection.perpendicular_vectors)
        for ref_line in projection.perpendicular_vectors:
            line = pg.PlotCurveItem()
            line.setData(
                x=ref_line[:, 0],
                y=ref_line[:, 2],
                pen=self.style.reference_line_pen,
            )
            self.plots.coronal.addItem(line)
            self.view_state.slice_lines.append(line)

    def _remove_slice_overlays(self) -> None:
        view_state = self.view_state
        self._remove_item(self.plots.coronal, view_state.traj_line)
        self._remove_item(self.plots.coronal, view_state.slice_chns)
        if view_state.slice_tip is not None:
            self._remove_item(self.plots.coronal, view_state.slice_tip)
        for line in view_state.slice_lines:
            self._remove_item(self.plots.coronal, line)

        if view_state.perp_probe_line is not None:
            self._remove_item(self.plots.perpendicular, view_state.perp_probe_line)
        if view_state.perp_channel_dots is not None:
            self._remove_item(self.plots.perpendicular, view_state.perp_channel_dots)
        if view_state.perp_tip_marker is not None:
            self._remove_item(self.plots.perpendicular, view_state.perp_tip_marker)

    def _add_slice_overlays(self) -> None:
        view_state = self.view_state
        self._add_item(self.plots.coronal, view_state.traj_line)
        self._add_item(self.plots.coronal, view_state.slice_chns)
        if view_state.slice_tip is not None:
            self._add_item(self.plots.coronal, view_state.slice_tip)
        for line in view_state.slice_lines:
            self._add_item(self.plots.coronal, line)

        if view_state.perp_probe_line is not None:
            self._add_item(self.plots.perpendicular, view_state.perp_probe_line)
        if view_state.perp_channel_dots is not None:
            self._add_item(self.plots.perpendicular, view_state.perp_channel_dots)
        if view_state.perp_tip_marker is not None:
            self._add_item(self.plots.perpendicular, view_state.perp_tip_marker)

    def _remove_histogram_item(self) -> None:
        view_state = self.view_state
        self._remember_current_histogram_levels()
        if view_state.slice_item is None:
            return
        if self.plots.coronal_layout is None:
            view_state.slice_item = None
            view_state.histogram_item = None
            return
        self.plots.coronal_layout.removeItem(view_state.slice_item)
        view_state.slice_item = None
        view_state.histogram_item = None

    def _remember_current_histogram_levels(self) -> None:
        histogram_item = self.view_state.histogram_item
        if histogram_item is None:
            return
        get_levels = getattr(histogram_item, "getLevels", None)
        if not callable(get_levels):
            return
        self._remember_slice_levels(get_levels())

    def _remember_slice_levels(self, levels: Any) -> None:
        selection = self.view_state.active_slice_selection
        if selection is None or levels is None:
            return
        min_level, max_level = levels
        remembered = (float(min_level), float(max_level))
        self.view_state.slice_hist_levels = remembered
        self.view_state.slice_levels_by_selection[selection] = remembered

    @staticmethod
    def _add_item(plot: Any, item: Any) -> None:
        if item is not None:
            plot.addItem(item)

    @staticmethod
    def _remove_item(plot: Any, item: Any) -> None:
        if item is not None:
            plot.removeItem(item)
