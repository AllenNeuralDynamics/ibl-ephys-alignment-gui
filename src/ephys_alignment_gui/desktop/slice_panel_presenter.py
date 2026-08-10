"""Desktop presenter/view adapter for coronal and perpendicular slice panels."""

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
from ephys_alignment_gui.core.slice_display_policy import SliceImageKind, SliceSelection
from ephys_alignment_gui.desktop.plot_elements import ColorBar
from ephys_alignment_gui.geometry.ephys_alignment import TIP_SIZE_UM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlicePanelPlots:
    """Pyqtgraph plot handles owned by the desktop slice panel."""

    coronal: Any
    coronal_layout: Any
    histogram_alt: Any
    perpendicular: Any


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
        self.plots.perpendicular.clear()
        self.view_state.reset_coronal_overlays()
        self.view_state.reset_perpendicular_overlays()
        self.view_state.slice_color_bar = None
        self.view_state.slice_hist_levels = None
        self.view_state.histogram_item = None

    def render_slice(
        self,
        render_state: ActiveSliceRenderState,
        *,
        plot_perpendicular_histology: Callable[[str], None] | None = None,
    ) -> None:
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
            view_state.reset_perpendicular_overlays()
            self.plots.perpendicular.clear()
            self.plots.coronal_layout.addItem(self.plots.histogram_alt, 0, 1)
            view_state.slice_item = self.plots.histogram_alt
        elif decision.kind is SliceImageKind.RGB:
            view_state.slice_hist_levels = None
            view_state.reset_perpendicular_overlays()
            self.plots.perpendicular.clear()
        else:
            self._render_scalar_slice_controls(
                img,
                render_state,
                plot_perpendicular_histology=plot_perpendicular_histology,
            )

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

    def _render_scalar_slice_controls(
        self,
        img: Any,
        render_state: ActiveSliceRenderState,
        *,
        plot_perpendicular_histology: Callable[[str], None] | None = None,
    ) -> None:
        decision = render_state.decision
        view_state = self.view_state
        view_state.slice_color_bar = ColorBar("cividis")
        img.setLookupTable(view_state.slice_color_bar.getColourMap())
        view_state.histogram_item = pg.HistogramLUTItem()
        view_state.histogram_item.axis.hide()
        view_state.histogram_item.setImageItem(img)
        view_state.histogram_item.gradient.setColorMap(view_state.slice_color_bar.map)
        view_state.histogram_item.autoHistogramRange()
        self.plots.coronal_layout.addItem(view_state.histogram_item, 0, 1)
        if decision.initial_levels is not None:
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

        view_state.slice_hist_levels = view_state.histogram_item.getLevels()
        if (
            render_state.scalar_channel is not None
            and plot_perpendicular_histology is not None
        ):
            plot_perpendicular_histology(render_state.scalar_channel)
        view_state.histogram_item.sigLevelsChanged.connect(
            self.update_perpendicular_levels
        )
        view_state.slice_item = view_state.histogram_item

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
        if view_state.slice_item is None:
            return
        if self.plots.coronal_layout is None:
            view_state.slice_item = None
            view_state.histogram_item = None
            return
        self.plots.coronal_layout.removeItem(view_state.slice_item)
        view_state.slice_item = None
        view_state.histogram_item = None

    @staticmethod
    def _add_item(plot: Any, item: Any) -> None:
        if item is not None:
            plot.addItem(item)

    @staticmethod
    def _remove_item(plot: Any, item: Any) -> None:
        if item is not None:
            plot.removeItem(item)


@dataclass
class SlicePanelPresenter:
    """Query app slice read models and render them through a slice view."""

    app: Any
    view: SlicePanelView
    action_group_provider: Callable[[], Any | None]

    def current_scalar_slice_channel(self) -> str | None:
        """Return the selected scalar slice channel, if the slice UI has one."""
        render_state = self.current_slice_render_state()
        if render_state is None:
            return None
        return render_state.scalar_channel

    def clear(self) -> None:
        """Clear slice-panel plot items and forget desktop handles."""
        self.view.clear()

    def current_slice_render_state(self) -> ActiveSliceRenderState | None:
        """Return render state for the currently checked slice action."""
        selection = self.current_slice_selection()
        if selection is None:
            return None
        return self.app.queries.slices.active_slice_render_state(selection)

    def current_slice_selection(self) -> SliceSelection | None:
        """Return the slice selection stored on the checked QAction."""
        action_group = self.action_group_provider()
        if action_group is None:
            return None
        action = action_group.checkedAction()
        if action is None:
            return None
        return SliceSelection.from_payload(action.data())

    def action_for_selection(self, selection: SliceSelection) -> Any:
        """Find the QAction that represents a slice selection."""
        action_group = self.action_group_provider()
        if action_group is None:
            return None
        for action in action_group.actions():
            action_selection = SliceSelection.from_payload(action.data())
            if action_selection == selection:
                return action
        return None

    def plot_slice_selection(self, selection: SliceSelection) -> None:
        """Render a coronal slice selection from the application read model."""
        if not self.view.histology_exists():
            return
        render_state = self.app.queries.slices.active_slice_render_state(selection)
        if render_state is None:
            logger.warning("No active slice render state for %s", selection)
            return
        self.render_slice(render_state)

    def render_slice(self, render_state: ActiveSliceRenderState) -> None:
        """Render a coronal slice payload with desktop plot items."""
        self.view.render_slice(
            render_state,
            plot_perpendicular_histology=self.plot_perpendicular_histology,
        )

    def plot_perpendicular_histology(self, channel_name: str = "ccf") -> None:
        """Plot the perpendicular histology slice for the current alignment."""
        if not self.view.histology_exists():
            return

        self.view.plots.perpendicular.clear()
        render_state = self.app.queries.slices.active_perpendicular_slice_state(
            channel_name
        )
        if render_state is None:
            return

        self.render_perpendicular_histology(render_state)

    def render_perpendicular_histology(
        self,
        render_state: PerpendicularSliceRenderState,
    ) -> None:
        """Render a perpendicular slice payload with desktop plot items."""
        self.view.render_perpendicular_histology(render_state)

    def update_perpendicular_levels(self) -> None:
        """Sync perpendicular plot levels with main slice histogram levels."""
        self.view.update_perpendicular_levels()

    def refresh_perpendicular_histology(self) -> None:
        """Refresh perpendicular slice for the selected scalar slice."""
        channel_name = self.current_scalar_slice_channel()
        if channel_name is None:
            return
        self.plot_perpendicular_histology(channel_name)

    def plot_channels(self, projection: Any = None) -> None:
        """Render or update channel/tip overlays on the coronal slice."""
        if projection is None:
            render_state = self.current_slice_render_state()
            if render_state is None:
                return
            projection = render_state.projection
        self.view.plot_channels(projection)

    def toggle_channel_visibility(self) -> None:
        """Toggle channel, tip, trajectory, and perpendicular overlays."""
        self.view.toggle_channel_visibility()

    def render_export_trajectory_overlay(self, pen: Any) -> None:
        """Render the coronal trajectory overlay used by overview exports."""
        channel_locations_ras = self.current_channel_locations_ras()
        self.view.render_export_trajectory_overlay(
            pen,
            channel_locations_ras=channel_locations_ras,
        )

    def current_channel_locations_ras(self) -> Any | None:
        """Return channel locations for the current slice overlay."""
        channel_locations_ras = self.view.current_channel_locations_ras()
        if channel_locations_ras is not None:
            return channel_locations_ras

        render_state = self.current_slice_render_state()
        if render_state is None:
            return None
        self.view.view_state.channel_projection = render_state.projection
        return render_state.projection.channel_locations_ras
