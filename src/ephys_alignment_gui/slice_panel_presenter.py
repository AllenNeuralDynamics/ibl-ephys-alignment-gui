"""Desktop presenter/view adapter for coronal and perpendicular slice panels."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui

from ephys_alignment_gui.alignment_read_models import (
    ActiveSliceRenderState,
    PerpendicularSliceRenderState,
)
from ephys_alignment_gui.ephys_alignment import TIP_SIZE_UM
from ephys_alignment_gui.plot_elements import ColorBar
from ephys_alignment_gui.slice_display_policy import SliceImageKind, SliceSelection

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
class SlicePanelPresenter:
    """Render slice query read models into the desktop pyqtgraph panels."""

    app: Any
    plots: SlicePanelPlots
    style: SlicePanelStyle
    session_provider: Callable[[], Any]
    histology_exists: Callable[[], bool]
    action_group_provider: Callable[[], Any | None]
    slice_item: Any = None
    histogram_item: Any = None

    def __post_init__(self) -> None:
        if self.slice_item is None:
            self.slice_item = self.plots.histogram_alt

    def current_scalar_slice_channel(self) -> str | None:
        """Return the selected scalar slice channel, if the slice UI has one."""
        render_state = self.current_slice_render_state()
        if render_state is None:
            return None
        return render_state.scalar_channel

    def current_slice_render_state(self) -> ActiveSliceRenderState | None:
        """Return render state for the currently checked slice action."""
        selection = self.current_slice_selection()
        if selection is None:
            return None
        return self.app.queries.active_slice_render_state(selection)

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

    def plot_slice(self, data: Any, img_type: str) -> None:
        """Compatibility wrapper for legacy slice-data call sites."""
        selection = self.selection_for_slice_payload(data, img_type)
        if selection is None:
            logger.warning("Cannot resolve legacy slice payload '%s'", img_type)
            return
        self.plot_slice_selection(selection)

    def plot_slice_selection(self, selection: SliceSelection) -> None:
        """Render a coronal slice selection from the application read model."""
        if not self.histology_exists():
            return
        render_state = self.app.queries.active_slice_render_state(selection)
        if render_state is None:
            logger.warning("No active slice render state for %s", selection)
            return
        self.render_slice(render_state)

    def render_slice(self, render_state: ActiveSliceRenderState) -> None:
        """Render a coronal slice payload with desktop plot items."""
        if not self.histology_exists():
            return

        session = self._session()
        decision = render_state.decision
        self.plots.coronal.clear()
        session.slice_chns = []
        session.slice_lines = []

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
            session.slice_hist_levels = None
            session.perp_image_item = None
            self.plots.perpendicular.clear()
            self.plots.coronal_layout.addItem(self.plots.histogram_alt, 0, 1)
            self.slice_item = self.plots.histogram_alt
        elif decision.kind is SliceImageKind.RGB:
            session.slice_hist_levels = None
            session.perp_image_item = None
            self.plots.perpendicular.clear()
        else:
            self._render_scalar_slice_controls(session, img, render_state)

        self.plots.coronal.addItem(img)
        session.traj_line = pg.PlotCurveItem()
        session.traj_line.setData(
            x=render_state.track_annos_and_ends_ras[:, 0],
            y=render_state.track_annos_and_ends_ras[:, 2],
            pen=self.style.solid_pen,
        )
        self.plots.coronal.addItem(session.traj_line)
        self.plot_channels(render_state.projection)

    def plot_perpendicular_histology(self, channel_name: str = "ccf") -> None:
        """Plot the perpendicular histology slice for the current alignment."""
        if not self.histology_exists():
            return

        self.plots.perpendicular.clear()
        render_state = self.app.queries.active_perpendicular_slice_state(channel_name)
        if render_state is None:
            return

        self.render_perpendicular_histology(render_state)

    def render_perpendicular_histology(
        self,
        render_state: PerpendicularSliceRenderState,
    ) -> None:
        """Render a perpendicular slice payload with desktop plot items."""
        session = self._session()
        session.perp_image_item = pg.ImageItem()
        session.perp_image_item.setImage(render_state.image)
        session.perp_image_item.setTransform(
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

        if session.slice_color_bar is None:
            session.slice_color_bar = ColorBar("cividis")
        session.perp_image_item.setLookupTable(session.slice_color_bar.getColourMap())

        if session.slice_hist_levels is not None:
            session.perp_image_item.setLevels(session.slice_hist_levels)

        self.plots.perpendicular.addItem(session.perp_image_item)
        self.plots.perpendicular.setXRange(
            min=-render_state.extent_um,
            max=render_state.extent_um,
            padding=0,
        )

        if session.channel_status:
            self._render_perpendicular_channel_overlay(session, render_state)

    def update_perpendicular_levels(self) -> None:
        """Sync perpendicular plot levels with main slice histogram levels."""
        session = self._session()
        if session.perp_image_item is None or self.histogram_item is None:
            return
        levels = self.histogram_item.getLevels()
        session.perp_image_item.setLevels(levels)
        session.slice_hist_levels = levels

    def refresh_perpendicular_histology(self) -> None:
        """Refresh perpendicular slice for the selected scalar slice."""
        channel_name = self.current_scalar_slice_channel()
        if channel_name is None:
            return
        self.plot_perpendicular_histology(channel_name)

    def plot_channels(self, projection: Any = None) -> None:
        """Render or update channel/tip overlays on the coronal slice."""
        if not self.histology_exists():
            return

        session = self._session()
        session.channel_status = True
        if projection is None:
            render_state = self.current_slice_render_state()
            if render_state is None:
                return
            projection = render_state.projection

        session.channel_locations_ras = projection.channel_locations_ras
        session.tip_location_ras = projection.tip_location_ras

        if not session.slice_chns:
            self._create_channel_overlay(session, projection)
            return
        self._update_channel_overlay(session, projection)

    def toggle_channel_visibility(self) -> None:
        """Toggle channel, tip, trajectory, and perpendicular overlays."""
        if not self.histology_exists():
            return

        session = self._session()
        session.channel_status = not session.channel_status
        if not session.channel_status:
            self._remove_slice_overlays(session)
            return
        self._add_slice_overlays(session)

    def selection_for_slice_payload(
        self,
        data: Any,
        img_type: str,
    ) -> SliceSelection | None:
        """Map a legacy slice mapping object back to a SliceSelection."""
        state = self.app.queries.active_slice_data_state()
        if state is None:
            return None
        if data is state.slice_data:
            return SliceSelection("slice_data", img_type)
        if data is state.fp_slice_data:
            return SliceSelection("fp_slice_data", img_type)
        return None

    def _render_scalar_slice_controls(
        self,
        session: Any,
        img: Any,
        render_state: ActiveSliceRenderState,
    ) -> None:
        decision = render_state.decision
        session.slice_color_bar = ColorBar("cividis")
        img.setLookupTable(session.slice_color_bar.getColourMap())
        self.histogram_item = pg.HistogramLUTItem()
        self.histogram_item.axis.hide()
        self.histogram_item.setImageItem(img)
        self.histogram_item.gradient.setColorMap(session.slice_color_bar.map)
        self.histogram_item.autoHistogramRange()
        self.plots.coronal_layout.addItem(self.histogram_item, 0, 1)
        if decision.initial_levels is not None:
            self.histogram_item.setLevels(
                min=decision.initial_levels[0],
                max=decision.initial_levels[1],
            )
        else:
            hist_levels = self.histogram_item.getLevels()
            hist_val, hist_count = img.getHistogram()
            populated = np.where(hist_count > 10)[0]
            if populated.size and hist_levels[0] != 0:
                upper_val = hist_val[populated[-1]]
                self.histogram_item.setLevels(min=hist_levels[0], max=upper_val)

        session.slice_hist_levels = self.histogram_item.getLevels()
        if render_state.scalar_channel is not None:
            self.plot_perpendicular_histology(render_state.scalar_channel)
        self.histogram_item.sigLevelsChanged.connect(self.update_perpendicular_levels)
        self.slice_item = self.histogram_item

    def _render_perpendicular_channel_overlay(
        self,
        session: Any,
        render_state: PerpendicularSliceRenderState,
    ) -> None:
        session.perp_probe_line = pg.InfiniteLine(
            pos=0, angle=90, pen=self.style.dotted_pen
        )
        self.plots.perpendicular.addItem(session.perp_probe_line)

        session.perp_channel_dots = pg.ScatterPlotItem()
        session.perp_channel_dots.setData(
            x=np.zeros(len(render_state.channel_depths_um)),
            y=render_state.channel_depths_um,
            pen="r",
            brush="r",
            size=4,
        )
        self.plots.perpendicular.addItem(session.perp_channel_dots)

        session.perp_tip_marker = pg.ScatterPlotItem()
        session.perp_tip_marker.setData(
            x=[0],
            y=[-TIP_SIZE_UM],
            pen="m",
            brush="m",
            size=5,
        )
        self.plots.perpendicular.addItem(session.perp_tip_marker)

    def _create_channel_overlay(self, session: Any, projection: Any) -> None:
        session.slice_lines = []
        session.slice_chns = pg.ScatterPlotItem()
        session.slice_chns.setData(
            x=session.channel_locations_ras[:, 0],
            y=session.channel_locations_ras[:, 2],
            pen="r",
            brush="r",
            size=4,
        )
        self.plots.coronal.addItem(session.slice_chns)

        session.slice_tip = pg.ScatterPlotItem()
        session.slice_tip.setData(
            x=[session.tip_location_ras[0]],
            y=[session.tip_location_ras[2]],
            pen="m",
            brush="m",
            size=5,
        )
        self.plots.coronal.addItem(session.slice_tip)

        self._add_perpendicular_vectors(session, projection)

    def _update_channel_overlay(self, session: Any, projection: Any) -> None:
        for line in session.slice_lines:
            self.plots.coronal.removeItem(line)
        session.slice_lines = []
        self._add_perpendicular_vectors(session, projection)
        session.slice_chns.setData(
            x=session.channel_locations_ras[:, 0],
            y=session.channel_locations_ras[:, 2],
            pen="r",
            brush="r",
        )
        session.slice_tip.setData(
            x=[session.tip_location_ras[0]],
            y=[session.tip_location_ras[2]],
            pen="m",
            brush="m",
            size=10,
        )

    def _add_perpendicular_vectors(self, session: Any, projection: Any) -> None:
        logger.debug("Reference lines: %s", projection.perpendicular_vectors)
        for ref_line in projection.perpendicular_vectors:
            line = pg.PlotCurveItem()
            line.setData(
                x=ref_line[:, 0],
                y=ref_line[:, 2],
                pen=self.style.reference_line_pen,
            )
            self.plots.coronal.addItem(line)
            session.slice_lines.append(line)

    def _remove_slice_overlays(self, session: Any) -> None:
        self._remove_item(self.plots.coronal, session.traj_line)
        self._remove_item(self.plots.coronal, session.slice_chns)
        if session.slice_tip is not None:
            self._remove_item(self.plots.coronal, session.slice_tip)
        for line in session.slice_lines:
            self._remove_item(self.plots.coronal, line)

        if session.perp_probe_line is not None:
            self._remove_item(self.plots.perpendicular, session.perp_probe_line)
        if session.perp_channel_dots is not None:
            self._remove_item(self.plots.perpendicular, session.perp_channel_dots)
        if session.perp_tip_marker is not None:
            self._remove_item(self.plots.perpendicular, session.perp_tip_marker)

    def _add_slice_overlays(self, session: Any) -> None:
        self._add_item(self.plots.coronal, session.traj_line)
        self._add_item(self.plots.coronal, session.slice_chns)
        if session.slice_tip is not None:
            self._add_item(self.plots.coronal, session.slice_tip)
        for line in session.slice_lines:
            self._add_item(self.plots.coronal, line)

        if session.perp_probe_line is not None:
            self._add_item(self.plots.perpendicular, session.perp_probe_line)
        if session.perp_channel_dots is not None:
            self._add_item(self.plots.perpendicular, session.perp_channel_dots)
        if session.perp_tip_marker is not None:
            self._add_item(self.plots.perpendicular, session.perp_tip_marker)

    def _remove_histogram_item(self) -> None:
        if self.slice_item is None:
            return
        self.plots.coronal_layout.removeItem(self.slice_item)
        self.slice_item = None
        self.histogram_item = None

    @staticmethod
    def _add_item(plot: Any, item: Any) -> None:
        if item is not None:
            plot.addItem(item)

    @staticmethod
    def _remove_item(plot: Any, item: Any) -> None:
        if item is not None:
            plot.removeItem(item)

    def _session(self) -> Any:
        session = self.session_provider()
        if session is None:
            raise RuntimeError("Slice panel session is not available")
        return session
