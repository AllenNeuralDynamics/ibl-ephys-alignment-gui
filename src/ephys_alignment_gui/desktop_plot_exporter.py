"""Desktop plot export orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph.exporters as pg_exporters

from ephys_alignment_gui.create_overview_plots import make_overview_plot
from ephys_alignment_gui.desktop_ephys_plot_exporter import (
    DesktopEphysPlotExporter,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SliceExportGeometry:
    """Captured slice plot geometry used while exporting zoomed slice images."""

    width: float
    height: float
    rect: Any


@dataclass(frozen=True)
class SliceExportHandles:
    """Desktop slice plot handles needed by plot export."""

    slice_display: Any
    slice_plot: Any


@dataclass(frozen=True)
class SliceExportStyle:
    """Desktop styling handles for slice export overlays."""

    trajectory_pen: Any


@dataclass(frozen=True)
class HistologyExportHandles:
    """Desktop histology plot handles needed by plot export."""

    histology_display: Any


@dataclass(frozen=True)
class DesktopPlotExportCallbacks:
    """Desktop callbacks needed by non-ephys plot export steps."""

    set_axis: Callable[..., Any]
    set_font: Callable[..., None]
    add_lines_points: Callable[[], None]
    slice_geometry: Callable[[], SliceExportGeometry]
    make_overview: Callable[..., Any] = make_overview_plot


@dataclass
class DesktopPlotExporter:
    """Export all desktop plot panels for the active shank."""

    ephys_exporter: DesktopEphysPlotExporter
    slice_handles: SliceExportHandles
    slice_style: SliceExportStyle
    histology_handles: HistologyExportHandles
    callbacks: DesktopPlotExportCallbacks
    image_exporter_factory: Callable[[Any], Any] = pg_exporters.ImageExporter

    def export(self, output_dir: Path, *, sess_info: str = "") -> None:
        """Export ephys, slice, histology, and overview plots."""
        output_dir = Path(output_dir)
        self.ephys_exporter.export(output_dir, sess_info=sess_info)
        self._export_slice_images(output_dir, sess_info)
        self._export_zoomed_slice_images(output_dir, sess_info)
        self._export_histology_image(output_dir, sess_info)
        self.callbacks.make_overview(output_dir, sess_info, save_folder=output_dir)
        self.callbacks.add_lines_points()

    def _export_slice_images(self, output_dir: Path, sess_info: str) -> None:
        action_group = self._slice_action_group()
        if action_group is None:
            return
        plot = None
        start_plot = action_group.checkedAction()
        while start_plot is not None and plot != start_plot:
            self._prepare_slice_export_overlay()
            slice_action = action_group.checkedAction()
            if slice_action is None:
                break
            self._export_item(
                self.slice_handles.slice_plot,
                output_dir / f"{sess_info}slice_{slice_action.text()}.png",
            )
            self._toggle_action_group(action_group)
            plot = action_group.checkedAction()

    def _export_zoomed_slice_images(self, output_dir: Path, sess_info: str) -> None:
        action_group = self._slice_action_group()
        if action_group is None:
            return
        geometry = self.callbacks.slice_geometry()
        plot = None
        start_plot = action_group.checkedAction()
        while start_plot is not None and plot != start_plot:
            self._prepare_slice_export_overlay()
            slice_action = action_group.checkedAction()
            if slice_action is None:
                break
            channel_locations_ras = (
                self.slice_handles.slice_display.current_channel_locations_ras()
            )
            if channel_locations_ras is None:
                self._toggle_action_group(action_group)
                plot = action_group.checkedAction()
                continue
            self._set_zoomed_slice_range(channel_locations_ras)
            self.slice_handles.slice_plot.resize(50, geometry.height)
            self._export_item(
                self.slice_handles.slice_plot,
                output_dir / f"{sess_info}slice_zoom_{slice_action.text()}.png",
            )
            self.slice_handles.slice_plot.resize(geometry.width, geometry.height)
            self.slice_handles.slice_plot.setRange(rect=geometry.rect)
            self._toggle_action_group(action_group)
            plot = action_group.checkedAction()

    def _prepare_slice_export_overlay(self) -> None:
        self.slice_handles.slice_display.toggle_channel_visibility()
        self.slice_handles.slice_display.render_export_trajectory_overlay(
            self.slice_style.trajectory_pen
        )
        self.slice_handles.slice_display.plot_channels()

    def _slice_action_group(self) -> Any | None:
        action_group = self.slice_handles.slice_display.action_group
        if action_group is None:
            logger.warning("No available slice plot actions to export")
            return None
        return action_group

    def _set_zoomed_slice_range(self, channel_locations_ras: Any) -> None:
        self.slice_handles.slice_plot.setXRange(
            min=np.min(channel_locations_ras[:, 0]) - 200 / 1e6,
            max=np.max(channel_locations_ras[:, 0]) + 200 / 1e6,
        )
        self.slice_handles.slice_plot.setYRange(
            min=np.min(channel_locations_ras[:, 2]) - 500 / 1e6,
            max=np.max(channel_locations_ras[:, 2]) + 500 / 1e6,
        )

    def _export_histology_image(self, output_dir: Path, sess_info: str) -> None:
        histology = self.histology_handles.histology_display
        self.callbacks.set_axis(histology.extra_y_axis, "left")
        self.callbacks.set_axis(histology.aligned_plot, "bottom", label="aligned")
        self.callbacks.set_font(histology.aligned_plot, "bottom", ptsize=12)
        self.callbacks.set_axis(histology.reference_plot, "bottom", label="original")
        self.callbacks.set_font(histology.reference_plot, "bottom", ptsize=12)
        self._export_item(
            histology.export_scene(),
            output_dir / f"{sess_info}hist.png",
        )
        self.callbacks.set_axis(histology.extra_y_axis, "left", pen=None)
        self.callbacks.set_font(histology.aligned_plot, "bottom", ptsize=8)
        self.callbacks.set_axis(
            histology.aligned_plot,
            "bottom",
            pen="w",
            label="blank",
        )
        self.callbacks.set_font(histology.reference_plot, "bottom", ptsize=8)
        self.callbacks.set_axis(
            histology.reference_plot,
            "bottom",
            pen="w",
            label="blank",
        )

    def _export_item(self, item: Any, output_path: Path) -> None:
        exporter = self.image_exporter_factory(item)
        exporter.export(str(output_path))

    @staticmethod
    def _toggle_action_group(action_group: Any) -> None:
        current_action = action_group.checkedAction()
        actions = action_group.actions()
        if not actions:
            logger.warning("No available plot actions to toggle")
            return
        if current_action is None:
            actions[0].setChecked(True)
            actions[0].trigger()
            return
        try:
            current_index = actions.index(current_action)
        except ValueError:
            actions[0].setChecked(True)
            actions[0].trigger()
            return
        next_index = (current_index + 1) % len(actions)
        actions[next_index].setChecked(True)
        actions[next_index].trigger()
