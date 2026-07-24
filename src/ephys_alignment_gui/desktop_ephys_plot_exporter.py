"""Desktop export orchestration for ephys plot panels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyqtgraph.exporters as pg_exporters

from ephys_alignment_gui.desktop_ephys_panel_view import DesktopEphysPanelView
from ephys_alignment_gui.desktop_ephys_plot_presenter import (
    DesktopEphysPlotPresenter,
)
from ephys_alignment_gui.plot_registry import PlotMenu


@dataclass(frozen=True)
class EphysExportLayout:
    """Desktop layout handles used while exporting ephys plot images."""

    graphics_layout: Any
    data_area: Any


@dataclass(frozen=True)
class EphysExportSizes:
    """Captured ephys panel dimensions that `set_view()` maintains."""

    probe_width: float
    axis_width: float


@dataclass(frozen=True)
class EphysExportCallbacks:
    """Desktop callbacks needed for ephys plot export layout changes."""

    reset_axis: Callable[[], None]
    set_view: Callable[..., None]
    set_axis: Callable[..., Any]
    set_font: Callable[..., None]
    add_lines_points: Callable[[], None]
    sizes: Callable[[], EphysExportSizes]


@dataclass
class DesktopEphysPlotExporter:
    """Export all available ephys plot menu selections for the active shank."""

    presenter: DesktopEphysPlotPresenter
    panel: DesktopEphysPanelView
    layout: EphysExportLayout
    callbacks: EphysExportCallbacks
    image_exporter_factory: Callable[[Any], Any] = pg_exporters.ImageExporter

    def export(self, output_dir: Path, *, sess_info: str = "") -> None:
        """Export image, probe, and line ephys plots into an output directory."""
        output_dir = Path(output_dir)
        plots = self.panel.plots

        self.callbacks.reset_axis()
        self.callbacks.set_view(view=1, configure=False)

        image_xlabel = plots.image.getAxis("bottom").label.toPlainText()
        line_xlabel = plots.line.getAxis("bottom").label.toPlainText()
        original_width = self.layout.data_area.width()
        original_height = self.layout.data_area.height()
        axis_width = plots.image.getAxis("left").width()
        colorbar_height = plots.image_colorbar.getAxis("top").height()
        sizes = self.callbacks.sizes()

        self._export_image_plots(output_dir, sess_info, axis_width, colorbar_height)
        self._restore_image_plot(axis_width, colorbar_height, image_xlabel)

        self._export_probe_plots(
            output_dir,
            sess_info,
            sizes,
            axis_width,
            colorbar_height,
            original_height,
        )
        probe_colorbar_label = self._prepare_line_export(
            axis_width,
            original_height,
        )
        self._export_action_cycle(output_dir, sess_info, menu="line", prefix="line")
        self._restore_default_layout(
            sizes,
            axis_width,
            colorbar_height,
            original_width,
            original_height,
            line_xlabel,
            probe_colorbar_label,
        )
        self.callbacks.set_view(view=1, configure=False)

    def _export_image_plots(
        self,
        output_dir: Path,
        sess_info: str,
        axis_width: float,
        colorbar_height: float,
    ) -> None:
        plots = self.panel.plots
        self.layout.graphics_layout.removeItem(plots.probe)
        self.layout.graphics_layout.removeItem(plots.probe_colorbar)
        self.layout.graphics_layout.removeItem(plots.line)
        self.callbacks.set_font(
            plots.image,
            "left",
            ptsize=15,
            width=axis_width + 20,
        )
        self.callbacks.set_font(plots.image, "bottom", ptsize=15)
        self.callbacks.set_font(
            plots.image_colorbar,
            "top",
            ptsize=15,
            height=colorbar_height + 15,
        )
        self.layout.data_area.resize(700, self.layout.data_area.height())
        self._export_action_cycle(
            output_dir,
            sess_info,
            menu="image",
            prefix="img",
            before_each=lambda: self.callbacks.set_font(
                plots.image_colorbar,
                "top",
                ptsize=15,
                height=colorbar_height + 15,
            ),
        )

    def _restore_image_plot(
        self,
        axis_width: float,
        colorbar_height: float,
        image_xlabel: str,
    ) -> None:
        plots = self.panel.plots
        self.callbacks.set_font(plots.image, "left", ptsize=8, width=axis_width)
        self.callbacks.set_font(plots.image, "bottom", ptsize=8)
        self.callbacks.set_font(
            plots.image_colorbar,
            "top",
            ptsize=8,
            height=colorbar_height,
        )
        self.callbacks.set_axis(plots.image, "bottom", label=image_xlabel)
        self.layout.graphics_layout.removeItem(plots.image)
        self.layout.graphics_layout.removeItem(plots.image_colorbar)

    def _export_probe_plots(
        self,
        output_dir: Path,
        sess_info: str,
        sizes: EphysExportSizes,
        axis_width: float,
        colorbar_height: float,
        original_height: float,
    ) -> None:
        plots = self.panel.plots
        self.layout.graphics_layout.addItem(plots.probe_colorbar, 0, 0, 1, 2)
        self.layout.graphics_layout.addItem(plots.probe, 1, 0)
        self.callbacks.set_axis(
            plots.probe,
            "left",
            label="Distance from probe tip (uV)",
        )
        plots.probe.setFixedWidth(sizes.probe_width + sizes.axis_width + 20)
        self.callbacks.set_font(
            plots.probe,
            "left",
            ptsize=15,
            width=axis_width + 20,
        )
        self.callbacks.set_font(
            plots.probe_colorbar,
            "top",
            ptsize=15,
            height=colorbar_height + 15,
        )
        self.layout.data_area.resize(250, original_height)
        self._export_action_cycle(
            output_dir,
            sess_info,
            menu="probe",
            prefix="probe",
            before_each=lambda: self.callbacks.set_font(
                plots.probe_colorbar,
                "top",
                ptsize=15,
                height=colorbar_height + 15,
            ),
        )

        plots.probe.setFixedWidth(sizes.probe_width + sizes.axis_width)
        self.callbacks.set_font(plots.probe, "left", ptsize=8, width=axis_width)
        self.callbacks.set_font(
            plots.probe_colorbar,
            "top",
            ptsize=8,
            height=colorbar_height,
        )
        self.callbacks.set_axis(plots.probe, "bottom", pen="w", label="blank")
        self.layout.graphics_layout.removeItem(plots.probe)
        self.layout.graphics_layout.removeItem(plots.probe_colorbar)

    def _prepare_line_export(
        self,
        axis_width: float,
        original_height: float,
    ) -> str:
        plots = self.panel.plots
        self.layout.graphics_layout.addItem(plots.probe_colorbar, 0, 0, 1, 2)
        plots.probe_colorbar.clear()
        probe_colorbar_label = plots.probe_colorbar.getAxis(
            "top"
        ).label.toPlainText()
        self.callbacks.set_axis(plots.probe_colorbar, "top", pen="w")
        self.layout.graphics_layout.addItem(plots.line, 1, 0)

        self.callbacks.set_axis(
            plots.line,
            "left",
            label="Distance from probe tip (um)",
        )
        self.callbacks.set_font(
            plots.line,
            "left",
            ptsize=15,
            width=axis_width + 20,
        )
        self.callbacks.set_font(plots.line, "bottom", ptsize=15)
        self.layout.data_area.resize(200, original_height)
        return probe_colorbar_label

    def _restore_default_layout(
        self,
        sizes: EphysExportSizes,
        axis_width: float,
        colorbar_height: float,
        original_width: float,
        original_height: float,
        line_xlabel: str,
        probe_colorbar_label: str,
    ) -> None:
        plots = self.panel.plots
        for cbar in self.panel.probe_colorbars:
            plots.probe_colorbar.addItem(cbar)

        self.callbacks.set_axis(
            plots.probe_colorbar,
            "top",
            pen="k",
            label=probe_colorbar_label,
        )
        self.callbacks.set_font(plots.line, "left", ptsize=8, width=axis_width)
        self.callbacks.set_font(plots.line, "bottom", ptsize=8)
        self.callbacks.set_axis(plots.line, "bottom", label=line_xlabel)
        plots.probe.setFixedWidth(sizes.probe_width + sizes.axis_width)
        self.layout.graphics_layout.removeItem(plots.line)
        self.layout.graphics_layout.removeItem(plots.probe_colorbar)
        self.layout.data_area.resize(original_width, original_height)
        self.layout.graphics_layout.addItem(plots.probe_colorbar, 0, 0, 1, 2)
        self.layout.graphics_layout.addItem(plots.image_colorbar, 0, 2)
        self.layout.graphics_layout.addItem(plots.probe, 1, 0)
        self.layout.graphics_layout.addItem(plots.line, 1, 1)
        self.layout.graphics_layout.addItem(plots.image, 1, 2)

    def _export_action_cycle(
        self,
        output_dir: Path,
        sess_info: str,
        *,
        menu: PlotMenu,
        prefix: str,
        before_each: Callable[[], None] | None = None,
    ) -> None:
        plot = None
        start_plot = self.presenter.checked_action(menu)
        while start_plot is not None and plot != start_plot:
            checked_action = self.presenter.checked_action(menu)
            if checked_action is None:
                break
            if before_each is not None:
                before_each()
            self._export_scene(
                output_dir / f"{sess_info}{prefix}_{checked_action.text()}.png"
            )
            self.callbacks.add_lines_points()
            self.presenter.toggle_plot(menu)
            plot = self.presenter.checked_action(menu)

    def _export_scene(self, output_path: Path) -> None:
        exporter = self.image_exporter_factory(self.layout.graphics_layout.scene())
        exporter.export(str(output_path))
