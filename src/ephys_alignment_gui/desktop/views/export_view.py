"""Desktop view operations and handles for plot export."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.displays.ephys_plot_exporter import (
    EphysExportCallbacks,
    EphysExportLayout,
    EphysExportSizes,
)
from ephys_alignment_gui.desktop.displays.plot_exporter import (
    DesktopPlotExportCallbacks,
    SliceExportGeometry,
    SliceExportHandles,
    SliceExportStyle,
)


@dataclass(frozen=True)
class DesktopExportView:
    """Own desktop-only export layout handles and helper callbacks."""

    ephys_graphics_layout: Any
    ephys_data_area: Any
    slice_plot: Any
    slice_trajectory_pen: Any
    reset_axis: Callable[[], None]
    set_view: Callable[..., None]
    set_axis: Callable[..., Any]
    set_font: Callable[..., None]
    ephys_sizes: Callable[[], tuple[float, float]]
    slice_geometry: Callable[[], tuple[float, float, Any]]

    def ephys_layout(self) -> EphysExportLayout:
        """Return ephys layout handles for export-time layout mutations."""
        return EphysExportLayout(
            graphics_layout=self.ephys_graphics_layout,
            data_area=self.ephys_data_area,
        )

    def ephys_callbacks(
        self,
        *,
        add_lines_points: Callable[[], None],
    ) -> EphysExportCallbacks:
        """Return callbacks needed by ephys plot export."""
        return EphysExportCallbacks(
            reset_axis=self.reset_axis,
            set_view=self.set_view,
            set_axis=self.set_axis,
            set_font=self.set_font,
            add_lines_points=add_lines_points,
            sizes=self.ephys_export_sizes,
        )

    def ephys_export_sizes(self) -> EphysExportSizes:
        """Return the current ephys panel sizes captured by ``set_view``."""
        probe_width, axis_width = self.ephys_sizes()
        return EphysExportSizes(
            probe_width=probe_width,
            axis_width=axis_width,
        )

    def slice_handles(self, slice_display: Any) -> SliceExportHandles:
        """Return slice display handles needed by plot export."""
        return SliceExportHandles(
            slice_display=slice_display,
            slice_plot=self.slice_plot,
        )

    def slice_style(self) -> SliceExportStyle:
        """Return slice export overlay styling handles."""
        return SliceExportStyle(
            trajectory_pen=self.slice_trajectory_pen,
        )

    def plot_callbacks(self) -> DesktopPlotExportCallbacks:
        """Return callbacks needed by non-ephys plot export steps."""
        return DesktopPlotExportCallbacks(
            set_axis=self.set_axis,
            set_font=self.set_font,
            slice_geometry=self.slice_export_geometry,
        )

    def slice_export_geometry(self) -> SliceExportGeometry:
        """Return the current slice plot geometry for zoomed exports."""
        width, height, rect = self.slice_geometry()
        return SliceExportGeometry(
            width=width,
            height=height,
            rect=rect,
        )
