"""Desktop view aggregate built from desktop-owned Qt handles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.shell.menu_setup import build_menu_bar
from ephys_alignment_gui.desktop.views.alignment_screen_view import (
    DesktopAlignmentScreenView,
)
from ephys_alignment_gui.desktop.views.depth_plot_view import DesktopDepthPlotView
from ephys_alignment_gui.desktop.views.export_view import DesktopExportView
from ephys_alignment_gui.desktop.views.path_view import DesktopPathView
from ephys_alignment_gui.desktop.views.selection_view import DesktopSelectionView
from ephys_alignment_gui.desktop.views.shank_screen_view import DesktopShankScreenView


@dataclass(frozen=True)
class DesktopViews:
    """Collect concrete desktop views built from Qt/pyqtgraph handles."""

    selection: DesktopSelectionView
    path: DesktopPathView
    depth: DesktopDepthPlotView
    shank_screen: DesktopShankScreenView
    alignment_screen: DesktopAlignmentScreenView
    export: DesktopExportView

    @classmethod
    def from_main_window(cls, window: Any, *, displays: Any) -> DesktopViews:
        """Build focused desktop view objects from a MainWindow shell."""
        selection = DesktopSelectionView(
            session_model=window.session_list,
            session_combobox=window.session_combobox,
            probe_model=window.probe_list,
            probe_combobox=window.probe_combobox,
            shank_model=window.shank_list,
            shank_combobox=window.shank_combobox,
            load_data_button=window.load_data_button,
        )
        path = DesktopPathView(
            mouse_root_button=window.mouse_root_button,
            mouse_root_line=window.mouse_root_line,
            output_folder_line=window.output_folder_line,
        )
        depth = DesktopDepthPlotView(
            default_range_plots=(
                displays.histology.aligned_plot,
                displays.histology.reference_plot,
                displays.ephys.panel.plots.image,
            ),
            range_plots={
                "fig_img": displays.ephys.panel.plots.image,
                "fig_line": displays.ephys.panel.plots.line,
                "fig_probe": displays.ephys.panel.plots.probe,
                "fig_hist": displays.histology.aligned_plot,
                "fig_hist_ref": displays.histology.reference_plot,
                "fig_hist_perp": displays.slice.perpendicular_plot,
                "fig_scale": displays.histology.scale_plot,
            },
            probe_tip_lines=displays.ephys.panel.probe_tip_lines,
            probe_top_lines=displays.ephys.panel.probe_top_lines,
            padding=lambda: window.pad,
        )
        shank_screen = DesktopShankScreenView(
            depth_plots=depth,
            init_menubar=lambda: build_menu_bar(window),
            set_view=window.set_view,
        )
        alignment_screen = DesktopAlignmentScreenView(
            depth_plots=depth,
            reference_lines=displays.reference_lines,
            lin_fit_checkbox=displays.histology.linear_fit_checkbox,
            current_index_label=window.idx_string,
            total_index_label=window.tot_idx_string,
        )
        export = DesktopExportView(
            ephys_graphics_layout=displays.ephys.graphics_layout,
            ephys_data_area=displays.ephys.area,
            slice_plot=displays.slice.coronal_plot,
            slice_trajectory_pen=window.rpen_dot,
            reset_axis=window.reset_axis_button_pressed,
            set_view=window.set_view,
            set_axis=window.set_axis,
            set_font=window.set_font,
            ephys_sizes=displays.ephys.export_sizes,
            slice_geometry=displays.slice.capture_export_geometry,
        )
        return cls(
            selection=selection,
            path=path,
            depth=depth,
            shank_screen=shank_screen,
            alignment_screen=alignment_screen,
            export=export,
        )
