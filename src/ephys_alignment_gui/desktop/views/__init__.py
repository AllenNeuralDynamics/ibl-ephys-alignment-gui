"""Desktop view aggregate built from desktop-owned Qt handles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.displays.axis_style import set_axis, set_font
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
        style = window.style
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
            padding=lambda: style.padding,
        )
        shank_screen = DesktopShankScreenView(
            depth_plots=depth,
            init_menubar=lambda: build_menu_bar(window),
            apply_ephys_view=displays.ephys.apply_view,
            capture_slice_export_geometry=displays.slice.capture_export_geometry,
        )
        alignment_screen = DesktopAlignmentScreenView(
            depth_plots=depth,
            reference_lines=displays.reference_lines,
            lin_fit_checkbox=displays.histology.linear_fit_checkbox,
            current_index_label=window.idx_string,
            total_index_label=window.tot_idx_string,
            alignment_model=window.align_list,
            alignment_combobox=window.align_combobox,
        )
        export = DesktopExportView(
            ephys_graphics_layout=displays.ephys.graphics_layout,
            ephys_data_area=displays.ephys.area,
            slice_plot=displays.slice.coronal_plot,
            slice_trajectory_pen=style.linear_fit_pen,
            reset_axis=window.shell_actions.reset_axis_button_pressed,
            set_view=shank_screen.set_view,
            set_axis=set_axis,
            set_font=set_font,
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
