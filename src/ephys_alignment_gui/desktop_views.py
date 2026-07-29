"""Desktop view aggregate built from MainWindow-owned Qt handles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop_alignment_screen_view import (
    DesktopAlignmentScreenView,
)
from ephys_alignment_gui.desktop_depth_plot_view import DesktopDepthPlotView
from ephys_alignment_gui.desktop_display_ports import (
    desktop_display_ports_from_main_window,
)
from ephys_alignment_gui.desktop_displays import DesktopDisplays
from ephys_alignment_gui.desktop_export_view import DesktopExportView
from ephys_alignment_gui.desktop_path_view import DesktopPathView
from ephys_alignment_gui.desktop_selection_view import DesktopSelectionView
from ephys_alignment_gui.desktop_shank_screen_view import DesktopShankScreenView


@dataclass(frozen=True)
class DesktopViews:
    """Collect concrete desktop views built from Qt/pyqtgraph handles."""

    selection: DesktopSelectionView
    path: DesktopPathView
    displays: DesktopDisplays
    depth: DesktopDepthPlotView
    shank_screen: DesktopShankScreenView
    alignment_screen: DesktopAlignmentScreenView
    export: DesktopExportView

    @classmethod
    def from_main_window(cls, window: Any, *, app: Any) -> DesktopViews:
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
        displays = DesktopDisplays.create(
            app=app,
            ports=desktop_display_ports_from_main_window(window),
        )
        depth = DesktopDepthPlotView(
            depth_view=lambda: window.display_state.depth_view,
            in_brain_depths_um=app.queries.active_in_brain_depths_um,
            default_range_plots=(window.fig_hist, window.fig_hist_ref, window.fig_img),
            range_plots={
                "fig_img": window.fig_img,
                "fig_line": window.fig_line,
                "fig_probe": window.fig_probe,
                "fig_hist": window.fig_hist,
                "fig_hist_ref": window.fig_hist_ref,
                "fig_hist_perp": window.fig_hist_perp,
                "fig_scale": window.fig_scale,
            },
            probe_tip_lines=window.probe_tip_lines,
            probe_top_lines=window.probe_top_lines,
            padding=lambda: window.pad,
        )
        shank_screen = DesktopShankScreenView(
            displays=displays,
            depth_plots=depth,
            init_menubar=window.init_menubar,
            set_view=window.set_view,
        )
        alignment_screen = DesktopAlignmentScreenView(
            depth_plots=depth,
            display_state=window.display_state,
            reference_lines=displays.reference_lines,
            active_alignment_state=lambda: window.document.active_alignment_state,
            lin_fit_checkbox=window.lin_fit_option,
            current_index_label=window.idx_string,
            total_index_label=window.tot_idx_string,
        )
        export = DesktopExportView(
            ephys_graphics_layout=window.fig_data_layout,
            ephys_data_area=window.fig_data_area,
            slice_plot=window.fig_slice,
            slice_trajectory_pen=window.rpen_dot,
            reset_axis=window.reset_axis_button_pressed,
            set_view=window.set_view,
            set_axis=window.set_axis,
            set_font=window.set_font,
            ephys_sizes=lambda: (
                window.fig_probe_width,
                window.fig_ax_width,
            ),
            slice_geometry=lambda: (
                window.slice_width,
                window.slice_height,
                window.slice_rect,
            ),
        )
        return cls(
            selection=selection,
            path=path,
            displays=displays,
            depth=depth,
            shank_screen=shank_screen,
            alignment_screen=alignment_screen,
            export=export,
        )
