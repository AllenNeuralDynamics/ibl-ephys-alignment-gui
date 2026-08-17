"""Adapt the concrete MainWindow shell to focused construction handles."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.desktop.displays.config import DesktopDisplayConfigHandles
from ephys_alignment_gui.desktop.shell.menu_setup import build_menu_bar
from ephys_alignment_gui.desktop.views import DesktopViewHandles
from ephys_alignment_gui.desktop.workbench.ports import DesktopWorkbenchPortHandles


def display_config_handles_from_main_window(window: Any) -> DesktopDisplayConfigHandles:
    """Extract display-config handles from the concrete desktop shell."""
    actions = window.shell_actions
    queries = window.app.queries.workspace
    style = window.style
    return DesktopDisplayConfigHandles(
        depth_view=queries.depth_view_settings,
        linear_fit_enabled=queries.linear_fit_enabled,
        histology_data_loaded=queries.histology_data_loaded,
        reset_axis=actions.reset_axis_button_pressed,
        cluster_clicked=actions.cluster_clicked,
        on_mouse_double_clicked=actions.on_mouse_double_clicked,
        on_mouse_hover=actions.on_mouse_hover,
        on_linear_fit_changed=actions.lin_fit_option_changed,
        solid_pen=style.solid_pen,
        dotted_pen=style.dotted_pen,
        fit_pen=style.fit_pen,
        linear_fit_pen=style.linear_fit_pen,
        reference_line_pen=style.reference_line_pen,
        padding=lambda: style.padding,
    )


def view_handles_from_main_window(window: Any) -> DesktopViewHandles:
    """Extract focused desktop-view handles from the concrete desktop shell."""
    style = window.style
    return DesktopViewHandles(
        session_model=window.session_list,
        session_combobox=window.session_combobox,
        probe_model=window.probe_list,
        probe_combobox=window.probe_combobox,
        shank_model=window.shank_list,
        shank_combobox=window.shank_combobox,
        mouse_root_button=window.mouse_root_button,
        mouse_root_line=window.mouse_root_line,
        output_folder_line=window.output_folder_line,
        current_index_label=window.idx_string,
        total_index_label=window.tot_idx_string,
        alignment_model=window.align_list,
        alignment_combobox=window.align_combobox,
        init_menubar=lambda: build_menu_bar(window),
        reset_axis=window.shell_actions.reset_axis_button_pressed,
        padding=lambda: style.padding,
        slice_trajectory_pen=style.linear_fit_pen,
    )


def workbench_port_handles_from_main_window(
    window: Any,
) -> DesktopWorkbenchPortHandles:
    """Extract Workbench port handles from the concrete desktop shell."""
    style = window.style
    return DesktopWorkbenchPortHandles(
        app=window.app,
        parent=window,
        displays=window.displays,
        views=window.views,
        popup_manager=window.popup_manager,
        shell_actions=window.shell_actions,
        use_docdb_checkbox=window.use_docdb_checkbox,
        complete_button=window.complete_button,
        reload_folder_line=window.reload_folder_line,
        reload_folder_button=window.reload_folder_button,
        export_view=window.export_view,
        offline=lambda: window.offline,
        qc_dialog=lambda: getattr(window, "qc_dialog", None),
        ephys_qc=lambda: getattr(window, "ephys_qc", None),
        struct_list=lambda: getattr(window, "struct_list", None),
        struct_view=lambda: getattr(window, "struct_view", None),
        struct_description=lambda: getattr(window, "struct_description", None),
        activate_window=window.activateWindow,
        bar_colour=style.bar_colour,
        solid_pen=style.solid_pen,
    )
