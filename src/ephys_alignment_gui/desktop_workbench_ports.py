"""Build desktop Workbench ports from the MainWindow shell."""

from __future__ import annotations

import gc
from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.desktop_busy_context import BusyContext
from ephys_alignment_gui.desktop_workbench import (
    DesktopAlignmentEditActionPorts,
    DesktopAlignmentRenderPorts,
    DesktopBusyPorts,
    DesktopInteractionPorts,
    DesktopLifecyclePorts,
    DesktopLoadDataPorts,
    DesktopPreviousAlignmentLoadPorts,
    DesktopRenderPorts,
    DesktopSaveWorkflowPorts,
    DesktopShankRenderPorts,
    DesktopWorkbenchPorts,
)


def desktop_workbench_ports_from_main_window(window: Any) -> DesktopWorkbenchPorts:
    """Adapt MainWindow widgets and legacy methods to Workbench ports."""

    def busy_context(*args: Any, **kwargs: Any) -> BusyContext:
        return BusyContext(window, *args, **kwargs)

    def render_alignment_choices(choices: list[str]) -> None:
        window.populate_lists(
            choices,
            window.align_list,
            window.align_combobox,
        )

    return DesktopWorkbenchPorts(
        alignment_edit_actions=DesktopAlignmentEditActionPorts(
            histology_available=lambda: window.histology_exists,
            tip_position_um=window.displays.histology.tip_position_um,
        ),
        busy=DesktopBusyPorts(busy_context=busy_context),
        load_data=DesktopLoadDataPorts(
            clear_empty_state=window._clear_empty_state,
            set_histology_available=window._set_histology_available,
        ),
        lifecycle=DesktopLifecyclePorts(
            close_popups=window.popup_manager.close_all,
            reset_raw_image_payloads=window.shank_screen_view.reset_raw_image_payloads,
            show_empty_state=window._show_empty_state,
            collect_garbage=gc.collect,
        ),
        render=DesktopRenderPorts(
            alignment=DesktopAlignmentRenderPorts(
                restore_lin_fit=window.alignment_screen_view.restore_lin_fit_from_edit,
                capture_depth_plot_y_ranges=(
                    window.alignment_screen_view.capture_depth_plot_y_ranges
                ),
                restore_depth_plot_y_ranges=(
                    window.alignment_screen_view.restore_depth_plot_y_ranges
                ),
                create_reference_lines_for_previous_alignment=(
                    window.alignment_screen_view.create_reference_lines_for_previous_alignment
                ),
                set_default_feature_y_range=(
                    window.alignment_screen_view.set_default_feature_y_range
                ),
                update_status=window.alignment_screen_view.update_status,
            ),
            shank=DesktopShankRenderPorts(
                capture_plot_selection=window.shank_screen_view.capture_plot_selection,
                render_alignment_choices=render_alignment_choices,
                apply_plot_data_state=window.shank_screen_view.apply_plot_data_state,
                raw_image_payloads=window.shank_screen_view.raw_image_payload_mapping,
                render_plot_menus=window.shank_screen_view.render_plot_menus,
                configure_view=(
                    window.shank_screen_view.configure_view_after_render
                ),
                offline=lambda: window.offline,
            ),
        ),
        save_workflow=DesktopSaveWorkflowPorts(
            use_docdb=lambda: window.use_docdb,
            render_alignment_choices=render_alignment_choices,
            busy_context=busy_context,
            complete_button=lambda: window.complete_button,
            histology_available=lambda: window.histology_exists,
            open_qc_dialog=window.qc_dialog.open,
            ephys_qc=window.ephys_qc.currentText,
            selected_qc_descriptions=window._selected_qc_descriptions,
            warning=lambda title, message: QtWidgets.QMessageBox.warning(
                window,
                title,
                message,
            ),
        ),
        previous_alignment_load=DesktopPreviousAlignmentLoadPorts(
            use_docdb=lambda: window.use_docdb,
            set_reload_folder_text=window.reload_folder_line.setText,
            render_alignment_choices=render_alignment_choices,
            busy_context=busy_context,
            reload_button=lambda: window.reload_folder_button,
        ),
        export=window.export_view,
        interaction=DesktopInteractionPorts(
            popup_manager=window.popup_manager,
            region_lookup_service=window.region_lookup_service,
            struct_list=window.struct_list,
            struct_view=window.struct_view,
            struct_description=window.struct_description,
            scale_plot=window.fig_scale,
            histology_plot=window.fig_hist,
            histology_reference_plot=window.fig_hist_ref,
            scale_axis=window.fig_scale_ax,
            bar_colour=window.bar_colour,
            line_pen=window.kpen_solid,
            histology_available=lambda: window.histology_exists,
            activate_window=window.activateWindow,
            set_axis=window.set_axis,
        ),
    )
