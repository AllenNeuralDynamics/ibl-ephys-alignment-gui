"""Build desktop Workbench ports from the MainWindow shell."""

from __future__ import annotations

import gc
from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.desktop.displays.axis_style import set_axis
from ephys_alignment_gui.desktop.shell.busy_context import BusyContext
from ephys_alignment_gui.desktop.workbench.port_types import (
    DesktopAlignmentEditActionPorts,
    DesktopAlignmentRenderPorts,
    DesktopBusyPorts,
    DesktopInteractionPorts,
    DesktopLifecyclePorts,
    DesktopLoadDataPorts,
    DesktopPreviousAlignmentLoadPorts,
    DesktopRenderPorts,
    DesktopSavePorts,
    DesktopShankRenderPorts,
    DesktopWorkbenchPorts,
)


def desktop_workbench_ports_from_main_window(window: Any) -> DesktopWorkbenchPorts:
    """Adapt MainWindow widgets and legacy methods to Workbench ports."""
    style = window.style

    def busy_context(*args: Any, **kwargs: Any) -> BusyContext:
        return BusyContext(window, *args, **kwargs)

    def open_qc_dialog() -> None:
        if qc_dialog := getattr(window, "qc_dialog", None):
            qc_dialog.open()

    def ephys_qc() -> str:
        if qc_widget := getattr(window, "ephys_qc", None):
            return qc_widget.currentText()
        return "Pass"

    def histology_available() -> bool:
        return window.app.queries.workspace.histology_data_loaded()

    def use_docdb() -> bool:
        return window.use_docdb_checkbox.isChecked()

    return DesktopWorkbenchPorts(
        alignment_edit_actions=DesktopAlignmentEditActionPorts(
            histology_available=histology_available,
            tip_position_um=window.displays.histology.tip_position_um,
        ),
        busy=DesktopBusyPorts(busy_context=busy_context),
        load_data=DesktopLoadDataPorts(
            clear_empty_state=window.displays.ephys.clear_empty_state,
        ),
        lifecycle=DesktopLifecyclePorts(
            close_popups=window.popup_manager.close_all,
            reset_raw_image_payloads=window.shank_screen_view.reset_raw_image_payloads,
            show_empty_state=window.displays.ephys.show_empty_state,
            collect_garbage=gc.collect,
        ),
        render=DesktopRenderPorts(
            alignment=DesktopAlignmentRenderPorts(
                capture_depth_plot_y_ranges=(
                    window.alignment_screen_view.capture_depth_plot_y_ranges
                ),
                restore_depth_plot_y_ranges=(
                    window.alignment_screen_view.restore_depth_plot_y_ranges
                ),
            ),
            shank=DesktopShankRenderPorts(
                capture_plot_selection=lambda preserve: (
                    window.shank_screen_view.capture_plot_selection(
                        preserve,
                        displays=window.displays,
                    )
                ),
                render_alignment_choices=(
                    window.alignment_screen_view.render_alignment_choices
                ),
                apply_plot_data_state=window.shank_screen_view.apply_plot_data_state,
                raw_image_payloads=window.shank_screen_view.raw_image_payload_mapping,
                render_plot_menus=lambda state: (
                    window.shank_screen_view.render_plot_menus(
                        state,
                        displays=window.displays,
                    )
                ),
                configure_view=(window.shank_screen_view.configure_view_after_render),
                offline=lambda: window.offline,
            ),
        ),
        save=DesktopSavePorts(
            use_docdb=use_docdb,
            render_alignment_choices=(
                window.alignment_screen_view.render_alignment_choices
            ),
            busy_context=busy_context,
            complete_button=lambda: window.complete_button,
            histology_available=histology_available,
            open_qc_dialog=open_qc_dialog,
            ephys_qc=ephys_qc,
            selected_qc_descriptions=window.shell_actions.selected_qc_descriptions,
            warning=lambda title, message: QtWidgets.QMessageBox.warning(
                window,
                title,
                message,
            ),
        ),
        previous_alignment_load=DesktopPreviousAlignmentLoadPorts(
            use_docdb=use_docdb,
            set_reload_folder_text=window.reload_folder_line.setText,
            render_alignment_choices=(
                window.alignment_screen_view.render_alignment_choices
            ),
            busy_context=busy_context,
            reload_button=lambda: window.reload_folder_button,
        ),
        export=window.export_view,
        interaction=DesktopInteractionPorts(
            popup_manager=window.popup_manager,
            struct_list=lambda: window.struct_list,
            struct_view=lambda: window.struct_view,
            struct_description=lambda: window.struct_description,
            scale_plot=window.displays.histology.scale_plot,
            histology_plot=window.displays.histology.aligned_plot,
            histology_reference_plot=window.displays.histology.reference_plot,
            scale_axis=window.displays.histology.scale_axis,
            bar_colour=style.bar_colour,
            line_pen=style.solid_pen,
            histology_available=histology_available,
            activate_window=window.activateWindow,
            set_axis=set_axis,
        ),
    )
