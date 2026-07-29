"""Build desktop Workbench ports from the MainWindow shell."""

from __future__ import annotations

from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.desktop_busy_context import BusyContext
from ephys_alignment_gui.desktop_workbench import (
    DesktopAlignmentRenderPorts,
    DesktopHistologyRenderPorts,
    DesktopPreviousAlignmentLoadPorts,
    DesktopRenderPorts,
    DesktopSaveWorkflowPorts,
    DesktopSelectionWorkflowCallbacks,
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
        selection=DesktopSelectionWorkflowCallbacks(
            capture_pending_reference_lines=window._capture_pending_reference_lines,
            stash_and_detach_current=window._stash_and_detach_current,
            teardown_session=window._teardown_session,
            init_session_variables=window.init_session_variables,
            select_shank_for_view=lambda shank_idx, source: (
                window._select_shank_for_view(shank_idx, source=source)
            ),
            setup_session_view=lambda preserve, shank_idx: window.setup_session_view(
                preserve_plot_selection=preserve,
                shank_idx=shank_idx,
            ),
            clear_empty_state=window._clear_empty_state,
            set_histology_available=window._set_histology_available,
            mouse_root_loaded=lambda: window.data_context.mouse_root is not None,
            active_shank_idx=window._active_shank_idx,
            show_empty_state=window._show_empty_state,
            evict_stream_cache=window._evict_stream_cache,
            clear_histology_context=window.histology_context.clear,
            select_first_session=lambda: window.on_session_combobox_activated(0),
            select_first_probe=lambda: window.on_probe_combobox_activated(0),
            busy_context=busy_context,
        ),
        render=DesktopRenderPorts(
            alignment=DesktopAlignmentRenderPorts(
                restore_lin_fit=window._restore_lin_fit_from_edit,
                clear_reference_lines=window.reference_lines.clear,
                capture_depth_plot_y_ranges=window._capture_depth_plot_y_ranges,
                restore_depth_plot_y_ranges=window._restore_depth_plot_y_ranges,
                reattach_reference_lines=window._reattach_reference_lines,
                plot_channels=window.slice_panel.plot_channels,
                refresh_perpendicular_histology=(
                    window.slice_panel.refresh_perpendicular_histology
                ),
                update_reference_lines_to_alignment=window.update_lines_points,
                create_reference_lines_for_previous_alignment=(
                    window._create_reference_lines_for_previous_alignment
                ),
                set_default_feature_y_range=window.set_default_feature_y_range,
                update_status=window.update_string,
            ),
            histology=DesktopHistologyRenderPorts(
                probe_extent_query_kwargs=window._probe_extent_query_kwargs,
                fit_depth_um=lambda: window.display_state.depth_view.fit_depth_um,
                lin_fit_enabled=lambda: window.display_state.edit_settings.lin_fit,
                scale_factor_y_range=window._scale_factor_y_range,
            ),
            shank=DesktopShankRenderPorts(
                capture_plot_selection=window._capture_shank_plot_selection,
                clear_reference_lines=window.reference_lines.clear,
                prepare_runtime=window._prepare_shank_runtime_for_view,
                prepare_histology=window._prepare_shank_histology_for_view,
                apply_plot_data_state=window._apply_shank_plot_data_state,
                raw_image_payloads=lambda: window.raw_image_payloads,
                render_plot_menus=window._render_shank_plot_menus,
                render_ephys_plots=window.ephys_plot_presenter.render_shank_ephys_plots,
                render_histology_plots=lambda shank_idx: window.render_histology_plots(
                    shank_idx=shank_idx,
                ),
                restore_slice_selection=window._restore_shank_slice_selection,
                configure_view=window._configure_shank_view_after_render,
                histology_available=lambda: window.histology_exists,
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
            select_alignment=window.on_alignment_selected,
            busy_context=busy_context,
            reload_button=lambda: window.reload_folder_button,
        ),
    )
