"""Build desktop Workbench ports from the MainWindow shell."""

from __future__ import annotations

from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.desktop_busy_context import BusyContext
from ephys_alignment_gui.desktop_displays import DesktopDisplayPorts
from ephys_alignment_gui.desktop_ephys_display import DesktopEphysDisplayPorts
from ephys_alignment_gui.desktop_histology_display import DesktopHistologyDisplayPorts
from ephys_alignment_gui.desktop_reference_line_display import (
    DesktopReferenceLineDisplayPorts,
)
from ephys_alignment_gui.desktop_slice_display import DesktopSliceDisplayPorts
from ephys_alignment_gui.desktop_workbench import (
    DesktopAlignmentRenderPorts,
    DesktopExportPorts,
    DesktopInteractionPorts,
    DesktopPreviousAlignmentLoadPorts,
    DesktopRenderPorts,
    DesktopSaveWorkflowPorts,
    DesktopSelectionWorkflowCallbacks,
    DesktopShankRenderPorts,
    DesktopWorkbenchPorts,
)


def desktop_ephys_display_ports_from_main_window(
    window: Any,
) -> DesktopEphysDisplayPorts:
    """Adapt MainWindow plot handles to ephys display ports."""
    return DesktopEphysDisplayPorts(
        image_plot=window.fig_img,
        image_colorbar=window.fig_img_cb,
        line_plot=window.fig_line,
        probe_plot=window.fig_probe,
        probe_colorbar=window.fig_probe_cb,
        graphics_layout=window.fig_data_layout,
        line_pen=window.kpen_solid,
        raw_image_payloads=lambda: window.raw_image_payloads,
        set_axis=window.set_axis,
        reset_axis=window.reset_axis_button_pressed,
        cluster_clicked=lambda *args: window.desktop_workbench.cluster_clicked(*args),
    )


def desktop_histology_display_ports_from_main_window(
    window: Any,
) -> DesktopHistologyDisplayPorts:
    """Adapt MainWindow plot handles to histology display ports."""
    return DesktopHistologyDisplayPorts(
        aligned_plot=window.fig_hist,
        reference_plot=window.fig_hist_ref,
        scale_plot=window.fig_scale,
        scale_colorbar=window.fig_scale_cb,
        aligned_axis=window.ax_hist,
        reference_axis=window.ax_hist_ref,
        layout=window.fig_hist_layout,
        extra_y_axis=window.fig_hist_extra_yaxis,
        dotted_pen=window.kpen_dot,
        fit_curve=window.fit_plot,
        fit_scatter=window.fit_scatter,
        linear_fit_curve=window.fit_plot_lin,
        set_axis=window.set_axis,
        padding_provider=lambda: window.pad,
        probe_extent_query_kwargs=window._probe_extent_query_kwargs,
        fit_depth_um=lambda: window.display_state.depth_view.fit_depth_um,
        lin_fit_enabled=lambda: window.display_state.edit_settings.lin_fit,
        scale_factor_y_range=window._scale_factor_y_range,
        histology_available=lambda: window.histology_exists,
        brain_atlas=lambda: window.histology_context.brain_atlas,
        allen=lambda: window.allen,
    )


def desktop_reference_line_display_ports_from_main_window(
    window: Any,
) -> DesktopReferenceLineDisplayPorts:
    """Adapt MainWindow plot handles to reference-line display ports."""
    return DesktopReferenceLineDisplayPorts(
        histology_plot=window.fig_hist,
        image_plot=window.fig_img,
        line_plot=window.fig_line,
        probe_plot=window.fig_probe,
        perpendicular_plot=window.fig_hist_perp,
        fit_plot=window.fig_fit,
        on_lines_changed=window._capture_pending_reference_lines,
    )


def desktop_slice_display_ports_from_main_window(
    window: Any,
) -> DesktopSliceDisplayPorts:
    """Adapt MainWindow plot handles to slice display ports."""
    return DesktopSliceDisplayPorts(
        coronal_plot=window.fig_slice,
        coronal_layout=window.fig_slice_layout,
        histogram_alt=window.fig_slice_hist_alt,
        perpendicular_plot=window.fig_hist_perp,
        dotted_pen=window.kpen_dot,
        solid_pen=window.kpen_solid,
        reference_line_pen=window.reference_line_kpen,
        histology_exists=lambda: getattr(window, "histology_exists", False),
        slice_item=window.slice_item,
    )


def desktop_display_ports_from_main_window(window: Any) -> DesktopDisplayPorts:
    """Adapt MainWindow plot handles to desktop display-region ports."""
    return DesktopDisplayPorts(
        ephys=desktop_ephys_display_ports_from_main_window(window),
        histology=desktop_histology_display_ports_from_main_window(window),
        reference_lines=desktop_reference_line_display_ports_from_main_window(window),
        slice=desktop_slice_display_ports_from_main_window(window),
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
                capture_depth_plot_y_ranges=window._capture_depth_plot_y_ranges,
                restore_depth_plot_y_ranges=window._restore_depth_plot_y_ranges,
                create_reference_lines_for_previous_alignment=(
                    window._create_reference_lines_for_previous_alignment
                ),
                set_default_feature_y_range=window.set_default_feature_y_range,
                update_status=window.update_string,
            ),
            shank=DesktopShankRenderPorts(
                capture_plot_selection=window._capture_shank_plot_selection,
                prepare_runtime=window._prepare_shank_runtime_for_view,
                prepare_histology=window._prepare_shank_histology_for_view,
                apply_plot_data_state=window._apply_shank_plot_data_state,
                raw_image_payloads=lambda: window.raw_image_payloads,
                render_plot_menus=window._render_shank_plot_menus,
                render_histology_plots=lambda shank_idx: window.render_histology_plots(
                    shank_idx=shank_idx,
                ),
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
        export=DesktopExportPorts(
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
        ),
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
            capture_pending_reference_lines=window._capture_pending_reference_lines,
        ),
    )
