"""Compose non-render desktop Workbench coordinator clusters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.coordinators.interaction_coordinator import (
    DesktopInteractionCallbacks,
    DesktopInteractionCoordinator,
    DesktopInteractionWidgets,
)
from ephys_alignment_gui.desktop.coordinators.lifecycle_coordinator import (
    DesktopLifecycleCallbacks,
    DesktopLifecycleCoordinator,
)
from ephys_alignment_gui.desktop.coordinators.load_data_coordinator import (
    DesktopLoadDataCallbacks,
    DesktopLoadDataCoordinator,
)
from ephys_alignment_gui.desktop.coordinators.load_preflight_coordinator import (
    DesktopLoadPreflightCoordinator,
    DesktopOutputFolderPrompt,
    OutputFolderPromptCallbacks,
)
from ephys_alignment_gui.desktop.coordinators.mouse_root_coordinator import (
    DesktopMouseRootCallbacks,
    DesktopMouseRootCoordinator,
)
from ephys_alignment_gui.desktop.coordinators.output_path_coordinator import (
    DesktopOutputPathCoordinator,
)
from ephys_alignment_gui.desktop.coordinators.path_dialog_coordinator import (
    DesktopPathDialogCallbacks,
    DesktopPathDialogCoordinator,
)
from ephys_alignment_gui.desktop.coordinators.plot_export_coordinator import (
    DesktopPlotExportCoordinator,
)
from ephys_alignment_gui.desktop.coordinators.previous_alignment_load_coordinator import (
    DesktopPreviousAlignmentLoadCoordinator,
    PreviousAlignmentLoadCallbacks,
)
from ephys_alignment_gui.desktop.coordinators.probe_selection_coordinator import (
    DesktopProbeSelectionCallbacks,
    DesktopProbeSelectionCoordinator,
)
from ephys_alignment_gui.desktop.coordinators.save_coordinator import (
    DesktopSaveCallbacks,
    DesktopSaveCoordinator,
)
from ephys_alignment_gui.desktop.coordinators.session_selection_coordinator import (
    DesktopSessionSelectionCallbacks,
    DesktopSessionSelectionCoordinator,
)
from ephys_alignment_gui.desktop.displays import DesktopDisplays
from ephys_alignment_gui.desktop.displays.ephys_plot_exporter import (
    DesktopEphysPlotExporter,
)
from ephys_alignment_gui.desktop.displays.plot_exporter import (
    DesktopPlotExporter,
    HistologyExportHandles,
)
from ephys_alignment_gui.desktop.shell.folder_dialog import DesktopFolderDialog
from ephys_alignment_gui.desktop.views import DesktopViews
from ephys_alignment_gui.desktop.workbench.port_types import (
    DesktopBusyPorts,
    DesktopInteractionPorts,
    DesktopLifecyclePorts,
    DesktopLoadDataPorts,
    DesktopPreviousAlignmentLoadPorts,
    DesktopSavePorts,
    DesktopWorkbenchPorts,
)
from ephys_alignment_gui.desktop.workbench.render_composition import (
    DesktopRenderCluster,
)


@dataclass(frozen=True)
class DesktopWorkbenchCoordinatorCluster:
    """Non-render coordinators and helpers owned by DesktopWorkbench."""

    load_data_coordinator: DesktopLoadDataCoordinator
    probe_selection_coordinator: DesktopProbeSelectionCoordinator
    session_selection_coordinator: DesktopSessionSelectionCoordinator
    mouse_root_coordinator: DesktopMouseRootCoordinator
    output_path_coordinator: DesktopOutputPathCoordinator
    path_dialog_coordinator: DesktopPathDialogCoordinator
    load_preflight_coordinator: DesktopLoadPreflightCoordinator
    output_folder_prompt: DesktopOutputFolderPrompt
    folder_dialog: DesktopFolderDialog
    save_coordinator: DesktopSaveCoordinator
    previous_alignment_load_coordinator: DesktopPreviousAlignmentLoadCoordinator
    plot_exporter: DesktopPlotExporter
    plot_export_coordinator: DesktopPlotExportCoordinator
    interaction_coordinator: DesktopInteractionCoordinator
    lifecycle_coordinator: DesktopLifecycleCoordinator


def build_desktop_workbench_coordinator_cluster(
    *,
    app: Any,
    parent: Any,
    views: DesktopViews,
    displays: DesktopDisplays,
    ports: DesktopWorkbenchPorts,
    render_cluster: DesktopRenderCluster,
) -> DesktopWorkbenchCoordinatorCluster:
    """Build desktop Workbench coordinators outside the Workbench class."""
    output_path_coordinator = DesktopOutputPathCoordinator(
        commands=app.commands.paths,
        events=app.events,
        path_view=views.path,
    )
    lifecycle_coordinator = DesktopLifecycleCoordinator(
        app=app,
        displays=displays,
        callbacks=_lifecycle_callbacks(ports.lifecycle),
    )
    load_data_coordinator = DesktopLoadDataCoordinator(
        app=app,
        selection_view=views.selection,
        callbacks=_load_data_callbacks(
            ports.load_data,
            ports.busy,
            output_path_coordinator,
            render_cluster.shank_presenter,
            lifecycle_coordinator,
            render_cluster.reference_line_presenter,
        ),
    )
    probe_selection_coordinator = DesktopProbeSelectionCoordinator(
        app=app,
        selection_view=views.selection,
        callbacks=_probe_selection_callbacks(
            ports.busy,
            output_path_coordinator,
            load_data_coordinator,
            lifecycle_coordinator.show_empty_state,
            render_cluster.reference_line_presenter,
        ),
    )
    session_selection_coordinator = DesktopSessionSelectionCoordinator(
        app=app,
        selection_view=views.selection,
        callbacks=_session_selection_callbacks(
            render_cluster.reference_line_presenter,
            probe_selection_coordinator,
            lifecycle_coordinator.show_empty_state,
        ),
    )
    mouse_root_coordinator = DesktopMouseRootCoordinator(
        commands=app.commands.metadata,
        path_view=views.path,
        selection_view=views.selection,
        callbacks=_mouse_root_callbacks(
            ports.busy,
            session_selection_coordinator,
        ),
    )
    folder_dialog = DesktopFolderDialog(parent=None)
    path_dialog_coordinator = DesktopPathDialogCoordinator(
        folder_dialog=folder_dialog,
        callbacks=DesktopPathDialogCallbacks(
            active_mouse_root=app.queries.workspace.active_mouse_root_path,
            set_mouse_root=mouse_root_coordinator.set_mouse_root,
            active_output_root=app.queries.workspace.active_output_root,
            set_save_root=output_path_coordinator.set_save_root,
        ),
    )
    output_folder_prompt = DesktopOutputFolderPrompt(
        parent=parent,
        callbacks=OutputFolderPromptCallbacks(
            derive_output_directory_from_save_root=(
                output_path_coordinator.derive_output_directory_from_save_root
            ),
            has_output_directory=app.queries.workspace.has_output_directory,
            select_output_folder=path_dialog_coordinator.select_output_root,
        ),
    )
    load_preflight_coordinator = DesktopLoadPreflightCoordinator(
        can_load_data=app.commands.load.can_load_data,
        load_heavy_data=load_data_coordinator.load_heavy_data,
        output_folder_prompt=output_folder_prompt,
    )
    save_coordinator = DesktopSaveCoordinator(
        commands=app.commands.persistence,
        events=app.events,
        callbacks=_save_callbacks(
            ports.save,
            output_folder_prompt,
            load_preflight_coordinator,
        ),
    )
    previous_alignment_load_coordinator = DesktopPreviousAlignmentLoadCoordinator(
        commands=app.commands.persistence,
        events=app.events,
        callbacks=_previous_alignment_load_callbacks(
            ports.previous_alignment_load,
            folder_dialog,
            render_cluster.alignment_selection_actions,
        ),
    )
    interaction_coordinator = _interaction_coordinator(
        ports.interaction,
        app=app,
        displays=displays,
        render_cluster=render_cluster,
    )
    plot_exporter = _plot_exporter(
        ports.export,
        displays=displays,
        render_cluster=render_cluster,
    )
    plot_export_coordinator = DesktopPlotExportCoordinator(
        app=app,
        plot_exporter=plot_exporter,
        output_folder_prompt=output_folder_prompt,
    )
    return DesktopWorkbenchCoordinatorCluster(
        load_data_coordinator=load_data_coordinator,
        probe_selection_coordinator=probe_selection_coordinator,
        session_selection_coordinator=session_selection_coordinator,
        mouse_root_coordinator=mouse_root_coordinator,
        output_path_coordinator=output_path_coordinator,
        path_dialog_coordinator=path_dialog_coordinator,
        load_preflight_coordinator=load_preflight_coordinator,
        output_folder_prompt=output_folder_prompt,
        folder_dialog=folder_dialog,
        save_coordinator=save_coordinator,
        previous_alignment_load_coordinator=previous_alignment_load_coordinator,
        plot_exporter=plot_exporter,
        plot_export_coordinator=plot_export_coordinator,
        interaction_coordinator=interaction_coordinator,
        lifecycle_coordinator=lifecycle_coordinator,
    )


def _save_callbacks(
    ports: DesktopSavePorts,
    output_folder_prompt: DesktopOutputFolderPrompt,
    load_preflight_coordinator: DesktopLoadPreflightCoordinator,
) -> DesktopSaveCallbacks:
    """Build callbacks for save/QC coordination."""
    return DesktopSaveCallbacks(
        ensure_output_directory=output_folder_prompt.ensure_for_save,
        log_requirement=load_preflight_coordinator.log_requirement,
        use_docdb=ports.use_docdb,
        render_alignment_choices=ports.render_alignment_choices,
        busy_context=ports.busy_context,
        complete_button=ports.complete_button,
        histology_available=ports.histology_available,
        open_qc_dialog=ports.open_qc_dialog,
        ephys_qc=ports.ephys_qc,
        selected_qc_descriptions=ports.selected_qc_descriptions,
        warning=ports.warning,
    )


def _previous_alignment_load_callbacks(
    ports: DesktopPreviousAlignmentLoadPorts,
    folder_dialog: DesktopFolderDialog,
    alignment_selection_actions: Any,
) -> PreviousAlignmentLoadCallbacks:
    """Build callbacks for previous-alignment loading."""
    return PreviousAlignmentLoadCallbacks(
        select_folder=lambda: folder_dialog.select_existing_directory(
            "Load Existing Alignments",
        ),
        use_docdb=ports.use_docdb,
        set_reload_folder_text=ports.set_reload_folder_text,
        render_alignment_choices=ports.render_alignment_choices,
        select_alignment=alignment_selection_actions.alignment_selected,
        busy_context=ports.busy_context,
        reload_button=ports.reload_button,
    )


def _plot_exporter(
    export_view: Any,
    *,
    displays: DesktopDisplays,
    render_cluster: DesktopRenderCluster,
) -> DesktopPlotExporter:
    """Build the desktop plot exporter cluster."""
    ephys_exporter = DesktopEphysPlotExporter(
        presenter=render_cluster.ephys_plot_presenter,
        panel=displays.ephys.panel,
        layout=export_view.ephys_layout(),
        callbacks=export_view.ephys_callbacks(
            add_lines_points=displays.reference_lines.add_to_plots,
        ),
    )
    return DesktopPlotExporter(
        ephys_exporter=ephys_exporter,
        slice_handles=export_view.slice_handles(
            displays.slice,
            slice_panel_presenter=render_cluster.slice_panel_presenter,
            slice_menu_coordinator=render_cluster.slice_menu_coordinator,
        ),
        slice_style=export_view.slice_style(),
        histology_handles=HistologyExportHandles(
            histology_display=displays.histology,
        ),
        callbacks=export_view.plot_callbacks(),
        add_lines_points=displays.reference_lines.add_to_plots,
    )


def _interaction_coordinator(
    ports: DesktopInteractionPorts,
    *,
    app: Any,
    displays: DesktopDisplays,
    render_cluster: DesktopRenderCluster,
) -> DesktopInteractionCoordinator:
    """Build the desktop interaction coordinator."""
    return DesktopInteractionCoordinator(
        app=app,
        popup_manager=ports.popup_manager,
        ephys_panel=displays.ephys.panel,
        histology_display=displays.histology,
        reference_line_display=displays.reference_lines,
        widgets=DesktopInteractionWidgets(
            struct_list=ports.struct_list,
            struct_view=ports.struct_view,
            struct_description=ports.struct_description,
            scale_plot=ports.scale_plot,
            histology_plot=ports.histology_plot,
            histology_reference_plot=ports.histology_reference_plot,
            scale_axis=ports.scale_axis,
            bar_colour=ports.bar_colour,
            line_pen=ports.line_pen,
        ),
        callbacks=DesktopInteractionCallbacks(
            histology_available=ports.histology_available,
            activate_window=ports.activate_window,
            set_axis=ports.set_axis,
            capture_pending_reference_lines=(
                render_cluster.reference_line_presenter.capture_pending_reference_lines
            ),
        ),
    )


def _lifecycle_callbacks(
    ports: DesktopLifecyclePorts,
) -> DesktopLifecycleCallbacks:
    """Build callbacks for desktop lifecycle coordination."""
    return DesktopLifecycleCallbacks(
        close_popups=ports.close_popups,
        reset_raw_image_payloads=ports.reset_raw_image_payloads,
        show_empty_state=ports.show_empty_state,
        collect_garbage=ports.collect_garbage,
    )


def _load_data_callbacks(
    load_data_ports: DesktopLoadDataPorts,
    busy_ports: DesktopBusyPorts,
    output_path_coordinator: DesktopOutputPathCoordinator,
    shank_presenter: Any,
    lifecycle_coordinator: DesktopLifecycleCoordinator,
    reference_line_presenter: Any,
) -> DesktopLoadDataCallbacks:
    """Build callbacks for cached/fresh data loading."""
    return DesktopLoadDataCallbacks(
        reference_line_positions=(
            reference_line_presenter.reference_line_display.positions
        ),
        prepare_for_fresh_stream_load=(
            lifecycle_coordinator.prepare_for_fresh_stream_load
        ),
        render_loaded_shank=lambda shank_idx, preserve: (
            shank_presenter.render_loaded_shank(
                shank_idx=shank_idx,
                preserve_plot_selection=preserve,
            )
        ),
        clear_empty_state=load_data_ports.clear_empty_state,
        busy_context=busy_ports.busy_context,
    )


def _probe_selection_callbacks(
    busy_ports: DesktopBusyPorts,
    output_path_coordinator: DesktopOutputPathCoordinator,
    load_data_coordinator: DesktopLoadDataCoordinator,
    show_empty_state: Callable[[], None],
    reference_line_presenter: Any,
) -> DesktopProbeSelectionCallbacks:
    """Build callbacks for probe selection."""
    return DesktopProbeSelectionCallbacks(
        capture_pending_reference_lines=(
            reference_line_presenter.capture_pending_reference_lines
        ),
        present_cached_probe_selection=(
            lambda session, probe, shank: (
                load_data_coordinator.present_cached_probe_selection(
                    session_name=session,
                    probe_name=probe,
                    target_shank=shank,
                )
            )
        ),
        show_empty_state=show_empty_state,
        busy_context=busy_ports.busy_context,
    )


def _session_selection_callbacks(
    reference_line_presenter: Any,
    probe_selection_coordinator: DesktopProbeSelectionCoordinator,
    show_empty_state: Callable[[], None],
) -> DesktopSessionSelectionCallbacks:
    """Build callbacks for session selection."""
    return DesktopSessionSelectionCallbacks(
        capture_pending_reference_lines=(
            reference_line_presenter.capture_pending_reference_lines
        ),
        show_empty_state=show_empty_state,
        select_first_probe=probe_selection_coordinator.probe_selected,
    )


def _mouse_root_callbacks(
    busy_ports: DesktopBusyPorts,
    session_selection_coordinator: DesktopSessionSelectionCoordinator,
) -> DesktopMouseRootCallbacks:
    """Build callbacks for mouse-root loading."""
    return DesktopMouseRootCallbacks(
        busy_context=busy_ports.busy_context,
        select_first_session=session_selection_coordinator.session_selected,
    )
