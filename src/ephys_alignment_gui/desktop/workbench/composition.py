"""Compose non-render desktop Workbench presenter clusters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.displays import DesktopDisplays
from ephys_alignment_gui.desktop.displays.ephys_plot_exporter import (
    DesktopEphysPlotExporter,
)
from ephys_alignment_gui.desktop.displays.plot_exporter import (
    DesktopPlotExporter,
    HistologyExportHandles,
)
from ephys_alignment_gui.desktop.presenters.interaction_presenter import (
    DesktopInteractionCallbacks,
    DesktopInteractionPresenter,
    DesktopInteractionWidgets,
)
from ephys_alignment_gui.desktop.presenters.lifecycle_presenter import (
    DesktopLifecycleCallbacks,
    DesktopLifecyclePresenter,
)
from ephys_alignment_gui.desktop.presenters.load_data_presenter import (
    DesktopLoadDataCallbacks,
    DesktopLoadDataPresenter,
)
from ephys_alignment_gui.desktop.presenters.load_preflight_presenter import (
    DesktopLoadPreflightPresenter,
    DesktopOutputFolderPrompt,
    OutputFolderPromptCallbacks,
)
from ephys_alignment_gui.desktop.presenters.mouse_root_presenter import (
    DesktopMouseRootCallbacks,
    DesktopMouseRootPresenter,
)
from ephys_alignment_gui.desktop.presenters.output_path_presenter import (
    DesktopOutputPathPresenter,
)
from ephys_alignment_gui.desktop.presenters.path_dialog_presenter import (
    DesktopPathDialogCallbacks,
    DesktopPathDialogPresenter,
)
from ephys_alignment_gui.desktop.presenters.plot_export_presenter import (
    DesktopPlotExportPresenter,
)
from ephys_alignment_gui.desktop.presenters.previous_alignment_load_presenter import (
    DesktopPreviousAlignmentLoadPresenter,
    PreviousAlignmentLoadCallbacks,
)
from ephys_alignment_gui.desktop.presenters.probe_selection_presenter import (
    DesktopProbeSelectionCallbacks,
    DesktopProbeSelectionPresenter,
)
from ephys_alignment_gui.desktop.presenters.save_presenter import (
    DesktopSaveCallbacks,
    DesktopSavePresenter,
)
from ephys_alignment_gui.desktop.presenters.session_selection_presenter import (
    DesktopSessionSelectionCallbacks,
    DesktopSessionSelectionPresenter,
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
class DesktopWorkbenchPresenterCluster:
    """Non-render presenters and helpers owned by DesktopWorkbench."""

    load_data_presenter: DesktopLoadDataPresenter
    probe_selection_presenter: DesktopProbeSelectionPresenter
    session_selection_presenter: DesktopSessionSelectionPresenter
    mouse_root_presenter: DesktopMouseRootPresenter
    output_path_presenter: DesktopOutputPathPresenter
    path_dialog_presenter: DesktopPathDialogPresenter
    load_preflight_presenter: DesktopLoadPreflightPresenter
    output_folder_prompt: DesktopOutputFolderPrompt
    folder_dialog: DesktopFolderDialog
    save_presenter: DesktopSavePresenter
    previous_alignment_load_presenter: DesktopPreviousAlignmentLoadPresenter
    plot_exporter: DesktopPlotExporter
    plot_export_presenter: DesktopPlotExportPresenter
    interaction_presenter: DesktopInteractionPresenter
    lifecycle_presenter: DesktopLifecyclePresenter


def build_desktop_workbench_presenter_cluster(
    *,
    app: Any,
    parent: Any,
    views: DesktopViews,
    displays: DesktopDisplays,
    ports: DesktopWorkbenchPorts,
    render_cluster: DesktopRenderCluster,
) -> DesktopWorkbenchPresenterCluster:
    """Build desktop Workbench presenters outside the Workbench class."""
    output_path_presenter = DesktopOutputPathPresenter(
        commands=app.commands.paths,
        path_view=views.path,
    )
    lifecycle_presenter = DesktopLifecyclePresenter(
        app=app,
        displays=displays,
        callbacks=_lifecycle_callbacks(ports.lifecycle),
    )
    load_data_presenter = DesktopLoadDataPresenter(
        app=app,
        selection_view=views.selection,
        callbacks=_load_data_callbacks(
            ports.load_data,
            ports.busy,
            output_path_presenter,
            render_cluster.shank_presenter,
            lifecycle_presenter,
            render_cluster.reference_line_presenter,
        ),
    )
    probe_selection_presenter = DesktopProbeSelectionPresenter(
        app=app,
        selection_view=views.selection,
        callbacks=_probe_selection_callbacks(
            ports.busy,
            output_path_presenter,
            load_data_presenter,
            lifecycle_presenter.show_empty_state,
            render_cluster.reference_line_presenter,
        ),
    )
    session_selection_presenter = DesktopSessionSelectionPresenter(
        app=app,
        selection_view=views.selection,
        callbacks=_session_selection_callbacks(
            render_cluster.reference_line_presenter,
            probe_selection_presenter,
            lifecycle_presenter.show_empty_state,
        ),
    )
    mouse_root_presenter = DesktopMouseRootPresenter(
        commands=app.commands.metadata,
        path_view=views.path,
        selection_view=views.selection,
        callbacks=_mouse_root_callbacks(
            ports.busy,
            session_selection_presenter,
        ),
    )
    folder_dialog = DesktopFolderDialog(parent=None)
    path_dialog_presenter = DesktopPathDialogPresenter(
        folder_dialog=folder_dialog,
        callbacks=DesktopPathDialogCallbacks(
            active_mouse_root=app.queries.workspace.active_mouse_root_path,
            set_mouse_root=mouse_root_presenter.set_mouse_root,
            active_output_root=app.queries.workspace.active_output_root,
            set_save_root=output_path_presenter.set_save_root,
        ),
    )
    output_folder_prompt = DesktopOutputFolderPrompt(
        parent=parent,
        callbacks=OutputFolderPromptCallbacks(
            derive_output_directory_from_save_root=(
                output_path_presenter.derive_output_directory_from_save_root
            ),
            has_output_directory=app.queries.workspace.has_output_directory,
            select_output_folder=path_dialog_presenter.select_output_root,
        ),
    )
    load_preflight_presenter = DesktopLoadPreflightPresenter(
        can_load_data=app.commands.load.can_load_data,
        load_heavy_data=load_data_presenter.load_heavy_data,
        output_folder_prompt=output_folder_prompt,
    )
    save_presenter = DesktopSavePresenter(
        commands=app.commands.persistence,
        events=app.events,
        callbacks=_save_callbacks(
            ports.save,
            output_folder_prompt,
            load_preflight_presenter,
        ),
    )
    previous_alignment_load_presenter = DesktopPreviousAlignmentLoadPresenter(
        commands=app.commands.persistence,
        events=app.events,
        callbacks=_previous_alignment_load_callbacks(
            ports.previous_alignment_load,
            folder_dialog,
            render_cluster.alignment_selection_actions,
        ),
    )
    interaction_presenter = _interaction_presenter(
        ports.interaction,
        app=app,
        displays=displays,
        render_cluster=render_cluster,
    )
    plot_exporter = _plot_exporter(
        ports.export,
        displays=displays,
    )
    plot_export_presenter = DesktopPlotExportPresenter(
        app=app,
        plot_exporter=plot_exporter,
        output_folder_prompt=output_folder_prompt,
    )
    return DesktopWorkbenchPresenterCluster(
        load_data_presenter=load_data_presenter,
        probe_selection_presenter=probe_selection_presenter,
        session_selection_presenter=session_selection_presenter,
        mouse_root_presenter=mouse_root_presenter,
        output_path_presenter=output_path_presenter,
        path_dialog_presenter=path_dialog_presenter,
        load_preflight_presenter=load_preflight_presenter,
        output_folder_prompt=output_folder_prompt,
        folder_dialog=folder_dialog,
        save_presenter=save_presenter,
        previous_alignment_load_presenter=previous_alignment_load_presenter,
        plot_exporter=plot_exporter,
        plot_export_presenter=plot_export_presenter,
        interaction_presenter=interaction_presenter,
        lifecycle_presenter=lifecycle_presenter,
    )


def _save_callbacks(
    ports: DesktopSavePorts,
    output_folder_prompt: DesktopOutputFolderPrompt,
    load_preflight_presenter: DesktopLoadPreflightPresenter,
) -> DesktopSaveCallbacks:
    """Build callbacks for save/QC presentation."""
    return DesktopSaveCallbacks(
        ensure_output_directory=output_folder_prompt.ensure_for_save,
        log_requirement=load_preflight_presenter.log_requirement,
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
        select_folder=lambda: folder_dialog.select_existing_directory_text(
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
) -> DesktopPlotExporter:
    """Build the desktop plot exporter cluster."""
    ephys_exporter = DesktopEphysPlotExporter(
        presenter=displays.ephys.plot_presenter,
        panel=displays.ephys.panel,
        layout=export_view.ephys_layout(),
        callbacks=export_view.ephys_callbacks(
            add_lines_points=displays.reference_lines.add_to_plots,
        ),
    )
    return DesktopPlotExporter(
        ephys_exporter=ephys_exporter,
        slice_handles=export_view.slice_handles(displays.slice),
        slice_style=export_view.slice_style(),
        histology_handles=HistologyExportHandles(
            histology_display=displays.histology,
        ),
        callbacks=export_view.plot_callbacks(),
        add_lines_points=displays.reference_lines.add_to_plots,
    )


def _interaction_presenter(
    ports: DesktopInteractionPorts,
    *,
    app: Any,
    displays: DesktopDisplays,
    render_cluster: DesktopRenderCluster,
) -> DesktopInteractionPresenter:
    """Build the desktop interaction presenter."""
    return DesktopInteractionPresenter(
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
    """Build callbacks for desktop lifecycle presentation."""
    return DesktopLifecycleCallbacks(
        close_popups=ports.close_popups,
        reset_raw_image_payloads=ports.reset_raw_image_payloads,
        show_empty_state=ports.show_empty_state,
        collect_garbage=ports.collect_garbage,
    )


def _load_data_callbacks(
    load_data_ports: DesktopLoadDataPorts,
    busy_ports: DesktopBusyPorts,
    output_path_presenter: DesktopOutputPathPresenter,
    shank_presenter: Any,
    lifecycle_presenter: DesktopLifecyclePresenter,
    reference_line_presenter: Any,
) -> DesktopLoadDataCallbacks:
    """Build callbacks for cached/fresh data loading."""
    return DesktopLoadDataCallbacks(
        reference_line_positions=(
            reference_line_presenter.reference_line_display.positions
        ),
        prepare_for_fresh_stream_load=(
            lifecycle_presenter.prepare_for_fresh_stream_load
        ),
        display_output_directory=output_path_presenter.display_output_directory,
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
    output_path_presenter: DesktopOutputPathPresenter,
    load_data_presenter: DesktopLoadDataPresenter,
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
                load_data_presenter.present_cached_probe_selection(
                    session_name=session,
                    probe_name=probe,
                    target_shank=shank,
                )
            )
        ),
        show_empty_state=show_empty_state,
        busy_context=busy_ports.busy_context,
        display_output_directory=output_path_presenter.display_output_directory,
    )


def _session_selection_callbacks(
    reference_line_presenter: Any,
    probe_selection_presenter: DesktopProbeSelectionPresenter,
    show_empty_state: Callable[[], None],
) -> DesktopSessionSelectionCallbacks:
    """Build callbacks for session selection."""
    return DesktopSessionSelectionCallbacks(
        capture_pending_reference_lines=(
            reference_line_presenter.capture_pending_reference_lines
        ),
        show_empty_state=show_empty_state,
        select_first_probe=probe_selection_presenter.probe_selected,
    )


def _mouse_root_callbacks(
    busy_ports: DesktopBusyPorts,
    session_selection_presenter: DesktopSessionSelectionPresenter,
) -> DesktopMouseRootCallbacks:
    """Build callbacks for mouse-root loading."""
    return DesktopMouseRootCallbacks(
        busy_context=busy_ports.busy_context,
        select_first_session=session_selection_presenter.session_selected,
    )
