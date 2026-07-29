"""Desktop composition shell for focused presenters."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ephys_alignment_gui.desktop_alignment_presenter import (
    DesktopAlignmentPresenter,
    DesktopAlignmentRenderCallbacks,
)
from ephys_alignment_gui.desktop_displays import DesktopDisplays
from ephys_alignment_gui.desktop_ephys_plot_exporter import (
    DesktopEphysPlotExporter,
    EphysExportCallbacks,
    EphysExportLayout,
    EphysExportSizes,
)
from ephys_alignment_gui.desktop_folder_dialog import DesktopFolderDialog
from ephys_alignment_gui.desktop_histology_refresh_presenter import (
    DesktopHistologyRefreshPresenter,
)
from ephys_alignment_gui.desktop_interaction_presenter import (
    DesktopInteractionCallbacks,
    DesktopInteractionPresenter,
    DesktopInteractionWidgets,
)
from ephys_alignment_gui.desktop_lifecycle_presenter import (
    DesktopLifecycleCallbacks,
    DesktopLifecyclePresenter,
)
from ephys_alignment_gui.desktop_load_data_presenter import (
    DesktopLoadDataCallbacks,
    DesktopLoadDataPresenter,
)
from ephys_alignment_gui.desktop_load_workflow_presenter import (
    DesktopLoadWorkflowPresenter,
    DesktopOutputFolderPrompt,
    OutputFolderPromptCallbacks,
)
from ephys_alignment_gui.desktop_mouse_root_presenter import (
    DesktopMouseRootCallbacks,
    DesktopMouseRootPresenter,
)
from ephys_alignment_gui.desktop_output_path_presenter import DesktopOutputPathPresenter
from ephys_alignment_gui.desktop_path_dialog_presenter import (
    DesktopPathDialogCallbacks,
    DesktopPathDialogPresenter,
)
from ephys_alignment_gui.desktop_plot_exporter import (
    DesktopPlotExportCallbacks,
    DesktopPlotExporter,
    HistologyExportHandles,
    SliceExportGeometry,
    SliceExportHandles,
    SliceExportStyle,
)
from ephys_alignment_gui.desktop_previous_alignment_load_presenter import (
    DesktopPreviousAlignmentLoadPresenter,
    PreviousAlignmentLoadCallbacks,
)
from ephys_alignment_gui.desktop_probe_selection_presenter import (
    DesktopProbeSelectionCallbacks,
    DesktopProbeSelectionPresenter,
)
from ephys_alignment_gui.desktop_reference_line_presenter import (
    DesktopReferenceLinePresenter,
)
from ephys_alignment_gui.desktop_save_workflow_presenter import (
    DesktopSaveWorkflowCallbacks,
    DesktopSaveWorkflowPresenter,
)
from ephys_alignment_gui.desktop_session_selection_presenter import (
    DesktopSessionSelectionCallbacks,
    DesktopSessionSelectionPresenter,
)
from ephys_alignment_gui.desktop_shank_presenter import (
    DesktopShankPresenter,
    DesktopShankRenderCallbacks,
)
from ephys_alignment_gui.event_bus import EventSubscription


@dataclass(frozen=True)
class DesktopAlignmentRenderPorts:
    """Desktop operations needed to render alignment edits."""

    restore_lin_fit: Callable[[bool | None], None]
    capture_depth_plot_y_ranges: Callable[[], Any]
    restore_depth_plot_y_ranges: Callable[[Any], None]
    create_reference_lines_for_previous_alignment: Callable[[], None]
    set_default_feature_y_range: Callable[[], None]
    update_status: Callable[[], None]


@dataclass(frozen=True)
class DesktopShankRenderPorts:
    """Desktop operations needed to render an active shank."""

    capture_plot_selection: Callable[[bool], Any]
    render_alignment_choices: Callable[[list[str]], None]
    apply_plot_data_state: Callable[[Any], None]
    raw_image_payloads: Callable[[], Any]
    render_plot_menus: Callable[[Any], None]
    configure_view: Callable[[bool], None]
    offline: Callable[[], bool]


@dataclass(frozen=True)
class DesktopRenderPorts:
    """MainWindow render ports consumed by focused desktop presenters."""

    alignment: DesktopAlignmentRenderPorts
    shank: DesktopShankRenderPorts


@dataclass(frozen=True)
class DesktopSaveWorkflowPorts:
    """Desktop operations needed by save and QC workflows."""

    use_docdb: Callable[[], bool]
    render_alignment_choices: Callable[[list[str]], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    complete_button: Callable[[], Any]
    histology_available: Callable[[], bool]
    open_qc_dialog: Callable[[], None]
    ephys_qc: Callable[[], str]
    selected_qc_descriptions: Callable[[], list[str]]
    warning: Callable[[str, str], Any]


@dataclass(frozen=True)
class DesktopPreviousAlignmentLoadPorts:
    """Desktop operations needed by previous-alignment loading."""

    use_docdb: Callable[[], bool]
    set_reload_folder_text: Callable[[str], None]
    render_alignment_choices: Callable[[list[str]], None]
    select_alignment: Callable[[int], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    reload_button: Callable[[], Any]


@dataclass(frozen=True)
class DesktopExportPorts:
    """Desktop operations and handles needed by plot export workflows."""

    ephys_graphics_layout: Any
    ephys_data_area: Any
    slice_plot: Any
    slice_trajectory_pen: Any
    reset_axis: Callable[[], None]
    set_view: Callable[..., None]
    set_axis: Callable[..., Any]
    set_font: Callable[..., None]
    ephys_sizes: Callable[[], tuple[float, float]]
    slice_geometry: Callable[[], tuple[float, float, Any]]


@dataclass(frozen=True)
class DesktopInteractionPorts:
    """Desktop operations and handles needed by interaction presentation."""

    popup_manager: Any
    region_lookup_service: Any
    struct_list: Any
    struct_view: Any
    struct_description: Any
    scale_plot: Any
    histology_plot: Any
    histology_reference_plot: Any
    scale_axis: Any
    bar_colour: Any
    line_pen: Any
    histology_available: Callable[[], bool]
    activate_window: Callable[[], None]
    set_axis: Callable[..., Any]


@dataclass(frozen=True)
class DesktopLifecyclePorts:
    """Desktop-only operations for stream/session lifecycle presentation."""

    close_popups: Callable[[], None]
    reset_raw_image_payloads: Callable[[], None]
    show_empty_state: Callable[[], None]
    collect_garbage: Callable[[], None]


@dataclass(frozen=True)
class DesktopWorkbenchPorts:
    """MainWindow ports consumed by Workbench presenter composition."""

    render: DesktopRenderPorts
    selection: DesktopSelectionWorkflowCallbacks
    lifecycle: DesktopLifecyclePorts
    save_workflow: DesktopSaveWorkflowPorts
    previous_alignment_load: DesktopPreviousAlignmentLoadPorts
    export: DesktopExportPorts
    interaction: DesktopInteractionPorts


@dataclass(frozen=True)
class DesktopSelectionWorkflowCallbacks:
    """MainWindow bridge callbacks for selection and load presenters."""

    select_shank_for_view: Callable[[int, str], int | None]
    clear_empty_state: Callable[[], None]
    set_histology_available: Callable[[bool], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    mouse_root_loaded: Callable[[], bool]
    active_shank_idx: Callable[[], int]
    clear_histology_context: Callable[[], None]
    select_first_session: Callable[[], None]
    select_first_probe: Callable[[], None]


@dataclass
class DesktopWorkbench:
    """Own focused desktop presenters and desktop event subscription lifecycle."""

    app: Any
    displays: DesktopDisplays
    alignment_presenter: DesktopAlignmentPresenter
    shank_presenter: DesktopShankPresenter
    load_data_presenter: DesktopLoadDataPresenter
    probe_selection_presenter: DesktopProbeSelectionPresenter
    session_selection_presenter: DesktopSessionSelectionPresenter
    mouse_root_presenter: DesktopMouseRootPresenter
    output_path_presenter: DesktopOutputPathPresenter
    path_dialog_presenter: DesktopPathDialogPresenter
    load_workflow_presenter: DesktopLoadWorkflowPresenter
    output_folder_prompt: DesktopOutputFolderPrompt
    folder_dialog: DesktopFolderDialog
    save_workflow_presenter: DesktopSaveWorkflowPresenter
    previous_alignment_load_presenter: DesktopPreviousAlignmentLoadPresenter
    plot_exporter: DesktopPlotExporter
    interaction_presenter: DesktopInteractionPresenter
    lifecycle_presenter: DesktopLifecyclePresenter
    reference_line_presenter: DesktopReferenceLinePresenter
    histology_refresh_presenter: DesktopHistologyRefreshPresenter
    _event_subscriptions: list[EventSubscription] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        app: Any,
        selection_view: Any,
        path_view: Any,
        parent: Any,
        displays: DesktopDisplays,
        ports: DesktopWorkbenchPorts,
    ) -> DesktopWorkbench:
        """Build and configure the focused desktop presenters."""
        output_path_presenter = DesktopOutputPathPresenter(
            commands=app.commands,
            path_view=path_view,
        )
        alignment_presenter = DesktopAlignmentPresenter(app.events)
        alignment_presenter.configure(
            queries=app.queries,
            callbacks=cls._alignment_render_callbacks(
                ports.render.alignment,
                displays,
            ),
        )
        histology_refresh_presenter = DesktopHistologyRefreshPresenter(
            app=app,
            histology_display=displays.histology,
            slice_display=displays.slice,
            reference_line_display=displays.reference_lines,
        )
        shank_presenter = DesktopShankPresenter(app)
        shank_presenter.configure(
            callbacks=cls._shank_render_callbacks(
                ports.render.shank,
                displays,
                histology_refresh_presenter,
            )
        )
        lifecycle_presenter = DesktopLifecyclePresenter(
            app=app,
            displays=displays,
            callbacks=cls._lifecycle_callbacks(ports.lifecycle),
        )
        reference_line_presenter = DesktopReferenceLinePresenter(
            app=app,
            reference_line_display=displays.reference_lines,
        )
        displays.reference_lines.set_lines_changed_callback(
            reference_line_presenter.capture_pending_reference_lines
        )
        load_data_presenter = DesktopLoadDataPresenter(
            app=app,
            selection_view=selection_view,
            callbacks=cls._load_data_callbacks(
                ports.selection,
                output_path_presenter,
                shank_presenter,
                lifecycle_presenter,
                reference_line_presenter,
            ),
        )
        probe_selection_presenter = DesktopProbeSelectionPresenter(
            commands=app.commands,
            selection_view=selection_view,
            callbacks=cls._probe_selection_callbacks(
                ports.selection,
                output_path_presenter,
                load_data_presenter,
                lifecycle_presenter,
                reference_line_presenter,
            ),
        )
        session_selection_presenter = DesktopSessionSelectionPresenter(
            commands=app.commands,
            selection_view=selection_view,
            callbacks=cls._session_selection_callbacks(
                ports.selection,
                lifecycle_presenter,
                reference_line_presenter,
            ),
        )
        mouse_root_presenter = DesktopMouseRootPresenter(
            commands=app.commands,
            path_view=path_view,
            selection_view=selection_view,
            callbacks=cls._mouse_root_callbacks(ports.selection),
        )
        folder_dialog = DesktopFolderDialog(parent=None)
        path_dialog_presenter = DesktopPathDialogPresenter(
            folder_dialog=folder_dialog,
            callbacks=DesktopPathDialogCallbacks(
                active_mouse_root=app.queries.active_mouse_root_path,
                set_mouse_root=mouse_root_presenter.set_mouse_root,
                active_output_root=app.queries.active_output_root,
                set_save_root=output_path_presenter.set_save_root,
            ),
        )
        output_folder_prompt = DesktopOutputFolderPrompt(
            parent=parent,
            callbacks=OutputFolderPromptCallbacks(
                derive_output_directory_from_save_root=(
                    output_path_presenter.derive_output_directory_from_save_root
                ),
                has_output_directory=app.queries.has_output_directory,
                select_output_folder=path_dialog_presenter.select_output_root,
            ),
        )
        load_workflow_presenter = DesktopLoadWorkflowPresenter(
            can_load_data=app.commands.can_load_data,
            load_heavy_data=load_data_presenter.load_heavy_data,
            output_folder_prompt=output_folder_prompt,
        )
        save_workflow_presenter = DesktopSaveWorkflowPresenter(
            commands=app.commands,
            callbacks=cls._save_workflow_callbacks(
                ports.save_workflow,
                output_folder_prompt,
                load_workflow_presenter,
            ),
        )
        previous_alignment_load_presenter = DesktopPreviousAlignmentLoadPresenter(
            commands=app.commands,
            callbacks=cls._previous_alignment_load_callbacks(
                ports.previous_alignment_load,
                folder_dialog,
            ),
        )
        interaction_presenter = cls._interaction_presenter(
            ports.interaction,
            app=app,
            displays=displays,
            reference_line_presenter=reference_line_presenter,
        )
        plot_exporter = cls._plot_exporter(
            ports.export,
            displays=displays,
        )
        return cls(
            app=app,
            displays=displays,
            alignment_presenter=alignment_presenter,
            shank_presenter=shank_presenter,
            load_data_presenter=load_data_presenter,
            probe_selection_presenter=probe_selection_presenter,
            session_selection_presenter=session_selection_presenter,
            mouse_root_presenter=mouse_root_presenter,
            output_path_presenter=output_path_presenter,
            path_dialog_presenter=path_dialog_presenter,
            load_workflow_presenter=load_workflow_presenter,
            output_folder_prompt=output_folder_prompt,
            folder_dialog=folder_dialog,
            save_workflow_presenter=save_workflow_presenter,
            previous_alignment_load_presenter=previous_alignment_load_presenter,
            plot_exporter=plot_exporter,
            interaction_presenter=interaction_presenter,
            lifecycle_presenter=lifecycle_presenter,
            reference_line_presenter=reference_line_presenter,
            histology_refresh_presenter=histology_refresh_presenter,
        )

    @staticmethod
    def _alignment_render_callbacks(
        ports: DesktopAlignmentRenderPorts,
        displays: DesktopDisplays,
    ) -> DesktopAlignmentRenderCallbacks:
        """Build callbacks for alignment edit rendering."""
        return DesktopAlignmentRenderCallbacks(
            restore_lin_fit=ports.restore_lin_fit,
            clear_reference_lines=displays.reference_lines.clear,
            capture_depth_plot_y_ranges=ports.capture_depth_plot_y_ranges,
            restore_depth_plot_y_ranges=ports.restore_depth_plot_y_ranges,
            reattach_reference_lines=displays.reference_lines.reattach,
            render_histology_alignment=displays.histology.render_alignment_edit,
            plot_channels=displays.slice.plot_channels,
            refresh_perpendicular_histology=(
                displays.slice.refresh_perpendicular_histology
            ),
            update_reference_lines_to_alignment=(
                displays.reference_lines.sync_track_to_feature
            ),
            create_reference_lines_for_previous_alignment=(
                ports.create_reference_lines_for_previous_alignment
            ),
            set_default_feature_y_range=ports.set_default_feature_y_range,
            update_status=ports.update_status,
        )

    @staticmethod
    def _shank_render_callbacks(
        ports: DesktopShankRenderPorts,
        displays: DesktopDisplays,
        histology_refresh_presenter: DesktopHistologyRefreshPresenter,
    ) -> DesktopShankRenderCallbacks:
        """Build callbacks for shank selection rendering."""
        return DesktopShankRenderCallbacks(
            capture_plot_selection=ports.capture_plot_selection,
            clear_reference_lines=displays.reference_lines.clear,
            render_alignment_choices=ports.render_alignment_choices,
            apply_plot_data_state=ports.apply_plot_data_state,
            raw_image_payloads=ports.raw_image_payloads,
            render_plot_menus=ports.render_plot_menus,
            render_ephys_plots=displays.ephys.render_shank_ephys_plots,
            render_histology_plots=(
                histology_refresh_presenter.render_loaded_shank_histology
            ),
            restore_slice_selection=displays.slice.restore_selection,
            configure_view=ports.configure_view,
            offline=ports.offline,
        )

    @staticmethod
    def _save_workflow_callbacks(
        ports: DesktopSaveWorkflowPorts,
        output_folder_prompt: DesktopOutputFolderPrompt,
        load_workflow_presenter: DesktopLoadWorkflowPresenter,
    ) -> DesktopSaveWorkflowCallbacks:
        """Build callbacks for save/QC workflows."""
        return DesktopSaveWorkflowCallbacks(
            ensure_output_directory=output_folder_prompt.ensure_for_save,
            log_requirement=load_workflow_presenter.log_requirement,
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

    @staticmethod
    def _previous_alignment_load_callbacks(
        ports: DesktopPreviousAlignmentLoadPorts,
        folder_dialog: DesktopFolderDialog,
    ) -> PreviousAlignmentLoadCallbacks:
        """Build callbacks for previous-alignment loading."""
        return PreviousAlignmentLoadCallbacks(
            select_folder=lambda: folder_dialog.select_existing_directory_text(
                "Load Existing Alignments",
            ),
            use_docdb=ports.use_docdb,
            set_reload_folder_text=ports.set_reload_folder_text,
            render_alignment_choices=ports.render_alignment_choices,
            select_alignment=ports.select_alignment,
            busy_context=ports.busy_context,
            reload_button=ports.reload_button,
        )

    @staticmethod
    def _plot_exporter(
        ports: DesktopExportPorts,
        *,
        displays: DesktopDisplays,
    ) -> DesktopPlotExporter:
        """Build the desktop plot exporter cluster."""
        ephys_exporter = DesktopEphysPlotExporter(
            presenter=displays.ephys.plot_presenter,
            panel=displays.ephys.panel,
            layout=EphysExportLayout(
                graphics_layout=ports.ephys_graphics_layout,
                data_area=ports.ephys_data_area,
            ),
            callbacks=DesktopWorkbench._ephys_export_callbacks(
                ports,
                displays,
            ),
        )
        return DesktopPlotExporter(
            ephys_exporter=ephys_exporter,
            slice_handles=SliceExportHandles(
                slice_display=displays.slice,
                slice_plot=ports.slice_plot,
            ),
            slice_style=SliceExportStyle(
                trajectory_pen=ports.slice_trajectory_pen,
            ),
            histology_handles=HistologyExportHandles(
                histology_display=displays.histology,
            ),
            callbacks=DesktopWorkbench._plot_export_callbacks(ports),
            add_lines_points=displays.reference_lines.add_to_plots,
        )

    @staticmethod
    def _interaction_presenter(
        ports: DesktopInteractionPorts,
        *,
        app: Any,
        displays: DesktopDisplays,
        reference_line_presenter: DesktopReferenceLinePresenter,
    ) -> DesktopInteractionPresenter:
        """Build the desktop interaction presenter."""
        return DesktopInteractionPresenter(
            app=app,
            popup_manager=ports.popup_manager,
            ephys_panel=displays.ephys.panel,
            histology_display=displays.histology,
            reference_line_display=displays.reference_lines,
            region_lookup_service=ports.region_lookup_service,
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
                    reference_line_presenter.capture_pending_reference_lines
                ),
            ),
        )

    @staticmethod
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

    @staticmethod
    def _ephys_export_callbacks(
        ports: DesktopExportPorts,
        displays: DesktopDisplays,
    ) -> EphysExportCallbacks:
        """Build callbacks for ephys plot export layout changes."""
        return EphysExportCallbacks(
            reset_axis=ports.reset_axis,
            set_view=ports.set_view,
            set_axis=ports.set_axis,
            set_font=ports.set_font,
            add_lines_points=displays.reference_lines.add_to_plots,
            sizes=lambda: EphysExportSizes(*ports.ephys_sizes()),
        )

    @staticmethod
    def _plot_export_callbacks(
        ports: DesktopExportPorts,
    ) -> DesktopPlotExportCallbacks:
        """Build callbacks for non-ephys plot export steps."""
        return DesktopPlotExportCallbacks(
            set_axis=ports.set_axis,
            set_font=ports.set_font,
            slice_geometry=lambda: SliceExportGeometry(*ports.slice_geometry()),
        )

    @staticmethod
    def _load_data_callbacks(
        callbacks: DesktopSelectionWorkflowCallbacks,
        output_path_presenter: DesktopOutputPathPresenter,
        shank_presenter: DesktopShankPresenter,
        lifecycle_presenter: DesktopLifecyclePresenter,
        reference_line_presenter: DesktopReferenceLinePresenter,
    ) -> DesktopLoadDataCallbacks:
        """Build callbacks for cached/fresh data loading."""
        return DesktopLoadDataCallbacks(
            capture_pending_reference_lines=(
                reference_line_presenter.capture_pending_reference_lines
            ),
            detach_active_stream=lifecycle_presenter.detach_active_stream,
            prepare_for_fresh_stream_load=(
                lifecycle_presenter.prepare_for_fresh_stream_load
            ),
            select_shank_for_view=callbacks.select_shank_for_view,
            display_output_directory=output_path_presenter.display_output_directory,
            render_loaded_shank=lambda shank_idx, preserve: (
                shank_presenter.render_loaded_shank(
                    shank_idx=shank_idx,
                    preserve_plot_selection=preserve,
                )
            ),
            clear_empty_state=callbacks.clear_empty_state,
            set_histology_available=callbacks.set_histology_available,
            busy_context=callbacks.busy_context,
        )

    @staticmethod
    def _probe_selection_callbacks(
        callbacks: DesktopSelectionWorkflowCallbacks,
        output_path_presenter: DesktopOutputPathPresenter,
        load_data_presenter: DesktopLoadDataPresenter,
        lifecycle_presenter: DesktopLifecyclePresenter,
        reference_line_presenter: DesktopReferenceLinePresenter,
    ) -> DesktopProbeSelectionCallbacks:
        """Build callbacks for probe selection."""
        return DesktopProbeSelectionCallbacks(
            mouse_root_loaded=callbacks.mouse_root_loaded,
            active_shank_idx=callbacks.active_shank_idx,
            capture_pending_reference_lines=(
                reference_line_presenter.capture_pending_reference_lines
            ),
            detach_active_stream=lifecycle_presenter.detach_active_stream,
            present_cached_probe_selection=(
                lambda session, probe, shank: (
                    load_data_presenter.present_cached_probe_selection(
                        session_name=session,
                        probe_name=probe,
                        target_shank=shank,
                    )
                )
            ),
            show_empty_state=lifecycle_presenter.show_empty_state,
            busy_context=callbacks.busy_context,
            select_shank_for_view=callbacks.select_shank_for_view,
            display_output_directory=output_path_presenter.display_output_directory,
        )

    @staticmethod
    def _session_selection_callbacks(
        callbacks: DesktopSelectionWorkflowCallbacks,
        lifecycle_presenter: DesktopLifecyclePresenter,
        reference_line_presenter: DesktopReferenceLinePresenter,
    ) -> DesktopSessionSelectionCallbacks:
        """Build callbacks for session selection."""
        return DesktopSessionSelectionCallbacks(
            mouse_root_loaded=callbacks.mouse_root_loaded,
            capture_pending_reference_lines=(
                reference_line_presenter.capture_pending_reference_lines
            ),
            evict_stream_cache=lifecycle_presenter.evict_stream_cache,
            show_empty_state=lifecycle_presenter.show_empty_state,
            select_first_probe=callbacks.select_first_probe,
        )

    @staticmethod
    def _mouse_root_callbacks(
        callbacks: DesktopSelectionWorkflowCallbacks,
    ) -> DesktopMouseRootCallbacks:
        """Build callbacks for mouse-root loading."""
        return DesktopMouseRootCallbacks(
            clear_histology_context=callbacks.clear_histology_context,
            busy_context=callbacks.busy_context,
            select_first_session=callbacks.select_first_session,
        )

    def connect_events(self) -> list[EventSubscription]:
        """Subscribe desktop presenters to semantic app events."""
        if self._event_subscriptions:
            return list(self._event_subscriptions)
        self._event_subscriptions.extend(
            self.alignment_presenter.connect_alignment_events()
        )
        self._event_subscriptions.extend(self.shank_presenter.connect_shank_events())
        return list(self._event_subscriptions)

    def disconnect_events(self) -> None:
        """Disconnect desktop event subscriptions."""
        for subscription in self._event_subscriptions:
            subscription.disconnect()
        self._event_subscriptions.clear()

    def render_loaded_shank(
        self,
        *,
        shank_idx: int,
        preserve_plot_selection: bool | None = None,
    ) -> None:
        """Render the loaded desktop view for one active shank."""
        self.shank_presenter.render_loaded_shank(
            shank_idx=shank_idx,
            preserve_plot_selection=preserve_plot_selection,
        )

    def render_active_aligned_histology(
        self,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> bool:
        """Render the active aligned histology panel."""
        return self.displays.histology.render_active_aligned(fig, movable=movable)

    def render_active_reference_histology(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render the active reference histology panel."""
        return self.displays.histology.render_active_reference(fig, movable=movable)

    def render_active_scale_factor(self) -> bool:
        """Render the active scale-factor panel."""
        return self.displays.histology.render_active_scale_factor()

    def render_active_fit(self) -> bool:
        """Render the active feature/track fit panel."""
        return self.displays.histology.render_active_fit()

    def render_active_histology_panels(self) -> bool:
        """Render reference histology, aligned histology, scale, and fit panels."""
        return self.displays.histology.render_active_panels()

    def render_loaded_shank_histology(self, shank_idx: int | None = None) -> bool:
        """Render loaded-shank histology, perpendicular slice, and line overlays."""
        return self.histology_refresh_presenter.render_loaded_shank_histology(
            shank_idx
        )

    def load_heavy_data(self) -> bool:
        """Load or activate the selected stream/shank for desktop display."""
        return self.load_data_presenter.load_heavy_data()

    def set_mouse_root(self, mouse_root: Any) -> bool:
        """Load a mouse-root datapackage through the desktop presenter."""
        return self.mouse_root_presenter.set_mouse_root(mouse_root)

    def mouse_root_edited(self) -> bool:
        """Handle direct text edits to the mouse-root line edit."""
        return self.mouse_root_presenter.mouse_root_edited()

    def session_selected(self) -> bool:
        """Select the current recording/session from the desktop widgets."""
        return self.session_selection_presenter.session_selected()

    def probe_selected(self) -> bool:
        """Select the current probe from the desktop widgets."""
        return self.probe_selection_presenter.probe_selected()

    def load_data_button_pressed(self) -> bool:
        """Run desktop load workflow policy and load data when allowed."""
        return self.load_workflow_presenter.load_data_button_pressed()

    def ensure_output_directory_for_save(self, requirement: Any | None = None) -> bool:
        """Require a save location before writing alignment outputs."""
        return self.output_folder_prompt.ensure_for_save(requirement)

    def set_save_root(self, save_root: Path) -> bool:
        """Set the save-root directory. Per-probe output lands under it."""
        return self.output_path_presenter.set_save_root(save_root)

    def select_mouse_root(self) -> bool:
        """Prompt for a mouse-root directory."""
        return self.path_dialog_presenter.select_mouse_root()

    def select_output_root(self) -> bool:
        """Prompt for a save-root directory."""
        return self.path_dialog_presenter.select_output_root()

    def output_folder_edited(self) -> bool:
        """Handle direct edits to the output-folder text field."""
        return self.output_path_presenter.output_folder_edited()

    def log_load_requirement(self, requirement: Any) -> None:
        """Log a load workflow requirement that has no desktop prompt action."""
        self.load_workflow_presenter.log_requirement(requirement)

    def select_existing_directory_text(self, title: str) -> str:
        """Prompt for an existing directory and return Qt-style text."""
        return self.folder_dialog.select_existing_directory_text(title)

    def load_existing_alignments(self) -> bool:
        """Prompt for and load previous alignments."""
        return self.previous_alignment_load_presenter.load_existing_alignments()

    def save_alignment_outputs(self) -> bool:
        """Save visited alignment outputs."""
        return self.save_workflow_presenter.save_alignment_outputs()

    def display_qc_options(self) -> bool:
        """Display alignment QC choices."""
        return self.save_workflow_presenter.display_qc_options()

    def qc_button_clicked(self) -> bool:
        """Handle the QC save button."""
        return self.save_workflow_presenter.qc_button_clicked()

    def export_plots(self, output_dir: Path, *, sess_info: str = "") -> None:
        """Export all desktop plot panels for the active shank."""
        self.plot_exporter.export(output_dir, sess_info=sess_info)

    def display_session_notes(self) -> None:
        """Show session notes for the active stream."""
        self.interaction_presenter.display_session_notes()

    def popup_closed(self, popup: Any) -> None:
        """Forget a closed cluster popup."""
        self.interaction_presenter.popup_closed(popup)

    def popup_moved(self) -> None:
        """Bring the main window back to front after popup movement."""
        self.interaction_presenter.popup_moved()

    def close_popups(self) -> None:
        """Close cluster detail popups."""
        self.interaction_presenter.close_popups()

    def minimise_popups(self) -> None:
        """Toggle cluster detail popups between minimized and normal."""
        self.interaction_presenter.minimise_popups()

    def cluster_clicked(self, item: Any, point: Any) -> Any | None:
        """Open cluster detail popup for a clicked ephys cluster point."""
        return self.interaction_presenter.cluster_clicked(item, point)

    def describe_labels_pressed(self) -> bool:
        """Show region information for the selected histology label."""
        return self.interaction_presenter.describe_labels_pressed()

    def label_closed(self, popup: Any) -> None:
        """Hide the label popup without forgetting reusable widgets."""
        self.interaction_presenter.label_closed(popup)

    def label_moved(self) -> None:
        """Bring the main window back to front after label popup movement."""
        self.interaction_presenter.label_moved()

    def label_pressed(self, item: Any) -> None:
        """Render region information for a clicked structure tree item."""
        self.interaction_presenter.label_pressed(item)

    def on_mouse_double_clicked(self, event: Any) -> bool:
        """Add a reference line from a double-clicked feature plot position."""
        return self.interaction_presenter.on_mouse_double_clicked(event)

    def on_mouse_hover(self, items: list[Any]) -> None:
        """Dispatch hover interactions to reference-line and histology views."""
        self.interaction_presenter.on_mouse_hover(items)
