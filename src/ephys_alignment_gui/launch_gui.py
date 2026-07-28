import gc
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

if platform.system() == "Darwin":
    if platform.release().split(".")[0] >= "20":
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

from random import randrange

import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QApplication

import ephys_alignment_gui.ephys_gui_setup as ephys_gui
from ephys_alignment_gui.controller import (
    AlignmentEditApplied,
    PreviousAlignmentSelected,
    ShankSelected,
)
from ephys_alignment_gui.alignment_read_models import (
    ActiveShankPlotDataState,
    ActiveSliceMenuState,
)
from ephys_alignment_gui.desktop_alignment_presenter import (
    DesktopAlignmentPresenter,
    DesktopAlignmentRenderCallbacks,
)
from ephys_alignment_gui.desktop_ephys_plot_exporter import (
    DesktopEphysPlotExporter,
    EphysExportCallbacks,
    EphysExportLayout,
    EphysExportSizes,
)
from ephys_alignment_gui.desktop_ephys_plot_presenter import (
    DesktopEphysPlotPresenter,
    EphysPlotRenderCallbacks,
)
from ephys_alignment_gui.desktop_ephys_panel_layout import (
    DesktopEphysPanelLayout,
    EphysPanelLayoutCallbacks,
    EphysPanelLayoutSizes,
)
from ephys_alignment_gui.desktop_ephys_panel_view import (
    DesktopEphysPanelView,
    EphysPanelPlots,
    EphysPanelStyle,
)
from ephys_alignment_gui.desktop_folder_dialog import DesktopFolderDialog
from ephys_alignment_gui.desktop_interaction_presenter import (
    DesktopInteractionCallbacks,
    DesktopInteractionPresenter,
    DesktopInteractionWidgets,
)
from ephys_alignment_gui.desktop_load_workflow_presenter import (
    DesktopLoadWorkflowPresenter,
    DesktopOutputFolderPrompt,
    OutputFolderPromptCallbacks,
)
from ephys_alignment_gui.desktop_load_data_presenter import (
    DesktopLoadDataCallbacks,
    DesktopLoadDataPresenter,
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
from ephys_alignment_gui.desktop_path_view import DesktopPathView
from ephys_alignment_gui.desktop_previous_alignment_load_presenter import (
    DesktopPreviousAlignmentLoadPresenter,
    PreviousAlignmentLoadCallbacks,
)
from ephys_alignment_gui.desktop_probe_selection_presenter import (
    DesktopProbeSelectionCallbacks,
    DesktopProbeSelectionPresenter,
)
from ephys_alignment_gui.desktop_selection_view import DesktopSelectionView
from ephys_alignment_gui.desktop_save_workflow_presenter import (
    DesktopSaveWorkflowCallbacks,
    DesktopSaveWorkflowPresenter,
)
from ephys_alignment_gui.desktop_session_selection_presenter import (
    DesktopSessionSelectionCallbacks,
    DesktopSessionSelectionPresenter,
)
from ephys_alignment_gui.desktop_plot_exporter import (
    DesktopPlotExportCallbacks,
    DesktopPlotExporter,
    HistologyExportHandles,
    SliceExportGeometry,
    SliceExportHandles,
    SliceExportStyle,
)
from ephys_alignment_gui.desktop_popup_manager import DesktopPopupManager
from ephys_alignment_gui.desktop_shank_presenter import (
    DesktopShankPresenter,
    DesktopShankRenderCallbacks,
    DesktopShankSelectionState,
)
from ephys_alignment_gui.event_bus import EventSubscription
from ephys_alignment_gui.histology_panel_presenter import (
    FitPanelItems,
    HistologyPanelAxes,
    HistologyPanelPlots,
    HistologyPanelPresenter,
    HistologyPanelStyle,
)
from ephys_alignment_gui.reference_line_layer import (
    ReferenceLineLayer,
    ReferenceLinePlots,
)
from ephys_alignment_gui.settings import (
    OUTPUT_ROOT_ENV_VAR,
    output_root_from_environment,
)
from ephys_alignment_gui.slice_display_policy import SliceSelection
from ephys_alignment_gui.slice_panel_presenter import (
    SlicePanelPlots,
    SlicePanelPresenter,
    SlicePanelStyle,
)
from ephys_alignment_gui.thread_worker import Worker
from ephys_alignment_gui.view_limits import default_feature_y_limits
from ephys_alignment_gui.workflow import (
    Failed,
    Requirement,
)
from ephys_alignment_gui.workspace import AlignmentWorkspace

logger = logging.getLogger(__name__)

ANTS_DIMENSION = 3


class BusyContext:
    """Context manager for long-running operations with visual feedback.

    Provides busy cursor, status messages, and UI element disabling with
    automatic cleanup and error handling.

    Example usage:
        # Simple usage - just busy cursor
        with BusyContext(self):
            do_work()

        # With status message and success confirmation
        with BusyContext(self, "Loading data...", "Data loaded successfully"):
            do_work()

        # Disable widgets during operation
        with BusyContext(self, "Loading...",
                         disable_widgets=[self.button1, self.button2]):
            do_work()

        # Multi-stage operation with progress updates
        with BusyContext(self, "Loading...", "All data loaded") as ctx:
            ctx.update_message("Loading ephys data...")
            load_ephys()
            ctx.update_message("Loading atlas...")
            load_atlas()
    """

    def __init__(
        self,
        window,
        message: str | None = None,
        success_message: str | None = None,
        error_message: str | None = None,
        disable_widgets: list | None = None,
        success_timeout_ms=3000,
        error_timeout_ms=5000,
    ):
        """
        Initialize context manager for busy state.

        :param window: MainWindow instance (for statusBar access)
        :param message: Status message to show while running
        :param success_message: Message to show on success (None = no message)
        :param disable_widgets: Widget or list of widgets to disable during operation
        :param success_timeout_ms: Timeout for success message in ms (0 = permanent)
        """
        self.window = window
        self.message = message
        self.success_message = success_message
        self.error_message = error_message
        self.success_timeout_ms = success_timeout_ms
        self.error_timeout_ms = error_timeout_ms

        # Normalize to list
        if disable_widgets is None:
            self.disable_widgets = []
        elif not isinstance(disable_widgets, list):
            self.disable_widgets = [disable_widgets]
        else:
            self.disable_widgets = disable_widgets

        self.widget_states = {}

    def __enter__(self):
        """Enter busy state: set cursor, show message, disable widgets."""
        # Set busy cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Show status message (no processEvents to avoid reentrancy)
        if self.message:
            self.window.statusBar().showMessage(self.message)

        # Disable widgets and save their states
        for widget in self.disable_widgets:
            self.widget_states[widget] = widget.isEnabled()
            widget.setEnabled(False)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit busy state: restore cursor, widgets, and handle status messages."""
        # Restore cursor
        QApplication.restoreOverrideCursor()

        # Restore widget states
        for widget, was_enabled in self.widget_states.items():
            widget.setEnabled(was_enabled)

        # Handle status message based on outcome
        if exc_type is not None:
            # Error occurred - show error message
            if self.error_message is None:
                error_msg = f"Error: {str(exc_val)}"
            else:
                error_msg = self.error_message
            self.window.statusBar().showMessage(error_msg, self.error_timeout_ms)
        elif self.success_message:
            # Success - show success message
            self.window.statusBar().showMessage(
                self.success_message, self.success_timeout_ms
            )
        else:
            # Clear status
            self.window.statusBar().clearMessage()

        # Don't suppress exceptions
        return False

    def update_message(self, new_message: str):
        """
        Update status message during a long operation.

        Use sparingly - calls processEvents() which can cause reentrancy issues.

        :param new_message: New message to display
        """
        if new_message:
            self.window.statusBar().showMessage(new_message)
            QApplication.processEvents()


class MainWindow(QtWidgets.QMainWindow, ephys_gui.Setup):
    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]

    @staticmethod
    def _get_or_create(title="Electrophysiology Atlas", **kwargs):
        av = next(
            filter(
                lambda e: e.isVisible() and e.windowTitle() == title,
                MainWindow._instances(),
            ),
            None,
        )
        if av is None:
            av = MainWindow(**kwargs)
            av.setWindowTitle(title)
        return av

    def __init__(
        self,
        offline=True,
        probe_id=None,
        one=None,
        histology=True,
        spike_collection=None,
        remote=False,
    ) -> None:
        super().__init__()
        requested_offline = self._normalize_offline_flag(offline)
        if not requested_offline:
            logger.warning(
                "ONE/Alyx online mode is not supported in this branch; "
                "using preprocessed datapackage mode. DocDB remains available "
                "through the DocDB checkbox."
            )
        offline = True

        self.workspace = AlignmentWorkspace()
        self.app = self.workspace.app
        self.runtime = self.workspace.runtime
        self.document = self.workspace.document
        self.display_state = self.workspace.display_state
        self.data_context = self.workspace.data_context
        self.histology_context = self.workspace.histology_context
        self.slice_service = self.workspace.slice_service
        self.probe_track_service = self.workspace.probe_track_service
        self.region_lookup_service = self.workspace.region_lookup_service
        self.controller = self.workspace.controller
        self.alignment_derived_data_service = (
            self.workspace.alignment_derived_data_service
        )
        self.desktop_alignment_presenter = DesktopAlignmentPresenter(self.app.events)
        self.desktop_shank_presenter = DesktopShankPresenter(self.app)
        self.plot_data_factory = self.workspace.plot_data_factory
        self.ephys_plot_presenter = DesktopEphysPlotPresenter(
            app=self.app,
            callbacks=EphysPlotRenderCallbacks(
                raw_image_payloads=lambda: self.raw_image_payloads,
                render_image=lambda data: self.ephys_panel.render_image(data),
                render_scatter=lambda data: self.ephys_panel.render_scatter(data),
                render_line=lambda data: self.ephys_panel.render_line(data),
                render_probe=lambda data, bounds: self.ephys_panel.render_probe(
                    data,
                    bounds=bounds,
                ),
            ),
        )
        self.popup_manager = DesktopPopupManager()
        self.init_variables()
        self._event_subscriptions: list[EventSubscription] = []
        self.offline: bool = offline
        self.init_layout(self, offline=offline)
        self.selection_view = DesktopSelectionView(
            session_model=self.session_list,
            session_combobox=self.session_combobox,
            probe_model=self.probe_list,
            probe_combobox=self.probe_combobox,
            shank_model=self.shank_list,
            shank_combobox=self.shank_combobox,
            load_data_button=self.load_data_button,
        )
        self.path_view = DesktopPathView(
            mouse_root_button=self.mouse_root_button,
            mouse_root_line=self.mouse_root_line,
            output_folder_line=self.output_folder_line,
        )
        self.ephys_panel = DesktopEphysPanelView(
            plots=EphysPanelPlots(
                image=self.fig_img,
                image_colorbar=self.fig_img_cb,
                line=self.fig_line,
                probe=self.fig_probe,
                probe_colorbar=self.fig_probe_cb,
            ),
            style=EphysPanelStyle(line_pen=self.kpen_solid),
            set_axis=self.set_axis,
            cluster_clicked=lambda *args: self.interaction_presenter.cluster_clicked(
                *args
            ),
        )
        self.ephys_panel_layout = DesktopEphysPanelLayout(
            panel=self.ephys_panel,
            graphics_layout=self.fig_data_layout,
            callbacks=EphysPanelLayoutCallbacks(
                set_axis=self.set_axis,
                reset_axis=self.reset_axis_button_pressed,
            ),
        )
        self.ephys_plot_exporter = DesktopEphysPlotExporter(
            presenter=self.ephys_plot_presenter,
            panel=self.ephys_panel,
            layout=EphysExportLayout(
                graphics_layout=self.fig_data_layout,
                data_area=self.fig_data_area,
            ),
            callbacks=EphysExportCallbacks(
                reset_axis=self.reset_axis_button_pressed,
                set_view=self.set_view,
                set_axis=self.set_axis,
                set_font=self.set_font,
                add_lines_points=self.reference_lines.add_to_plots,
                sizes=lambda: EphysExportSizes(
                    probe_width=self.fig_probe_width,
                    axis_width=self.fig_ax_width,
                ),
            ),
        )
        self.reference_lines = ReferenceLineLayer(
            plots=ReferenceLinePlots(
                histology=self.fig_hist,
                image=self.fig_img,
                line=self.fig_line,
                probe=self.fig_probe,
                perpendicular=self.fig_hist_perp,
                fit=self.fig_fit,
            ),
            style_factory=self.create_line_style,
            on_lines_changed=self._capture_pending_reference_lines,
        )
        self.slice_panel = SlicePanelPresenter(
            app=self.app,
            plots=SlicePanelPlots(
                coronal=self.fig_slice,
                coronal_layout=self.fig_slice_layout,
                histogram_alt=self.fig_slice_hist_alt,
                perpendicular=self.fig_hist_perp,
            ),
            style=SlicePanelStyle(
                dotted_pen=self.kpen_dot,
                solid_pen=self.kpen_solid,
                reference_line_pen=self.reference_line_kpen,
            ),
            histology_exists=lambda: getattr(self, "histology_exists", False),
            action_group_provider=lambda: getattr(self, "slice_options_group", None),
            slice_item=self.slice_item,
        )
        self.histology_panel = HistologyPanelPresenter(
            plots=HistologyPanelPlots(
                aligned=self.fig_hist,
                reference=self.fig_hist_ref,
                scale=self.fig_scale,
                scale_colorbar=self.fig_scale_cb,
            ),
            axes=HistologyPanelAxes(
                aligned=self.ax_hist,
                reference=self.ax_hist_ref,
            ),
            style=HistologyPanelStyle(dotted_pen=self.kpen_dot),
            set_axis=self.set_axis,
            padding_provider=lambda: self.pad,
            fit_items=FitPanelItems(
                fit_curve=self.fit_plot,
                fit_scatter=self.fit_scatter,
                linear_fit_curve=self.fit_plot_lin,
            ),
        )
        self._init_interaction_presenter()
        self._init_plot_exporter()
        self.desktop_alignment_presenter.configure(
            queries=self.app.queries,
            callbacks=self._desktop_alignment_render_callbacks(),
        )
        self.desktop_shank_presenter.configure(
            callbacks=self._desktop_shank_render_callbacks(),
        )
        self._connect_alignment_changed_handlers()
        self._connect_shank_changed_handlers()
        self._init_output_path_presenter()
        self._init_load_data_presenter()
        self._init_probe_selection_presenter()
        self._init_session_selection_presenter()
        self._init_mouse_root_presenter()
        self._init_path_dialog_presenter()
        self._init_load_workflow_presenter()
        self._init_save_workflow_presenter()
        self._init_previous_alignment_load_presenter()
        self._set_default_output_root_from_environment()

        self.configure: bool = True
        self.histology_exists: bool = True
        self.use_docdb: bool = True
        self._empty_state_item: Any = None

        self.allen = self.region_lookup_service.load_allen_csv()
        self.init_region_lookup(self.allen)

    def _init_load_workflow_presenter(self) -> None:
        """Wire desktop load workflow prompts and command gating."""
        self.output_folder_prompt = DesktopOutputFolderPrompt(
            parent=self,
            callbacks=OutputFolderPromptCallbacks(
                derive_output_directory_from_save_root=(
                    self.output_path_presenter.derive_output_directory_from_save_root
                ),
                has_output_directory=lambda: self.document.output_directory is not None,
                select_output_folder=self.on_output_folder_selected,
            ),
        )
        self.load_workflow_presenter = DesktopLoadWorkflowPresenter(
            can_load_data=self.controller.can_load_data,
            load_heavy_data=self.load_data_presenter.load_heavy_data,
            output_folder_prompt=self.output_folder_prompt,
        )

    def _init_save_workflow_presenter(self) -> None:
        """Wire desktop behavior for save and QC workflows."""
        self.save_workflow_presenter = DesktopSaveWorkflowPresenter(
            commands=self.app.commands,
            callbacks=DesktopSaveWorkflowCallbacks(
                ensure_output_directory=self.output_folder_prompt.ensure_for_save,
                log_requirement=self.load_workflow_presenter.log_requirement,
                use_docdb=lambda: self.use_docdb,
                render_alignment_choices=lambda choices: self.populate_lists(
                    choices,
                    self.align_list,
                    self.align_combobox,
                ),
                busy_context=lambda *args, **kwargs: BusyContext(
                    self,
                    *args,
                    **kwargs,
                ),
                complete_button=lambda: self.complete_button,
                histology_available=lambda: self.histology_exists,
                open_qc_dialog=self.qc_dialog.open,
                ephys_qc=self.ephys_qc.currentText,
                selected_qc_descriptions=self._selected_qc_descriptions,
                warning=lambda title, message: QtWidgets.QMessageBox.warning(
                    self,
                    title,
                    message,
                ),
            ),
        )

    def _init_interaction_presenter(self) -> None:
        """Wire desktop popup and mouse interaction behavior."""
        self.interaction_presenter = DesktopInteractionPresenter(
            app=self.app,
            popup_manager=self.popup_manager,
            ephys_panel=self.ephys_panel,
            histology_panel=self.histology_panel,
            reference_lines=self.reference_lines,
            region_lookup_service=self.region_lookup_service,
            widgets=DesktopInteractionWidgets(
                struct_list=self.struct_list,
                struct_view=self.struct_view,
                struct_description=self.struct_description,
                scale_plot=self.fig_scale,
                histology_plot=self.fig_hist,
                histology_reference_plot=self.fig_hist_ref,
                scale_axis=self.fig_scale_ax,
                bar_colour=self.bar_colour,
                line_pen=self.kpen_solid,
            ),
            callbacks=DesktopInteractionCallbacks(
                histology_available=lambda: self.histology_exists,
                activate_window=self.activateWindow,
                set_axis=self.set_axis,
                capture_pending_reference_lines=self._capture_pending_reference_lines,
            ),
        )

    def _init_output_path_presenter(self) -> None:
        """Wire desktop behavior for output path rendering."""
        self.output_path_presenter = DesktopOutputPathPresenter(
            commands=self.app.commands,
            path_view=self.path_view,
        )

    def _init_load_data_presenter(self) -> None:
        """Wire desktop behavior for cached/fresh data loading."""
        self.load_data_presenter = DesktopLoadDataPresenter(
            app=self.app,
            selection_view=self.selection_view,
            callbacks=DesktopLoadDataCallbacks(
                capture_pending_reference_lines=self._capture_pending_reference_lines,
                stash_and_detach_current=self._stash_and_detach_current,
                teardown_session=self._teardown_session,
                init_session_variables=self.init_session_variables,
                select_shank_for_view=lambda shank_idx, source: (
                    self._select_shank_for_view(shank_idx, source=source)
                ),
                display_output_directory=(
                    self.output_path_presenter.display_output_directory
                ),
                setup_session_view=lambda preserve, shank_idx: self.setup_session_view(
                    preserve_plot_selection=preserve,
                    shank_idx=shank_idx,
                ),
                clear_empty_state=self._clear_empty_state,
                set_histology_available=self._set_histology_available,
                busy_context=lambda *args, **kwargs: BusyContext(
                    self,
                    *args,
                    **kwargs,
                ),
            ),
        )

    def _init_probe_selection_presenter(self) -> None:
        """Wire desktop behavior for probe selection."""
        self.probe_selection_presenter = DesktopProbeSelectionPresenter(
            commands=self.app.commands,
            selection_view=self.selection_view,
            callbacks=DesktopProbeSelectionCallbacks(
                mouse_root_loaded=lambda: self.data_context.mouse_root is not None,
                active_shank_idx=self._active_shank_idx,
                capture_pending_reference_lines=self._capture_pending_reference_lines,
                stash_and_detach_current=self._stash_and_detach_current,
                present_cached_probe_selection=lambda session, probe, shank: (
                    self.load_data_presenter.present_cached_probe_selection(
                        session_name=session,
                        probe_name=probe,
                        target_shank=shank,
                    )
                ),
                show_empty_state=self._show_empty_state,
                busy_context=lambda *args, **kwargs: BusyContext(
                    self,
                    *args,
                    **kwargs,
                ),
                init_session_variables=self.init_session_variables,
                select_shank_for_view=lambda shank_idx, source: (
                    self._select_shank_for_view(shank_idx, source=source)
                ),
                display_output_directory=(
                    self.output_path_presenter.display_output_directory
                ),
            ),
        )

    def _init_session_selection_presenter(self) -> None:
        """Wire desktop behavior for session selection."""
        self.session_selection_presenter = DesktopSessionSelectionPresenter(
            commands=self.app.commands,
            selection_view=self.selection_view,
            callbacks=DesktopSessionSelectionCallbacks(
                mouse_root_loaded=lambda: self.data_context.mouse_root is not None,
                capture_pending_reference_lines=self._capture_pending_reference_lines,
                evict_stream_cache=self._evict_stream_cache,
                show_empty_state=self._show_empty_state,
                select_first_probe=lambda: self.on_probe_combobox_activated(0),
            ),
        )

    def _init_mouse_root_presenter(self) -> None:
        """Wire desktop behavior for mouse-root loading."""
        self.mouse_root_presenter = DesktopMouseRootPresenter(
            commands=self.app.commands,
            path_view=self.path_view,
            selection_view=self.selection_view,
            callbacks=DesktopMouseRootCallbacks(
                clear_histology_context=self.histology_context.clear,
                busy_context=lambda *args, **kwargs: BusyContext(
                    self,
                    *args,
                    **kwargs,
                ),
                select_first_session=lambda: self.on_session_combobox_activated(0),
            ),
        )

    def _init_path_dialog_presenter(self) -> None:
        """Wire desktop behavior for path-selection folder dialogs."""
        self.folder_dialog = DesktopFolderDialog(parent=None)
        self.path_dialog_presenter = DesktopPathDialogPresenter(
            folder_dialog=self.folder_dialog,
            callbacks=DesktopPathDialogCallbacks(
                active_mouse_root=self._active_mouse_root_path,
                set_mouse_root=self.mouse_root_presenter.set_mouse_root,
                active_output_root=lambda: self.document.output_root,
                set_save_root=self.output_path_presenter.set_save_root,
            ),
        )

    def _init_previous_alignment_load_presenter(self) -> None:
        """Wire desktop workflow for loading previous alignments."""
        self.previous_alignment_load_presenter = DesktopPreviousAlignmentLoadPresenter(
            commands=self.app.commands,
            callbacks=PreviousAlignmentLoadCallbacks(
                select_folder=lambda: self.folder_dialog.select_existing_directory_text(
                    "Load Existing Alignments",
                ),
                use_docdb=lambda: self.use_docdb,
                set_reload_folder_text=self.reload_folder_line.setText,
                render_alignment_choices=lambda choices: self.populate_lists(
                    choices,
                    self.align_list,
                    self.align_combobox,
                ),
                select_alignment=self.on_alignment_selected,
                busy_context=lambda *args, **kwargs: BusyContext(
                    self,
                    *args,
                    **kwargs,
                ),
                reload_button=lambda: self.reload_folder_button,
            ),
        )

    def _init_plot_exporter(self) -> None:
        """Wire desktop plot export orchestration after panels are available."""
        self.plot_exporter = DesktopPlotExporter(
            ephys_exporter=self.ephys_plot_exporter,
            slice_handles=SliceExportHandles(
                action_group=self.slice_options_group,
                slice_panel=self.slice_panel,
                slice_plot=self.fig_slice,
            ),
            slice_style=SliceExportStyle(trajectory_pen=self.rpen_dot),
            histology_handles=HistologyExportHandles(
                layout=self.fig_hist_layout,
                extra_y_axis=self.fig_hist_extra_yaxis,
                aligned=self.fig_hist,
                reference=self.fig_hist_ref,
            ),
            callbacks=DesktopPlotExportCallbacks(
                set_axis=self.set_axis,
                set_font=self.set_font,
                add_lines_points=self.reference_lines.add_to_plots,
                slice_geometry=lambda: SliceExportGeometry(
                    width=self.slice_width,
                    height=self.slice_height,
                    rect=self.slice_rect,
                ),
            ),
        )

    @staticmethod
    def _normalize_offline_flag(offline: Any) -> bool:
        """Interpret legacy CLI/string offline flags."""
        if isinstance(offline, str):
            return offline.strip().lower() not in {"0", "false", "no", "off"}
        return bool(offline)

    def _show_one_unsupported(self, feature: str) -> None:
        """Report unsupported ONE/Alyx-only actions without crashing."""
        message = (
            f"{feature} requires ONE/Alyx online mode, which is not supported "
            "in this preprocessed datapackage workflow."
        )
        logger.warning(message)
        QtWidgets.QMessageBox.information(self, "Unavailable", message)

    def _set_default_output_root_from_environment(self) -> None:
        """Use an environment-provided save root as the startup default."""
        output_root = output_root_from_environment()
        if output_root is None:
            return
        if self.set_save_root(output_root):
            logger.info(
                "Default save root set from %s: %s",
                OUTPUT_ROOT_ENV_VAR,
                output_root,
            )

    def _active_mouse_root_path(self) -> Path | None:
        """Return the currently loaded mouse-root path for dialog defaults."""
        mouse_root = self.data_context.mouse_root
        if mouse_root is None:
            return None
        return mouse_root.root

    def init_variables(self) -> None:
        """
        Initialise variables
        """
        # Line styles and fonts
        self.kpen_dot = pg.mkPen(color="k", style=QtCore.Qt.DotLine, width=2)
        self.reference_line_kpen = pg.mkPen(
            color="k", style=QtCore.Qt.DotLine, width=10
        )
        self.rpen_dot = pg.mkPen(color="r", style=QtCore.Qt.DotLine, width=2)
        self.kpen_solid = pg.mkPen(color="k", style=QtCore.Qt.SolidLine, width=2)
        self.bpen_solid = pg.mkPen(color="b", style=QtCore.Qt.SolidLine, width=3)
        self.bar_colour = QtGui.QColor(160, 160, 160)

        # Padding to add to figures to make sure always same size viewbox
        self.pad = 0.05
        self.init_session_variables()

        # Guide the user before any data is loaded / after clearing.
        if hasattr(self, "fig_img"):
            self._show_empty_state()

    def init_session_variables(self) -> None:
        """Initialise variables that need to be reset for each session."""
        self.popup_manager.close_all()
        self.raw_image_payloads: dict[str, Any] = {}
        self.runtime.clear_active_stream()
        self.display_state.reset_region_annotation_source()
        self.display_state.reset_unit_filter()
        self.display_state.reset_visibility_toggles()
        self.display_state.reset_depth_view()
        self.display_state.reset_edit_settings()

    def set_axis(self, fig, ax, show=True, label=None, pen="k", ticks=True):
        """
        Show/hide and configure axis of figure
        :param fig: figure associated with axis
        :type fig: pyqtgraph PlotWidget
        :param ax: orientation of axis, must be one of 'left', 'right', 'top' or 'bottom'
        :type ax: string
        :param show: 'True' to show axis, 'False' to hide axis
        :type show: bool
        :param label: axis label
        :type label: string
        :parm pen: colour on axis
        :type pen: string
        :param ticks: 'True' to show axis ticks, 'False' to hide axis ticks
        :param ticks: bool
        :return axis: axis object
        :type axis: pyqtgraph AxisItem
        """
        if not label:
            label = ""
        if type(fig) == pg.PlotItem:
            axis = fig.getAxis(ax)
        else:
            axis = fig.plotItem.getAxis(ax)
        if show:
            axis.show()
            axis.setPen(pen)
            axis.setTextPen(pen)
            axis.setLabel(label)
            if not ticks:
                axis.setTicks([[(0, ""), (0.5, ""), (1, "")]])
        else:
            axis.hide()

        return axis

    def set_font(self, fig, ax, ptsize=8, width=None, height=None) -> None:
        if type(fig) == pg.PlotItem:
            axis = fig.getAxis(ax)
        else:
            axis = fig.plotItem.getAxis(ax)

        font = QtGui.QFont()
        font.setPointSize(ptsize)
        axis.setStyle(tickFont=font)
        labelStyle = {"font-size": f"{ptsize}pt"}
        axis.setLabel(**labelStyle)

        if width:
            axis.setWidth(width)
        if height:
            axis.setHeight(height)

    def set_lims(self, min, max) -> None:
        self.display_state.depth_view.set_probe_limits(min, max)

        [top_line.setY(max) for top_line in self.probe_top_lines]
        [tip_line.setY(min) for tip_line in self.probe_tip_lines]

    def default_feature_y_limits(self) -> tuple[float, float]:
        """Return the current default feature-depth display limits."""
        in_brain_depths_um = self.app.queries.active_in_brain_depths_um()
        depth_view = self.display_state.depth_view
        return default_feature_y_limits(
            probe_tip_um=depth_view.probe_tip_um,
            probe_top_um=depth_view.probe_top_um,
            probe_extra_um=depth_view.probe_extra_um,
            in_brain_depths_um=in_brain_depths_um,
        )

    def set_default_feature_y_range(self) -> None:
        """Apply the default feature-depth range to the linked depth plots."""
        y_min, y_max = self.default_feature_y_limits()
        self.fig_hist.setYRange(min=y_min, max=y_max, padding=self.pad)
        self.fig_hist_ref.setYRange(min=y_min, max=y_max, padding=self.pad)
        self.fig_img.setYRange(min=y_min, max=y_max, padding=self.pad)

    def _capture_depth_plot_y_ranges(self) -> dict[str, tuple[float, float]]:
        """Capture current y-ranges on the linked depth plots."""
        ranges: dict[str, tuple[float, float]] = {}
        for name in (
            "fig_img",
            "fig_line",
            "fig_probe",
            "fig_hist",
            "fig_hist_ref",
            "fig_hist_perp",
            "fig_scale",
        ):
            fig = getattr(self, name, None)
            if fig is None:
                continue
            try:
                y_min, y_max = fig.viewRange()[1]
            except (AttributeError, IndexError, TypeError):
                continue
            ranges[name] = (float(y_min), float(y_max))
        return ranges

    def _restore_depth_plot_y_ranges(
        self,
        ranges: dict[str, tuple[float, float]],
    ) -> None:
        """Restore y-ranges captured before an alignment redraw."""
        for name, (y_min, y_max) in ranges.items():
            fig = getattr(self, name, None)
            if fig is None or y_min == y_max:
                continue
            fig.setYRange(min=y_min, max=y_max, padding=0)

    def populate_lists(self, data, list_name, combobox) -> None:
        """
        Populate drop down lists with subject/session/alignment options
        :param data: list of options to add to widget
        :type data: 1D array of strings
        :param list_name: widget object to which to add data to
        :type list_name: QtGui.QStandardItemModel
        :param combobox: combobox object to which to add data to
        :type combobox: QtWidgets.QComboBox
        """
        list_name.clear()
        for dat in data:
            item = QtGui.QStandardItem(dat)
            item.setEditable(False)
            list_name.appendRow(item)

        # This makes sure the drop down menu is wide enough to showw full length of string
        min_width = combobox.fontMetrics().width(max(data, key=len))
        min_width += combobox.view().autoScrollMargin()
        min_width += combobox.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
        combobox.view().setMinimumWidth(min_width)

        # Set the default to be the first option
        combobox.setCurrentIndex(0)

    def set_view(self, view=1, configure=False) -> None:
        """
        Layout of ephys data figures, can be changed using Shift+1, Shift+2, Shift+3
        :param view: from left to right
            1: img plot, line plot, probe plot
            2: img plot, probe plot, line plot
            3: probe plot, line plot, img_plot
        :type view: int
        :param configure: Returns the width of each image, set to 'True' once during the setup to
                          ensure figures are always the same width
        :type configure: bool
        """
        if configure:
            self.fig_ax_width = self.fig_data_ax.width()
            self.fig_img_width = self.fig_img.width() - self.fig_ax_width
            self.fig_line_width = self.fig_line.width()
            self.fig_probe_width = self.fig_probe.width()
            self.slice_width = self.fig_slice.width()
            self.slice_height = self.fig_slice.height()
            self.slice_rect = self.fig_slice.viewRect()

        self.ephys_panel_layout.apply_view(
            view,
            EphysPanelLayoutSizes(
                axis_width=self.fig_ax_width,
                image_width=self.fig_img_width,
                line_width=self.fig_line_width,
                probe_width=self.fig_probe_width,
            ),
        )

    def save_plots(self, save_path=None) -> None:
        """
        Saves all plots from the GUI into folder
        """
        # make folder to save plots to
        sess_info = ""

        if save_path:
            image_path_overview = Path(save_path)
        else:
            if self.document.output_directory is None:
                self.on_output_folder_selected()
            if self.document.output_directory is None:
                return
            image_path_overview = Path(
                self.document.output_directory
                / f"Plots_Shank_{self._active_shank_idx() + 1}"
            )

        image_path_overview.mkdir(exist_ok=True)
        self.plot_exporter.export(image_path_overview, sess_info=sess_info)

    def toggle_plots(self, options_group, reverse=False) -> None:
        """
        Allows user to toggle through image, line, probe and slice plots using keyboard shortcuts
        Alt+1, Alt+2, Alt+3 and Alt+4 respectively
        :param options_group: Set of plots to toggle through
        :param reverse: if True, goes backward
        :type options_group: QtGui.QActionGroup
        """

        current_act = options_group.checkedAction()
        actions = options_group.actions()
        if not actions:
            logger.warning("No available plot actions to toggle")
            return
        if current_act is None:
            actions[0].setChecked(True)
            actions[0].trigger()
            return
        try:
            current_idx = actions.index(current_act)
        except ValueError:
            actions[0].setChecked(True)
            actions[0].trigger()
            return
        next_idx = np.mod(current_idx + (-1 if reverse else 1), len(actions))
        actions[next_idx].setChecked(True)
        actions[next_idx].trigger()

    """
    Plot functions
    """

    def plot_histology(self, fig=None, ax="left", movable=True) -> None:
        """Compatibility wrapper for aligned histology rendering."""
        state = self._active_histology_panel_state()
        if state is not None:
            self.histology_panel.render_aligned(state, fig, movable=movable)

    def plot_histology_ref(self, fig=None, ax="right", movable=False) -> None:
        """Compatibility wrapper for reference histology rendering."""
        state = self._active_histology_panel_state()
        if state is not None:
            self.histology_panel.render_reference(state, fig, movable=movable)

    def plot_histology_nearby(self, fig=None, ax="right", movable=False) -> None:
        """Compatibility wrapper for nearby histology boundary rendering."""
        state = self._active_nearby_boundary_state()
        if state is not None:
            self.histology_panel.render_nearby(state, fig, movable=movable)

    def _probe_extent_query_kwargs(self) -> dict[str, float]:
        depth_view = self.display_state.depth_view
        return {
            "probe_tip_um": depth_view.probe_tip_um,
            "probe_top_um": depth_view.probe_top_um,
            "probe_extra_um": depth_view.probe_extra_um,
        }

    def _active_histology_panel_state(self):
        state = self.app.queries.active_histology_panel_state(
            **self._probe_extent_query_kwargs()
        )
        if state is None:
            logger.error("Cannot render histology: active alignment data is not loaded")
        return state

    def _active_scale_factor_state(self):
        state = self.app.queries.active_scale_factor_state(
            **self._probe_extent_query_kwargs()
        )
        if state is None:
            logger.error(
                "Cannot render scale factor: active alignment data is not loaded"
            )
        return state

    def _active_fit_plot_state(self):
        state = self.app.queries.active_fit_plot_state(
            depth_um=self.display_state.depth_view.fit_depth_um,
            lin_fit=self.display_state.edit_settings.lin_fit,
        )
        if state is None:
            logger.error("Cannot render fit: active alignment data is not loaded")
        return state

    def _active_nearby_boundary_state(self):
        if not self.histology_exists:
            return None
        brain_atlas = self.histology_context.brain_atlas
        if brain_atlas is None:
            logger.error("Cannot render nearby boundaries: brain atlas is not loaded")
            return None
        state = self.app.queries.active_nearby_boundary_state(
            **self._probe_extent_query_kwargs(),
            allen=self.allen,
            brain_atlas=brain_atlas,
        )
        if state is None:
            logger.error(
                "Cannot render nearby boundaries: active alignment data is not loaded"
            )
        return state

    def offset_hist_data(self, track_shift_m: float = 0.0) -> bool:
        """
        Offset location of probe tip along probe track
        """
        # If no histology we can't do alignment
        if not self.histology_exists:
            return False

        tip_position_um = self.histology_panel.tip_position_um()
        if tip_position_um is None:
            logger.error("Cannot offset alignment: probe tip line is not rendered")
            return False

        result = self.app.commands.offset_alignment_from_tip(
            tip_position_um=tip_position_um,
            probe_tip_um=self.display_state.depth_view.probe_tip_um,
            lin_fit=self.display_state.edit_settings.lin_fit,
            track_shift_m=track_shift_m,
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        return isinstance(result, AlignmentEditApplied)

    def scale_hist_data(self) -> bool:
        """
        Scale brain regions along probe track
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return False

        line_positions = self.reference_lines.positions()
        if line_positions is None:
            line_feature = np.array([], dtype=float)
            line_track = np.array([], dtype=float)
        else:
            # Feature comes from ephys plots; track comes from histology plots.
            line_feature, line_track = line_positions
        shank_runtime = self._active_shank_runtime()
        if shank_runtime is None:
            logger.error("Cannot fit alignment: active shank runtime is not loaded")
            return False

        result = self.app.commands.fit_alignment_to_reference_lines(
            shank_runtime,
            line_features_um=line_feature,
            line_tracks_um=line_track,
            lin_fit=self.display_state.edit_settings.lin_fit,
            extend_feature=self.display_state.edit_settings.extend_feature,
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        return isinstance(result, AlignmentEditApplied)

    def _connect_alignment_changed_handlers(self) -> None:
        self._event_subscriptions.extend(
            self.desktop_alignment_presenter.connect_alignment_events()
        )

    def _connect_shank_changed_handlers(self) -> None:
        self._event_subscriptions.extend(
            self.desktop_shank_presenter.connect_shank_events()
        )

    def _desktop_alignment_render_callbacks(self) -> DesktopAlignmentRenderCallbacks:
        return DesktopAlignmentRenderCallbacks(
            restore_lin_fit=self._restore_lin_fit_from_edit,
            clear_reference_lines=self.reference_lines.clear,
            capture_depth_plot_y_ranges=self._capture_depth_plot_y_ranges,
            restore_depth_plot_y_ranges=self._restore_depth_plot_y_ranges,
            reattach_reference_lines=self._reattach_reference_lines,
            probe_extent_query_kwargs=self._probe_extent_query_kwargs,
            fit_depth_um=lambda: self.display_state.depth_view.fit_depth_um,
            lin_fit_enabled=lambda: self.display_state.edit_settings.lin_fit,
            scale_factor_y_range=self._scale_factor_y_range,
            render_histology=self.histology_panel.render_aligned,
            render_scale_factor=(
                lambda state, y_range: self.histology_panel.render_scale_factor(
                    state,
                    y_range=y_range,
                )
            ),
            render_fit=self.histology_panel.render_fit,
            plot_channels=self.slice_panel.plot_channels,
            refresh_perpendicular_histology=(
                self.slice_panel.refresh_perpendicular_histology
            ),
            update_reference_lines_to_alignment=self.update_lines_points,
            create_reference_lines_for_previous_alignment=(
                self._create_reference_lines_for_previous_alignment
            ),
            set_default_feature_y_range=self.set_default_feature_y_range,
            update_status=self.update_string,
        )

    def _scale_factor_y_range(self) -> tuple[float, float]:
        y_min, y_max = self.fig_img.viewRange()[1]
        return float(y_min), float(y_max)

    def _create_reference_lines_for_previous_alignment(self) -> None:
        feature_prev = self._active_previous_feature()
        if feature_prev is not None and np.any(feature_prev):
            self.reference_lines.create_lines(np.asarray(feature_prev)[1:-1] * 1e6)

    def _desktop_shank_render_callbacks(self) -> DesktopShankRenderCallbacks:
        return DesktopShankRenderCallbacks(
            capture_plot_selection=self._capture_shank_plot_selection,
            clear_reference_lines=self.reference_lines.clear,
            prepare_runtime=self._prepare_shank_runtime_for_view,
            prepare_histology=self._prepare_shank_histology_for_view,
            apply_plot_data_state=self._apply_shank_plot_data_state,
            raw_image_payloads=lambda: self.raw_image_payloads,
            render_plot_menus=self._render_shank_plot_menus,
            render_ephys_plots=self.ephys_plot_presenter.render_shank_ephys_plots,
            render_histology_plots=self.render_histology_plots,
            restore_slice_selection=self._restore_shank_slice_selection,
            configure_view=self._configure_shank_view_after_render,
            histology_available=lambda: self.histology_exists,
            offline=lambda: self.offline,
        )

    def plot_scale_factor(self) -> None:
        """
        Plots the scale factor applied to brain regions along probe track, displayed
        alongside histology figure
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        state = self._active_scale_factor_state()
        if state is None:
            return
        self.histology_panel.render_scale_factor(
            state,
            y_range=self._scale_factor_y_range(),
        )

    def plot_fit(self) -> None:
        """
        Plots the scale factor and offset applied to channels along depth of probe track
        relative to orignal position of channels
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        state = self._active_fit_plot_state()
        if state is not None:
            self.histology_panel.render_fit(state)

    ### --------- interaction functions --------- ###
    def _teardown_session(self) -> None:
        """Break reference cycles from the previous active stream view."""
        self._clear_active_stream_presentation()
        self.runtime.clear_active_stream()
        gc.collect()

    def _clear_active_stream_presentation(self) -> None:
        """Clear desktop-owned plot and popup items for the active stream."""
        self.reference_lines.clear()
        self.popup_manager.close_all()
        self.ephys_panel.clear()
        self.slice_panel.clear()
        self.histology_panel.clear()

    def _active_shank_idx(self) -> int:
        """Return the document-owned active shank index."""
        return self.app.queries.active_shank_selection().shank_idx

    def _active_shank_runtime(self):
        """Return runtime data for the active shank, if it has been built."""
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return None
        return stream_runtime.shank_runtime_by_idx.get(self._active_shank_idx())

    def _active_alignment_state(self):
        """Return document-owned editable state for the active alignment."""
        return self.document.active_alignment_state

    def _active_alignment(self):
        """Return the document-owned active alignment, if present."""
        state = self._active_alignment_state()
        return None if state is None else state.active_alignment

    def _active_previous_feature(self):
        """Return the selected previous feature alignment, if any."""
        state = self._active_alignment_state()
        return None if state is None else state.feature_prev

    def _valid_shank_idx(self, shank_idx: int) -> int:
        """Return a shank index valid for the selected probe metadata."""
        n_shanks = self.data_context.n_shanks
        if n_shanks <= 0 or not 0 <= shank_idx < n_shanks:
            return 0
        return shank_idx

    def _select_shank_for_view(
        self,
        shank_idx: int,
        *,
        source: str,
    ) -> int | None:
        """Select a document shank for desktop presentation."""
        target_shank = self._valid_shank_idx(shank_idx)
        result = self.app.commands.select_shank(target_shank, source=source)
        if isinstance(result, Failed):
            logger.error(result.message)
            return None
        if not isinstance(result, ShankSelected):
            return None
        return result.shank_idx

    # -- Empty-state placeholder ---------------------------------------

    def _show_empty_state(self, text: str = "Select and load data") -> None:
        """Show centered guidance text in the image plot when nothing is shown."""
        if self._empty_state_item is not None:
            return
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=(160, 160, 160))
        vb = self.fig_img.getViewBox()
        vb.addItem(item, ignoreBounds=True)

        def _center(*_args):
            (x0, x1), (y0, y1) = vb.viewRange()
            item.setPos((x0 + x1) / 2.0, (y0 + y1) / 2.0)

        _center()
        vb.sigRangeChanged.connect(_center)
        self._empty_state_item = (item, _center)

    def _clear_empty_state(self) -> None:
        if self._empty_state_item is None:
            return
        item, center = self._empty_state_item
        vb = self.fig_img.getViewBox()
        try:
            vb.sigRangeChanged.disconnect(center)
        except (TypeError, RuntimeError):
            pass
        vb.removeItem(item)
        self._empty_state_item = None

    # -- Per-stream runtime cache --------------------------------------

    def _stash_and_detach_current(self) -> None:
        """Tear down the displayed view session, keeping stream runtime cached."""
        self.runtime.clear_active_stream()
        self._clear_active_stream_presentation()

    def _evict_stream_cache(self) -> None:
        """Tear down the active view session and clear cached stream runtimes.

        Called when the recording session changes — the cache belongs to one
        recording session, so this bounds memory to a single session's streams.
        """
        self._clear_active_stream_presentation()
        self.runtime.clear_stream_cache()
        gc.collect()

    def load_heavy_data(self) -> bool:
        """Load all heavy data - ephys, atlas, histology. Called once per session."""
        return self.load_data_presenter.load_heavy_data()

    def load_existing_alignments(self) -> bool:
        return self.previous_alignment_load_presenter.load_existing_alignments()

    def set_mouse_root(self, mouse_root: Path) -> bool:
        """Point the GUI at a preprocessed mouse-root directory.

        Loads ``datapackage.json``, populates the session dropdown, and clears
        probe/shank state. The user then picks a session + probe, at which
        point channel info is read from the corresponding ephys ALF.

        :param mouse_root: Directory containing ``datapackage.json``.
        :return: ``True`` on success.
        """
        return self.mouse_root_presenter.set_mouse_root(mouse_root)

    def on_mouse_root_selected(self) -> bool:
        """Prompt for the mouse-root directory."""
        return self.path_dialog_presenter.select_mouse_root()

    def on_mouse_root_edited(self) -> None:
        """Triggered when the user finishes editing the mouse-root text field."""
        self.mouse_root_presenter.mouse_root_edited()

    def on_session_combobox_activated(self, _idx: int) -> None:
        """Populate the probe dropdown for the selected session."""
        self.session_selection_presenter.session_selected()

    def on_probe_combobox_activated(self, _idx: int) -> None:
        """Select a probe: load channel info, populate shank list, derive output dir."""
        self.probe_selection_presenter.probe_selected()

    def _set_histology_available(self, available: bool) -> None:
        """Set the desktop histology availability flag."""
        self.histology_exists = available

    def _capture_pending_reference_lines(self) -> None:
        """Capture active reference-line coordinates as document state.

        Called when line handles change or when navigating away from a loaded
        alignment. The document stores only coordinates; pyqtgraph handles stay
        in the view/session layer.
        """
        if not self.document.data_loaded:
            return
        positions = self.reference_lines.positions()
        shank_idx = self._active_shank_idx()
        if positions is None:
            result = self.controller.clear_pending_reference_lines(shank_idx)
        else:
            line_feature, line_track = positions
            result = self.controller.set_pending_reference_lines(
                feature_positions_um=line_feature,
                track_positions_um=line_track,
                shank_idx=shank_idx,
            )
        if isinstance(result, Failed):
            logger.error(result.message)
            return
        logger.debug(
            "Captured reference lines for %s",
            self.document.selected_alignment_key,
        )

    def on_use_docdb_changed(self, state) -> None:
        """Handler for Use DocDB checkbox state changes"""
        self.use_docdb = state == QtCore.Qt.Checked
        logger.info(f"Use DocDB: {self.use_docdb}")

    def on_load_data_button_pressed(self) -> None:
        """Triggered when user clicks 'Load Data' button"""
        self.load_workflow_presenter.load_data_button_pressed()

    def _ensure_output_directory_for_save(
        self, requirement: Requirement | None = None
    ) -> bool:
        """Require a save location before writing alignment outputs."""
        return self.output_folder_prompt.ensure_for_save(requirement)

    def set_save_root(self, save_root: Path) -> bool:
        """Set the save-root directory. Per-probe output lands under it."""
        return self.output_path_presenter.set_save_root(save_root)

    def on_output_folder_selected(self) -> bool:
        """Prompt the user for a save-root directory."""
        return self.path_dialog_presenter.select_output_root()

    def on_output_folder_edited(self) -> None:
        """Triggered when user finishes editing output_folder_line text field."""
        self.output_path_presenter.output_folder_edited()

    def recreate_alignment_and_regions(
        self,
        track_annotations_ras: Any | None = None,
    ) -> bool:
        """Initialize active shank alignment runtime through the app command."""
        if not self.histology_exists:
            return True
        brain_atlas = self.histology_context.brain_atlas
        if brain_atlas is None:
            logger.error("Cannot recreate alignment: brain atlas is not loaded")
            return False
        shank_runtime = self._active_shank_runtime()
        if shank_runtime is None:
            logger.error(
                "Cannot recreate alignment: active shank runtime is not loaded"
            )
            return False
        if track_annotations_ras is None:
            track_annotations_ras = shank_runtime.track_annotations_ras
        if track_annotations_ras is None:
            logger.error("Cannot recreate alignment: track annotations are not loaded")
            return False

        result = self.app.commands.initialize_shank_runtime(
            shank_runtime,
            track_annotations_ras=track_annotations_ras,
            brain_atlas=brain_atlas,
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return False

        return True

    def render_histology_plots(self, *, shank_idx: int | None = None) -> None:
        """Render all histology plots. Common code."""
        if not self.histology_exists:
            return
        if shank_idx is None:
            shank_idx = self.app.queries.active_shank_selection().shank_idx

        self.plot_histology_ref()
        self.plot_histology()
        self.slice_panel.refresh_perpendicular_histology()
        self.histology_panel.set_labels_visible(True)
        self.plot_scale_factor()
        self.plot_fit()

        pending_lines = self.controller.active_pending_reference_lines(shank_idx)
        if isinstance(pending_lines, Failed):
            logger.error(pending_lines.message)
            pending_lines = None
        if pending_lines is not None:
            self.reference_lines.create_lines(
                pending_lines.feature_positions_um,
                pending_lines.track_positions_um,
            )
        else:
            feature_prev = self._active_previous_feature()
            if feature_prev is not None and np.any(feature_prev):
                self.reference_lines.create_lines(np.asarray(feature_prev)[1:-1] * 1e6)

    def setup_session_view(
        self,
        preserve_plot_selection: bool | None = None,
        *,
        shank_idx: int | None = None,
    ) -> None:
        """Setup/refresh view for current session. Used by both initial load and session switching."""
        logger.info("Setting up session view")
        if shank_idx is None:
            shank_idx = self.app.queries.active_shank_selection().shank_idx
        self.desktop_shank_presenter.render_loaded_shank(
            shank_idx=shank_idx,
            preserve_plot_selection=preserve_plot_selection,
        )

    def _capture_shank_plot_selection(
        self,
        preserve_plot_selection: bool,
    ) -> DesktopShankSelectionState:
        """Capture desktop plot selections to preserve across shank redraw."""
        prev_slice_action = (
            self.slice_options_group.checkedAction()
            if hasattr(self, "slice_options_group")
            else None
        )
        prev_slice_selection = SliceSelection.from_payload(
            prev_slice_action.data() if prev_slice_action is not None else None
        )
        prev_slice_label = (
            prev_slice_action.text() if prev_slice_action is not None else None
        )
        prev_ephys_plot_keys = (
            self.ephys_plot_presenter.current_plot_keys()
            if preserve_plot_selection and self.ephys_plot_presenter.has_plot_menus()
            else None
        )
        return DesktopShankSelectionState(
            previous_slice_selection=prev_slice_selection,
            previous_slice_label=prev_slice_label,
            previous_ephys_plot_keys=prev_ephys_plot_keys,
        )

    def _prepare_shank_runtime_for_view(self, shank_idx: int) -> None:
        """Ensure runtime state exists for the selected shank."""
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is not None:
            collection = stream_runtime.shank_runtime_for(shank_idx).collection
            logger.debug(f"Selected {len(collection.depths)} channels for this shank")

    def _prepare_shank_histology_for_view(self, shank_idx: int) -> bool:
        """Load shank histology state and recreate alignment-derived regions."""
        if not self.histology_exists:
            return True

        probe = self.data_context.probe_info
        brain_atlas = self.histology_context.brain_atlas
        if probe is None:
            raise RuntimeError("No probe selected. Please select a probe first.")
        if brain_atlas is None:
            raise RuntimeError("brain_atlas not yet loaded")
        track_annotations_ras = self.probe_track_service.load_track_annotations(
            probe=probe,
            shank_idx=shank_idx,
            brain_atlas=brain_atlas,
        )
        logger.debug("Loaded track_annotations_ras for shank")

        choices = self.controller.active_alignment_choices(shank_idx)
        if isinstance(choices, Failed):
            logger.error(choices.message)
            return False
        self.populate_lists(
            choices.choices,
            self.align_list,
            self.align_combobox,
        )
        if self._active_alignment() is None and not self._select_alignment_choice(0):
            return False
        return self.recreate_alignment_and_regions(
            track_annotations_ras=track_annotations_ras,
        )

    def _apply_shank_plot_data_state(
        self,
        state: ActiveShankPlotDataState,
    ) -> None:
        """Apply prepared shank plot-data bounds to desktop depth plots."""
        self.set_lims(np.min([0, state.channel_min_um]), state.channel_max_um)
        self.raw_image_payloads = {}

    def _render_shank_plot_menus(
        self,
        plot_menu_state: Any,
    ) -> None:
        """Refresh ephys plot menus for the selected shank."""
        if not self.ephys_plot_presenter.has_plot_menus():
            self.init_menubar()
        self.ephys_plot_presenter.render_menus(plot_menu_state)

    def _restore_shank_slice_selection(
        self,
        slice_menu_state: ActiveSliceMenuState | None,
        previous_selection: SliceSelection | None,
        previous_label: str | None,
    ) -> None:
        """Restore or choose the active slice menu selection after shank redraw."""
        if slice_menu_state is None:
            logger.warning("No default slice selection is available")
        else:
            choice = slice_menu_state.selection
            selected_action = self.slice_panel.action_for_selection(choice.selection)
            if selected_action is None:
                selected_action = self.slice_init
            if selected_action is None:
                logger.warning("No slice action is available")
            else:
                if (
                    previous_selection is not None
                    and SliceSelection.from_payload(selected_action.data())
                    != previous_selection
                    and not choice.used_previous
                ):
                    logger.info(
                        f"Slice selection '{previous_label}' not available "
                        f"for this probe; falling back to '{selected_action.text()}'"
                    )
                selected_action.setChecked(True)
                selected_selection = SliceSelection.from_payload(selected_action.data())
                if selected_selection is not None:
                    self.slice_panel.plot_slice_selection(selected_selection)

    def _configure_shank_view_after_render(self, preserve_plot_selection: bool) -> None:
        """Apply one-time view configuration after shank rendering."""
        self.set_view(view=1, configure=self.configure and not preserve_plot_selection)
        if not preserve_plot_selection:
            self.configure = False

    def on_shank_selected(self, idx) -> None:
        """Triggered when selecting shank from dropdown"""
        shank_text = self.shank_combobox.currentText()
        new_shank_id = int(shank_text.split("/")[0])
        new_shank_idx = new_shank_id - 1
        selection = self.app.queries.active_shank_selection()
        if new_shank_idx == selection.shank_idx:
            logger.info(f"Shank {new_shank_id} already selected")
            return

        result = self.app.commands.select_shank(
            new_shank_idx,
            outgoing_reference_lines=self.reference_lines.positions(),
            source="dropdown",
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return
        if not isinstance(result, ShankSelected):
            return

        logger.info(f"Shank {new_shank_id} selected (index {result.shank_idx})")

    def on_alignment_selected(self, idx) -> None:
        """Triggered when selecting alignment from dropdown"""
        logger.info(f"Alignment index {idx} selected")

        if not self._select_alignment_choice(idx):
            return

        if not self.document.data_loaded:
            # Data not loaded yet - just update alignment params
            logger.info("Data not loaded yet, alignment params updated")
            return

        if not self.recreate_alignment_and_regions():
            return

        self.render_histology_plots()

        logger.info("Alignment change complete")

    def _select_alignment_choice(self, idx: int) -> bool:
        """Select an alignment choice through the controller and project it."""
        result = self.app.commands.select_previous_alignment(idx)
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        assert isinstance(result, PreviousAlignmentSelected)
        return True

    def toggle_histology_button_pressed(self) -> None:
        boundaries_visible = self.display_state.toggle_histology_boundaries_visible()
        if not boundaries_visible:
            self.plot_histology_nearby()
        else:
            self.plot_histology_ref()

    def toggle_histology_map_button_pressed(self) -> None:
        self.display_state.toggle_region_annotation_source()

        self.plot_histology()
        self.plot_histology_ref()
        self.plot_scale_factor()
        self._reattach_reference_lines()

    def fit_button_pressed(self) -> None:
        """
        Triggered when fit button or Enter key pressed, applies scaling factor to brain regions
        according to locations of reference lines on ephys and histology plots. Updates all plots
        and indices after scaling has been applied
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if not self.scale_hist_data():
            return

    def offset_button_pressed(
        self, _checked: bool = False, *, track_shift_m: float = 0.0
    ) -> None:
        """
        Triggered when offset button or o key pressed, applies offset to brain regions according to
        locations of probe tip line on histology plot. Updates all plots and indices after offset
        has been applied
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if not self.offset_hist_data(track_shift_m=track_shift_m):
            return

    def movedown_button_pressed(self) -> None:
        """
        Triggered when Shift+down key pressed. Moves probe tip down by 50um and offsets data
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        alignment = self._active_alignment()
        shank_runtime = self._active_shank_runtime()
        if (
            alignment is not None
            and shank_runtime is not None
            and alignment.track[-1] - 50 / 1e6
            >= np.max(shank_runtime.chn_depths) / 1e6
        ):
            self.offset_button_pressed(track_shift_m=-50 / 1e6)

    def moveup_button_pressed(self) -> None:
        """
        Triggered when Shift+down key pressed. Moves probe tip up by 50um and offsets data
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        alignment = self._active_alignment()
        shank_runtime = self._active_shank_runtime()
        if (
            alignment is not None
            and shank_runtime is not None
            and alignment.track[0] + 50 / 1e6
            <= np.min(shank_runtime.chn_depths) / 1e6
        ):
            self.offset_button_pressed(track_shift_m=50 / 1e6)

    def toggle_labels_button_pressed(self) -> None:
        """
        Triggered when Shift+A key pressed. Shows/hides labels Allen atlas labels on brain regions
        in histology plots
        """
        self.histology_panel.toggle_labels()

    def toggle_line_button_pressed(self) -> None:
        """
        Triggered when Shift+L key pressed. Shows/hides reference lines on ephys and histology
        plots
        """
        lines_visible = self.display_state.toggle_reference_lines_visible()
        if not lines_visible:
            self.reference_lines.remove_from_plots()
        else:
            self.reference_lines.add_to_plots()

    def toggle_channel_button_pressed(self) -> None:
        """
        Triggered when Shift+C key pressed. Shows/hides channels, tip, and trajectory on slice image
        and perpendicular slice image
        """
        self.slice_panel.toggle_channel_visibility()

    def delete_line_button_pressed(self) -> None:
        """
        Triggered when mouse hovers over a reference line and shift+D keys are pressed.
        Deletes a reference line from the ephys and histology plots
        """

        self.reference_lines.delete_selected()

    def describe_labels_pressed(self) -> None:
        self.interaction_presenter.describe_labels_pressed()

    def label_closed(self, popup) -> None:
        self.interaction_presenter.label_closed(popup)

    def label_moved(self) -> None:
        self.interaction_presenter.label_moved()

    def label_pressed(self, item) -> None:
        self.interaction_presenter.label_pressed(item)

    def next_button_pressed(self) -> None:
        """
        Triggered when right key pressed. Updates all plots and indices with next move. Ensures
        user cannot go past latest move
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        result = self.app.commands.go_next_alignment()
        if isinstance(result, Failed):
            logger.error(result.message)
            return

    def prev_button_pressed(self) -> None:
        """
        Triggered when left key pressed. Updates all plots and indices with previous move.
        Ensures user cannot go back past the active edit-history buffer.
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        result = self.app.commands.go_previous_alignment()
        if isinstance(result, Failed):
            logger.error(result.message)
            return

    def reset_button_pressed(self) -> None:
        """
        Triggered when reset button or Shift+R key pressed. Resets channel locations to orignal
        location
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        shank_runtime = self._active_shank_runtime()
        if shank_runtime is None:
            logger.error("Cannot reset alignment: active shank runtime is not loaded")
            return

        result = self.app.commands.reset_alignment_to_initial(
            shank_runtime,
            lin_fit=self.display_state.edit_settings.lin_fit,
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return
        if not isinstance(result, AlignmentEditApplied):
            return

    def _restore_lin_fit_from_edit(self, lin_fit: bool | None) -> None:
        if lin_fit is None:
            return
        self.display_state.edit_settings.set_lin_fit(lin_fit)
        self.lin_fit_option.blockSignals(True)
        self.lin_fit_option.setChecked(self.display_state.edit_settings.lin_fit)
        self.lin_fit_option.blockSignals(False)

    def run_complete_button_in_thread(self) -> None:
        self.thread = QThread()
        self.worker = Worker(self.complete_button_pressed_offline)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)  # Gracefully stop thread
        self.worker.finished.connect(self.worker.deleteLater)  # Clean up worker
        self.thread.finished.connect(self.thread.deleteLater)  # Clean up thread

    def complete_button_pressed_offline(self) -> None:
        """
        Triggered when save button or Shift+S keys are pressed.
        Saves final channel locations for all visited shanks to JSON files.
        """
        self.save_workflow_presenter.save_alignment_outputs()

    def display_qc_options(self) -> None:
        self.save_workflow_presenter.display_qc_options()

    def qc_button_clicked(self) -> None:
        self.save_workflow_presenter.qc_button_clicked()

    def _selected_qc_descriptions(self) -> list[str]:
        """Return selected QC description labels."""
        ephys_desc = []
        for button in self.desc_buttons.buttons():
            if button.isChecked():
                ephys_desc.append(button.text())
        return ephys_desc

    def reset_axis_button_pressed(self) -> None:
        self.set_default_feature_y_range()
        feature_xrange = self.ephys_panel.feature_xrange
        if feature_xrange is not None:
            self.fig_img.setXRange(
                min=feature_xrange[0],
                max=feature_xrange[1],
                padding=0,
            )

    def display_session_notes(self) -> None:
        self.interaction_presenter.display_session_notes()

    def display_nearby_sessions(self) -> None:
        self._show_one_unsupported("Nearby sessions")

    def popup_closed(self, popup) -> None:
        self.interaction_presenter.popup_closed(popup)

    def popup_moved(self) -> None:
        self.interaction_presenter.popup_moved()

    def close_popups(self) -> None:
        self.interaction_presenter.close_popups()

    def minimise_popups(self) -> None:
        self.interaction_presenter.minimise_popups()

    def lin_fit_option_changed(self, state) -> None:
        """
        Triggered when Linear fit checkbox state changes.
        Updates the flag and recomputes alignment by calling
        fit_button_pressed.
        """
        # Update the flag
        self.display_state.edit_settings.set_lin_fit(state != 0)

        # Only recompute if we have reference lines and histology
        # If no lines yet, just update the flag for future use
        if not self.histology_exists or not self.reference_lines.has_lines():
            return

        # Recompute alignment with new setting using existing fit logic
        self.fit_button_pressed()

    def cluster_clicked(self, item, point):
        return self.interaction_presenter.cluster_clicked(item, point)

    def display_subject_scaling(self) -> None:
        self._show_one_unsupported("Subject scaling")

    def display_region_features(self) -> None:
        self._show_one_unsupported("Region features")

    def on_mouse_double_clicked(self, event) -> None:
        """
        Triggered when a double click event is detected on ephys of histology plots. Adds reference
        line on ephys and histology plot that can be moved to align ephys signatures with brain
        regions. Also adds scatter point on fit plot
        :param event: double click event signals
        :type event: pyqtgraph mouseEvents
        """
        self.interaction_presenter.on_mouse_double_clicked(event)

    def on_mouse_hover(self, items) -> None:
        """
        Returns the pyqtgraph items that the mouse is hovering over. Used to identify reference
        lines so that they can be deleted
        """
        self.interaction_presenter.on_mouse_hover(items)

    def tip_line_moved(self) -> None:
        """
        Triggered when dotted line indicating probe tip on self.fig_hist moved. Gets the y pos of
        probe tip line and ensures the probe top line is set to probe tip line y pos + 3840
        """
        self.histology_panel.sync_top_to_tip()

    def top_line_moved(self) -> None:
        """
        Triggered when dotted line indicating probe top on self.fig_hist moved. Gets the y pos of
        probe top line and ensures the probe tip line is set to probe top line y pos - 3840
        """
        self.histology_panel.sync_tip_to_top()

    def _reattach_reference_lines(self) -> None:
        self.reference_lines.remove_from_plots()
        self.reference_lines.add_to_plots()

    def update_lines_points(self) -> None:
        """
        Updates position of reference lines on histology plot after fit has been applied. Also
        updates location of scatter point
        """
        self.reference_lines.sync_track_to_feature()

    def create_line_style(self):
        """
        Create random choice of colour and style for reference line
        :return pen: style to use for the line
        :type pen: pyqtgraph Pen
        :return brush: colour use for the line
        :type brush: pyqtgraph Brush
        """
        colours = [
            "#cc0000",
            "#6aa84f",
            "#ff8d00",
            "#00FFF7",
            "#03fc84",
            "#fc03e7",
            "#1c03fc",
            "#000000",
        ]
        style = [
            QtCore.Qt.SolidLine,
            QtCore.Qt.DashLine,
            QtCore.Qt.DashDotLine,
        ]
        col = QtGui.QColor(colours[randrange(len(colours))])
        sty = style[randrange(len(style))]
        pen = pg.mkPen(color=col, style=sty, width=7)
        brush = pg.mkBrush(color=col)
        return pen, brush

    def update_string(self) -> None:
        """
        Updates text boxes to indicate to user which move they are looking at
        """
        state = self._active_alignment_state()
        current_idx = 0 if state is None else state.edit_history.current_idx
        total_idx = 0 if state is None else state.edit_history.total_idx
        self.idx_string.setText(f"Current Index = {current_idx}")
        self.tot_idx_string.setText(f"Total Index = {total_idx}")


def viewer(probe_id, one=None, histology=False, spike_collection=None, title=None):
    """ """
    qt.create_app()
    av = MainWindow._get_or_create(
        probe_id=probe_id,
        one=one,
        histology=histology,
        spike_collection=spike_collection,
        title=title,
    )
    av.show()
    return av


def setup_logging(log_level=logging.INFO, log_file=None) -> None:
    """
    Setup logging configuration for the entire application.

    Parameters
    ----------
    log_level : int
        Logging level (logging.DEBUG, logging.INFO, etc.)
    log_file : Path or str, optional
        If provided, also log to this file
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_file}")

    # Log initial message
    root_logger.info("=" * 60)
    root_logger.info("Ephys Alignment GUI Starting")
    root_logger.info("=" * 60)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="IBL ephys alignment GUI for preprocessed datapackages"
    )
    parser.add_argument(
        "-o",
        "--offline",
        default=True,
        required=False,
        help="Legacy flag; ONE/Alyx online mode is not supported.",
    )
    parser.add_argument(
        "-r",
        "--remote",
        default=False,
        required=False,
        action="store_true",
        help="Remote mode",
    )
    parser.add_argument(
        "-i",
        "--insertion",
        default=None,
        required=False,
        help="Insertion mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        required=False,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        required=False,
        type=str,
        help="Path to log file (optional, logs to console by default)",
    )
    args = parser.parse_args()

    # Setup logging FIRST, before anything else
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level=log_level, log_file=args.log_file)

    # Get logger for main module
    logger.info(f"Arguments: {args}")

    app = QtWidgets.QApplication([])
    mainapp = MainWindow(
        offline=args.offline, probe_id=args.insertion, remote=args.remote
    )
    # mainapp = MainWindow(offline=True)
    mainapp.show()

    logger.info("Starting Qt event loop")
    app.exec_()


if __name__ == "__main__":
    main()
