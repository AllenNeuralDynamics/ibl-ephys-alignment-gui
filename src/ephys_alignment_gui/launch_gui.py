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
from ephys_alignment_gui.app import CachedEphysDataActivated
from ephys_alignment_gui.controller import (
    AlignmentChoicesUpdated,
    AlignmentEditApplied,
    AlignmentOutputsSaved,
    MouseRootLoaded,
    OutputRootSet,
    PreviousAlignmentSelected,
    ProbeSelected,
    RecordingSelected,
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
from ephys_alignment_gui.desktop_load_workflow_presenter import (
    DesktopLoadWorkflowPresenter,
    DesktopOutputFolderPrompt,
    OutputFolderPromptCallbacks,
)
from ephys_alignment_gui.desktop_previous_alignment_load_presenter import (
    DesktopPreviousAlignmentLoadPresenter,
    PreviousAlignmentLoadCallbacks,
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
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.ephys_stream_runtime import StreamKey
from ephys_alignment_gui.event_bus import EventSubscription
from ephys_alignment_gui.histology_panel_presenter import (
    FitPanelItems,
    HistologyPanelAxes,
    HistologyPanelPlots,
    HistologyPanelPresenter,
    HistologyPanelStyle,
)
from ephys_alignment_gui.histology_data_workflow import (
    HistologyDataLoaded,
    HistologyDataUnavailable,
)
from ephys_alignment_gui.reference_line_layer import (
    ReferenceLineLayer,
    ReferenceLinePlots,
)
from ephys_alignment_gui.session_runtime import (
    LoadDataAlreadyActive,
    LoadDataCachedStreamAvailable,
)
from ephys_alignment_gui.settings import (
    INPUT_ROOT_ENV_VAR,
    OUTPUT_ROOT_ENV_VAR,
    input_root_from_environment,
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
    Blocked,
    Failed,
    Ok,
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
            cluster_clicked=self.cluster_clicked,
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
                add_lines_points=self.add_lines_points,
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
        self._init_load_workflow_presenter()
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
                    self._derive_output_directory_from_save_root
                ),
                has_output_directory=lambda: self.document.output_directory is not None,
                select_output_folder=self.on_output_folder_selected,
            ),
        )
        self.load_workflow_presenter = DesktopLoadWorkflowPresenter(
            can_load_data=self.controller.can_load_data,
            load_heavy_data=self.load_heavy_data,
            output_folder_prompt=self.output_folder_prompt,
        )

    def _init_previous_alignment_load_presenter(self) -> None:
        """Wire desktop workflow for loading previous alignments."""
        self.previous_alignment_load_presenter = DesktopPreviousAlignmentLoadPresenter(
            commands=self.app.commands,
            callbacks=PreviousAlignmentLoadCallbacks(
                select_folder=lambda: QtWidgets.QFileDialog.getExistingDirectory(
                    None,
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
                add_lines_points=self.add_lines_points,
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
        self.popup_manager = DesktopPopupManager()
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

    def plot_perpendicular_histology(self, channel_name: str = "ccf") -> None:
        """Compatibility wrapper for perpendicular slice rendering."""
        self.slice_panel.plot_perpendicular_histology(channel_name)

    def update_perpendicular_levels(self) -> None:
        """Compatibility wrapper for slice/perpendicular lookup synchronization."""
        self.slice_panel.update_perpendicular_levels()

    def refresh_perpendicular_histology(self) -> None:
        """Compatibility wrapper for refreshing the perpendicular slice."""
        self.slice_panel.refresh_perpendicular_histology()

    def _current_scalar_slice_channel(self) -> str | None:
        """Compatibility wrapper for current scalar slice selection."""
        return self.slice_panel.current_scalar_slice_channel()

    def _current_slice_render_state(self) -> Any:
        """Compatibility wrapper for current slice render state."""
        return self.slice_panel.current_slice_render_state()

    def _current_slice_selection(self) -> SliceSelection | None:
        """Compatibility wrapper for current slice selection."""
        return self.slice_panel.current_slice_selection()

    def _slice_action_for_selection(self, selection: SliceSelection) -> Any:
        """Compatibility wrapper for slice QAction lookup."""
        return self.slice_panel.action_for_selection(selection)

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
            self.create_lines(np.asarray(feature_prev)[1:-1] * 1e6)

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

    def plot_slice(self, data, img_type) -> None:
        """Compatibility wrapper for legacy slice-data call sites."""
        self.slice_panel.plot_slice(data, img_type)

    def _selection_for_slice_payload(
        self,
        data: Any,
        img_type: str,
    ) -> SliceSelection | None:
        """Compatibility wrapper for legacy slice-data selection lookup."""
        return self.slice_panel.selection_for_slice_payload(data, img_type)

    def plot_slice_selection(self, selection: SliceSelection) -> None:
        """Compatibility wrapper for coronal slice selection rendering."""
        self.slice_panel.plot_slice_selection(selection)

    def render_slice(self, render_state: Any) -> None:
        """Compatibility wrapper for coronal slice rendering."""
        self.slice_panel.render_slice(render_state)

    def plot_channels(self, projection=None) -> None:
        """Compatibility wrapper for coronal slice channel overlays."""
        self.slice_panel.plot_channels(projection)

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

    def _stream_key_for_selection(
        self,
        recording_id: str,
        probe_name: str,
    ) -> StreamKey | None:
        """Return the ephys stream key for a recording/probe selection."""
        return self.app.queries.stream_key_for_selection(recording_id, probe_name)

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

    def _activate_cached_stream(
        self,
        session_name: str,
        probe_name: str,
        stream_key: StreamKey,
        shank_idx: int,
    ) -> bool:
        """Display an already-loaded stream from the cache — no heavy reload.

        Reuses cached stream data and per-shank PlotData. A fresh view session
        is created as a view adapter; document-owned edit state is projected
        onto the active shank compatibility object.
        """
        self.init_session_variables()
        result = self.app.commands.activate_cached_ephys_data(
            recording_id=session_name,
            probe_name=probe_name,
            stream_key=stream_key,
            shank_idx=shank_idx,
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        assert isinstance(result, CachedEphysDataActivated)
        target_shank = result.shank_idx

        self._clear_empty_state()

        if result.probe.shanks:
            self.populate_lists(
                result.probe.shanks,
                self.shank_list,
                self.shank_combobox,
            )
            self.shank_combobox.setCurrentIndex(target_shank)
        self._display_output_directory(result.probe.output_directory)

        self.setup_session_view(preserve_plot_selection=True, shank_idx=target_shank)
        logger.info(f"Activated cached stream {stream_key}")
        return True

    def load_heavy_data(self) -> None:
        """Load all heavy data - ephys, atlas, histology. Called once per session."""

        target_shank = self.app.queries.active_shank_selection().shank_idx
        probe_name = self.probe_combobox.currentText()
        stream_key = self._stream_key_for_selection(
            self.session_combobox.currentText(),
            probe_name,
        )
        load_plan = self.app.queries.plan_load_data(stream_key, target_shank)
        if isinstance(load_plan, LoadDataAlreadyActive):
            logger.info(
                "Data already loaded for stream %s shank %s; skipping load",
                stream_key,
                target_shank,
            )
            return
        if isinstance(load_plan, LoadDataCachedStreamAvailable):
            self._capture_pending_reference_lines()
            self._stash_and_detach_current()
            cached_stream_key = load_plan.target.stream_key
            assert cached_stream_key is not None
            if self._activate_cached_stream(
                self.session_combobox.currentText(),
                probe_name,
                cached_stream_key,
                load_plan.target.shank_idx,
            ):
                self.load_data_button.setEnabled(True)
            return

        with BusyContext(
            self,
            "Loading heavy data...",
            "Data loaded successfully",
            disable_widgets=self.load_data_button,
        ) as ctx:
            logger.info("=== Starting heavy data load ===")
            self._capture_pending_reference_lines()
            prepared = self.controller.prepare_load_data()
            # Preserve the shank the user has selected so a (re)load lands on
            # that shank rather than snapping back to shank 0. Load Probe is a
            # per-probe op; shank switching is done via the dropdown. Fixes the
            # dropdown <-> displayed-shank drift on load.
            # Drop any stale cache entry for this stream before rebuilding it,
            # so we never leave a torn-down session in the cache.
            self.runtime.prepare_fresh_load(stream_key)
            self._teardown_session()
            self.init_session_variables()
            selected_shank = self._select_shank_for_view(
                target_shank,
                source="load-data",
            )
            if selected_shank is None:
                return
            target_shank = selected_shank
            logger.info(f"Loading probe data, active shank index {target_shank}")

            # Load ephys data (session-specific, always reload)
            ctx.update_message("Loading ephys data...")
            logger.info("Loading ephys data...")
            load_result = self.app.commands.load_fresh_ephys_data(target_shank)
            if isinstance(load_result, Failed):
                logger.error(load_result.message)
                return
            stream_runtime = load_result.stream_runtime
            target_shank = load_result.shank_idx

            logger.info(f"Loaded ephys data from {stream_runtime.stream.ephys_dir}")

            # Load atlas and histology (subject-level, cached if same subject)
            if not self.app.queries.histology_data_loaded():
                ctx.update_message("Loading atlas and histology...")
                logger.info("Loading atlas and histology...")
            histology_result = self.app.commands.load_histology_data()
            if isinstance(histology_result, HistologyDataLoaded):
                logger.info("Atlas and histology loaded successfully")
            elif isinstance(histology_result, HistologyDataUnavailable):
                logger.error(histology_result.message)
                self.histology_exists = False

            # Setup view for current shank (common with switch_shank_view)
            ctx.update_message("Setting up visualization...")
            self.setup_session_view(
                preserve_plot_selection=prepared.preserve_plot_selection,
                shank_idx=target_shank,
            )

            self._clear_empty_state()
            logger.info("=== Heavy data load complete ===")

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
        with BusyContext(
            self,
            "Loading datapackage...",
            "Mouse root loaded",
            disable_widgets=[self.mouse_root_button, self.mouse_root_line],
        ):
            result = self.controller.set_mouse_root(mouse_root)
            if isinstance(result, Failed):
                logger.error(result.message)
                return False
            assert isinstance(result, MouseRootLoaded)
            if result.root_changed:
                self.histology_context.clear()
            mr = result.mouse_root

            self.mouse_root_line.setText(str(mouse_root))

            sessions = mr.sessions
            self.populate_lists(sessions, self.session_list, self.session_combobox)
            self.probe_list.clear()
            self.shank_list.clear()
            self.load_data_button.setEnabled(False)
            n_probes = sum(len(rec_probes) for rec_probes in mr.probes.values())
            logger.info(
                f"Loaded mouse {mr.mouse_id!r} with "
                f"{len(sessions)} session(s), {n_probes} probe(s)"
            )
            # Auto-select the first session + probe, if any.
            if sessions:
                self.session_combobox.setCurrentIndex(0)
                self.on_session_combobox_activated(0)
        return True

    def on_mouse_root_selected(self) -> bool:
        """Open a QFileDialog for the mouse-root directory."""
        start_dir = self._mouse_root_dialog_start_dir()
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Mouse Root", directory=start_dir
        )
        if not folder:
            return False
        return self.set_mouse_root(Path(folder))

    def _mouse_root_dialog_start_dir(self) -> str:
        """Return the directory the mouse-root dialog should open in."""
        if self.data_context.mouse_root is not None:
            return str(self.data_context.mouse_root.root)
        input_root = input_root_from_environment()
        if input_root is None:
            return ""
        if input_root.is_dir():
            return str(input_root)
        logger.warning(
            "Ignoring %s because it is not a directory: %s",
            INPUT_ROOT_ENV_VAR,
            input_root,
        )
        return ""

    def on_mouse_root_edited(self) -> None:
        """Triggered when the user finishes editing the mouse-root text field."""
        text = self.mouse_root_line.text().strip()
        if not text:
            self.load_data_button.setEnabled(False)
            return
        try:
            path = Path(text)
            ok = self.set_mouse_root(path)
        except Exception as e:
            logger.error(f"Invalid mouse-root path: {e}")
            self.load_data_button.setEnabled(False)
            return
        if not ok:
            self.load_data_button.setEnabled(False)

    def on_session_combobox_activated(self, _idx: int) -> None:
        """Populate the probe dropdown for the selected session."""
        if self.data_context.mouse_root is None:
            return
        session = self.session_combobox.currentText()
        if not session:
            return
        # The stream cache belongs to one recording session. Capture the
        # current probe's WIP (survives stream eviction), then
        # evict every cached stream so memory is bounded to one session.
        self._capture_pending_reference_lines()
        self._evict_stream_cache()
        result = self.controller.select_recording(session)
        if isinstance(result, Failed):
            logger.error(result.message)
            return
        assert isinstance(result, RecordingSelected)
        self._show_empty_state()
        probes = result.probes
        self.populate_lists(probes, self.probe_list, self.probe_combobox)
        self.shank_list.clear()
        self.load_data_button.setEnabled(False)
        if probes:
            self.probe_combobox.setCurrentIndex(0)
            self.on_probe_combobox_activated(0)

    def on_probe_combobox_activated(self, _idx: int) -> None:
        """Select a probe: load channel info, populate shank list, derive output dir."""
        if self.data_context.mouse_root is None:
            return
        session = self.session_combobox.currentText()
        probe_name = self.probe_combobox.currentText()
        if not session or not probe_name:
            return

        # Capture outgoing reference-line coordinates before their pyqtgraph
        # handles are torn down. Applied alignment history already lives on
        # the document state.
        self._capture_pending_reference_lines()
        # Free the figures from the outgoing view session. Loaded stream data
        # stays in the stream-runtime cache.
        self._stash_and_detach_current()

        # Cache HIT: show the already-loaded stream instantly (no heavy reload).
        stream_key = self._stream_key_for_selection(session, probe_name)
        load_plan = self.app.queries.plan_load_data(
            stream_key,
            self._active_shank_idx(),
        )
        if isinstance(load_plan, LoadDataCachedStreamAvailable):
            cached_stream_key = load_plan.target.stream_key
            assert cached_stream_key is not None
            if self._activate_cached_stream(
                session,
                probe_name,
                cached_stream_key,
                load_plan.cached_shank_idx,
            ):
                self.load_data_button.setEnabled(True)
            return

        # Cache MISS: clear the display and prepare the loader + a fresh session
        # for an explicit Load. Nothing is shown until the user loads.
        self._show_empty_state()
        with BusyContext(
            self,
            "Loading channel info...",
            "Ready",
            disable_widgets=[self.probe_combobox, self.session_combobox],
        ):
            result = self.controller.select_probe(session, probe_name)
            if isinstance(result, Failed):
                logger.error(result.message)
                self.load_data_button.setEnabled(False)
                return
            assert isinstance(result, ProbeSelected)

            if result.shanks:
                self.populate_lists(result.shanks, self.shank_list, self.shank_combobox)
                logger.info(f"Found {self.data_context.n_shanks} shanks in data.")

            # Fresh desktop session for pre-Load view cleanup; document owns
            # the selected shank.
            self.init_session_variables()
            if self._select_shank_for_view(0, source="probe-selected") is None:
                self.load_data_button.setEnabled(False)
                return

            self._display_output_directory(result.output_directory)

        self.load_data_button.setEnabled(True)

    def _display_output_directory(self, output_directory: Path | None) -> None:
        """Reflect a derived per-probe output directory in the UI."""
        if output_directory is None:
            return
        self.output_folder_line.setText(str(output_directory))
        logger.info(f"Output dir: {output_directory}")

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

    def _derive_output_directory_from_save_root(self) -> bool:
        """Derive and display the probe output directory if a save root exists."""
        if self.document.output_root is None:
            return False
        result = self.controller.derive_output_directory()
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        if result.output_directory is None:
            return False
        self._display_output_directory(result.output_directory)
        return True

    def _ensure_output_directory_for_save(
        self, requirement: Requirement | None = None
    ) -> bool:
        """Require a save location before writing alignment outputs."""
        return self.output_folder_prompt.ensure_for_save(requirement)

    def set_save_root(self, save_root: Path) -> bool:
        """Set the save-root directory. Per-probe output lands under it."""
        result = self.controller.set_output_root(save_root)
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        assert isinstance(result, OutputRootSet)
        save_root = result.output_root
        logger.info(f"Save root set to: {save_root}")
        if result.output_directory is not None:
            self._display_output_directory(result.output_directory)
        else:
            # No probe yet — show the save-root itself until a probe is picked.
            self.output_folder_line.setText(str(save_root))
        return True

    def on_output_folder_selected(self) -> bool:
        """Prompt the user for a save-root directory."""
        output_root = self.document.output_root
        start_dir = str(output_root) if output_root is not None else ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Save Root", directory=start_dir
        )
        if not folder:
            return False
        return self.set_save_root(Path(folder))

    def on_output_folder_edited(self) -> None:
        """Triggered when user finishes editing output_folder_line text field."""
        text = self.output_folder_line.text().strip()
        if not text:
            return
        try:
            path = Path(text)
        except Exception as e:
            logger.error(f"Invalid output path: {e}")
            return
        # Editing this field is taken as setting a new save-root.
        self.set_save_root(path)

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

        result = self.app.commands.initialize_shank_alignment_runtime(
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
            self.create_lines(
                pending_lines.feature_positions_um,
                pending_lines.track_positions_um,
            )
        else:
            feature_prev = self._active_previous_feature()
            if feature_prev is not None and np.any(feature_prev):
                self.create_lines(np.asarray(feature_prev)[1:-1] * 1e6)

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
            if preserve_plot_selection
            and self.ephys_plot_presenter.has_plot_menus()
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
        self.remove_lines_points()
        self.add_lines_points()

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
            self.remove_lines_points()
        else:
            self.add_lines_points()

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
        # if no histology don't show
        if not self.histology_exists:
            return

        idx = self.histology_panel.selected_region_index()
        shank_runtime = self._active_shank_runtime()
        if (
            idx is not None
            and shank_runtime is not None
            and shank_runtime.ephysalign is not None
        ):
            description, lookup = self.region_lookup_service.get_region_description(
                shank_runtime.ephysalign.region_id[idx][0]
            )
            item = self.struct_list.findItems(lookup, flags=QtCore.Qt.MatchRecursive)
            model_item = self.struct_list.indexFromItem(item[0])
            self.struct_view.collapseAll()
            self.struct_view.scrollTo(model_item)
            self.struct_view.setCurrentIndex(model_item)
            self.struct_description.setText(description)

            if self.popup_manager.label_window is None:
                label_window = ephys_gui.PopupWindow(
                    title="Structure Information",
                    size=(500, 700),
                    graphics=False,
                )
                label_window.layout.addWidget(self.struct_view)
                label_window.layout.addWidget(self.struct_description)
                label_window.layout.setRowStretch(0, 7)
                label_window.layout.setRowStretch(1, 3)
                label_window.closed.connect(self.label_closed)
                label_window.moved.connect(self.label_moved)
                self.popup_manager.label_window = label_window
                self.activateWindow()
            else:
                self.popup_manager.label_window.show()
                self.activateWindow()

    def label_closed(self, popup) -> None:
        if self.popup_manager.label_window is not None:
            self.popup_manager.label_window.hide()

    def label_moved(self) -> None:
        self.activateWindow()

    def label_pressed(self, item) -> None:
        idx = int(item.model().itemFromIndex(item).accessibleText())
        description, lookup = self.region_lookup_service.get_region_description(idx)
        item = self.struct_list.findItems(lookup, flags=QtCore.Qt.MatchRecursive)
        model_item = self.struct_list.indexFromItem(item[0])
        self.struct_view.setCurrentIndex(model_item)
        self.struct_description.setText(description)

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
        save_ready = self.controller.can_save_alignment_output()
        if isinstance(save_ready, Blocked):
            if not self._ensure_output_directory_for_save(save_ready.first):
                return
            save_ready = self.controller.can_save_alignment_output()

        if not isinstance(save_ready, Ok):
            if isinstance(save_ready, Blocked):
                self.load_workflow_presenter.log_requirement(save_ready.first)
            return

        with BusyContext(
            self,
            "Saving...",
            "Saved successfully",
            disable_widgets=self.complete_button,
        ):
            output_inputs, states_by_key = self._visited_alignment_output_inputs()
            if not output_inputs:
                logger.error("No visited alignments are ready to save")
                return

            outputs = self.controller.build_alignment_outputs(output_inputs)
            if isinstance(outputs, Failed):
                logger.error(outputs.message)
                return

            for key, state in states_by_key.items():
                alignment = state.active_alignment
                if alignment is None:
                    continue
                state.add_alignment(alignment.feature, alignment.track)

            logger.info("Saving output files to results folder...")
            saved_count = 0
            for key, output in outputs.items():
                state = states_by_key[key]
                saved = self.controller.save_alignment_output(
                    output,
                    state.alignments,
                    key.shank_idx,
                    self.use_docdb,
                )
                if isinstance(saved, Failed):
                    logger.error(saved.message)
                    return
                assert isinstance(saved, AlignmentOutputsSaved)
                saved_count += 1

                if saved.saved.docdb_probe_name is not None:
                    if saved.saved.docdb_error is not None:
                        logger.error(
                            "Failed to write to DocDB with error %s. Output saved to results folder",
                            saved.saved.docdb_error,
                        )
                    else:
                        logger.info(
                            "Channels locations saved, and ccf coordinates saved for %s",
                            saved.saved.docdb_probe_name,
                        )

            active_choices = self.controller.active_alignment_choices(
                self._active_shank_idx()
            )
            if isinstance(active_choices, AlignmentChoicesUpdated):
                self.populate_lists(
                    active_choices.choices,
                    self.align_list,
                    self.align_combobox,
                )
            logger.info(
                "Channel locations saved to results folder for %d visited alignment(s)",
                saved_count,
            )

    def _visited_alignment_output_inputs(
        self,
    ) -> tuple[
        dict[AlignmentKey, tuple[Any, Any]],
        dict[AlignmentKey, Any],
    ]:
        """Collect channel-location save inputs for visited shanks."""
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return {}, {}
        probe = self.data_context.probe_info
        if probe is None:
            return {}, {}

        states_for_probe = self.document.alignment_states_for_current_probe()
        output_inputs: dict[AlignmentKey, tuple[Any, Any]] = {}
        states_by_key: dict[AlignmentKey, Any] = {}
        for shank_idx, shank_runtime in stream_runtime.visited_shank_runtimes().items():
            key = AlignmentKey(
                recording_id=probe.recording_id,
                ephys_collection=probe.ephys_collection,
                shank_idx=shank_idx,
            )
            state = states_for_probe.get(key)
            if state is None or state.active_alignment is None:
                continue
            if shank_runtime.ephysalign is None or shank_runtime.chn_coords is None:
                logger.info(
                    "Skipping shank %d during save because it has not been rendered",
                    shank_idx + 1,
                )
                continue
            alignment = state.active_alignment
            channel_locations_ras = (
                self.alignment_derived_data_service.compute_channel_locations(
                    ephysalign=shank_runtime.ephysalign,
                    feature=alignment.feature,
                    track=alignment.track,
                )
            )
            output_inputs[key] = (channel_locations_ras, shank_runtime.chn_coords)
            states_by_key[key] = state
        return output_inputs, states_by_key

    def display_qc_options(self) -> None:
        # If not histology don't show
        if not self.histology_exists:
            return

        self.qc_dialog.open()

    def qc_button_clicked(self) -> None:
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        align_qc = self.align_qc.currentText()
        ephys_qc = self.ephys_qc.currentText()
        ephys_desc = []
        for button in self.desc_buttons.buttons():
            if button.isChecked():
                ephys_desc.append(button.text())

        if ephys_qc != "Pass" and len(ephys_desc) == 0:
            QtWidgets.QMessageBox.warning(
                self, "Status", "You must select a reason for qc choice"
            )
            self.display_qc_options()
            return

        logger.warning(
            "Alyx QC upload is unavailable without ONE; saving local/DocDB "
            "alignment output instead."
        )
        self.complete_button_pressed_offline()

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
        notes_window = ephys_gui.PopupWindow(
            title="Session notes from Alyx", size=(200, 100), graphics=False
        )
        notes = QtWidgets.QTextEdit()
        notes.setReadOnly(True)
        notes.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        stream_runtime = self.runtime.active_stream_runtime
        session_notes = (
            stream_runtime.stream.session_notes if stream_runtime is not None else ""
        )
        notes.setText(session_notes)
        notes_window.layout.addWidget(notes)
        self.popup_manager.notes_window = notes_window

    def display_nearby_sessions(self) -> None:
        self._show_one_unsupported("Nearby sessions")

    def popup_closed(self, popup) -> None:
        self.popup_manager.remove_cluster_popup(popup)

    def popup_moved(self) -> None:
        self.activateWindow()

    def close_popups(self) -> None:
        self.popup_manager.close_cluster_popups()

    def minimise_popups(self) -> None:
        self.popup_manager.toggle_cluster_minimized()
        self.activateWindow()

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
        point_pos = point[0].pos()
        clust_idx = self.ephys_panel.cluster_index_for_plot_x(point_pos.x())
        if clust_idx is None:
            logger.error("Cannot show cluster detail: clicked point is not a cluster")
            return None

        detail = self.app.queries.active_cluster_detail(clust_idx)
        if detail is None:
            logger.error(
                "Cannot show cluster detail: active ephys stream is not loaded"
            )
            return None

        autocorr_plot = pg.PlotItem()
        autocorr_plot.setXRange(
            min=np.min(detail.t_autocorr),
            max=np.max(detail.t_autocorr),
        )
        autocorr_plot.setYRange(min=0, max=1.05 * np.max(detail.autocorr))
        self.set_axis(autocorr_plot, "bottom", label="T (ms)")
        self.set_axis(autocorr_plot, "left", label="Number of spikes")
        plot = pg.BarGraphItem(
            x=detail.t_autocorr,
            height=detail.autocorr,
            width=0.24,
            brush=self.bar_colour,
        )
        autocorr_plot.addItem(plot)

        template_plot = pg.PlotItem()
        plot = pg.PlotCurveItem()
        template_plot.setXRange(
            min=np.min(detail.t_template),
            max=np.max(detail.t_template),
        )
        self.set_axis(template_plot, "bottom", label="T (ms)")
        self.set_axis(template_plot, "left", label="Amplitude (a.u.)")
        plot.setData(
            x=detail.t_template,
            y=detail.template_waveform,
            pen=self.kpen_solid,
        )
        template_plot.addItem(plot)

        clust_layout = pg.GraphicsLayout()
        clust_layout.addItem(autocorr_plot, 0, 0)
        clust_layout.addItem(template_plot, 1, 0)

        clust_win = ephys_gui.PopupWindow(title=f"Cluster {detail.cluster_no}")
        clust_win.closed.connect(self.popup_closed)
        clust_win.moved.connect(self.popup_moved)
        clust_win.popup_widget.addItem(autocorr_plot, 0, 0)
        clust_win.popup_widget.addItem(template_plot, 1, 0)
        self.popup_manager.add_cluster_popup(clust_win)
        self.activateWindow()

        return detail.cluster_no

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
        # If no histology no point adding lines
        if not self.histology_exists:
            return

        if event.double():
            feature_y_um = self.ephys_panel.feature_y_from_scene(event.scenePos())
            if feature_y_um is None:
                return
            self.reference_lines.create_lines([feature_y_um])
            self._capture_pending_reference_lines()

    def on_mouse_hover(self, items) -> None:
        """
        Returns the pyqtgraph items that the mouse is hovering over. Used to identify reference
        lines so that they can be deleted
        """
        if len(items) > 1:
            self.reference_lines.clear_selection()
            if type(items[0]) == pg.InfiniteLine:
                self.reference_lines.select_line(items[0])
            elif (items[0] == self.fig_scale) & (type(items[1]) == pg.LinearRegionItem):
                scale_factor = self.histology_panel.scale_factor_for_region_item(
                    items[1]
                )
                if scale_factor is not None:
                    self.fig_scale_ax.setLabel(
                        "Scale Factor = " + str(np.around(scale_factor, 2))
                    )
            elif (items[0] == self.fig_hist) & (type(items[1]) == pg.LinearRegionItem):
                self.histology_panel.select_region(items[1])
            elif (items[0] == self.fig_hist_ref) & (
                type(items[1]) == pg.LinearRegionItem
            ):
                self.histology_panel.select_region(items[1])

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

    def remove_lines_points(self) -> None:
        """
        Removes all reference lines and scatter points from the ephys, histology and fit plots
        """
        self.reference_lines.remove_from_plots()

    def add_lines_points(self) -> None:
        """
        Adds all reference lines and scatter points from the ephys, histology and fit plots
        """
        self.reference_lines.add_to_plots()

    def _reattach_reference_lines(self) -> None:
        self.remove_lines_points()
        self.add_lines_points()

    def update_lines_points(self) -> None:
        """
        Updates position of reference lines on histology plot after fit has been applied. Also
        updates location of scatter point
        """
        self.reference_lines.sync_track_to_feature()

    def create_lines(self, positions, track_positions=None) -> None:
        self.reference_lines.create_lines(positions, track_positions)

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
