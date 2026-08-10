import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

if platform.system() == "Darwin":
    if platform.release().split(".")[0] >= "20":
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread

import ephys_alignment_gui.desktop.setup as ephys_gui
from ephys_alignment_gui.desktop.display_ports import (
    desktop_display_ports_from_main_window,
)
from ephys_alignment_gui.desktop.displays import DesktopDisplays
from ephys_alignment_gui.desktop.popup_manager import DesktopPopupManager
from ephys_alignment_gui.desktop.views import DesktopViews
from ephys_alignment_gui.desktop.workbench import DesktopWorkbench
from ephys_alignment_gui.desktop.workbench_ports import (
    desktop_workbench_ports_from_main_window,
)
from ephys_alignment_gui.core.settings import (
    OUTPUT_ROOT_ENV_VAR,
    output_root_from_environment,
)
from ephys_alignment_gui.desktop.thread_worker import Worker
from ephys_alignment_gui.application.workflow import Requirement
from ephys_alignment_gui.application.workspace import AlignmentWorkspace

logger = logging.getLogger(__name__)

ANTS_DIMENSION = 3


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

        self._workspace = AlignmentWorkspace()
        self.app = self._workspace.app
        self.popup_manager = DesktopPopupManager()
        self.init_variables()
        self.offline: bool = offline
        self._empty_state_item: Any = None
        self.init_layout(self, offline=offline)
        self.displays = DesktopDisplays.create(
            app=self.app,
            ports=desktop_display_ports_from_main_window(self),
        )
        self.views = DesktopViews.from_main_window(self, displays=self.displays)
        self.selection_view = self.views.selection
        self.path_view = self.views.path
        self.depth_plot_view = self.views.depth
        self.shank_screen_view = self.views.shank_screen
        self.alignment_screen_view = self.views.alignment_screen
        self.export_view = self.views.export
        self.desktop_workbench = DesktopWorkbench.create(
            app=self.app,
            parent=self,
            views=self.views,
            displays=self.displays,
            ports=desktop_workbench_ports_from_main_window(self),
        )
        self.desktop_workbench.initialize_startup_stream_state()
        self.desktop_workbench.connect_events()
        self._set_default_output_root_from_environment()

        self.desktop_workbench.initialize_region_lookup(self.init_region_lookup)

    def closeEvent(self, event) -> None:
        """Disconnect desktop event subscriptions before the Qt window closes."""
        workbench = getattr(self, "desktop_workbench", None)
        if workbench is not None:
            workbench.disconnect_events()
        super().closeEvent(event)

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

        # Guide the user before any data is loaded / after clearing.
        if hasattr(self, "fig_img"):
            self._show_empty_state()

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

        self.displays.ephys.apply_view(
            view=view,
            axis_width=self.fig_ax_width,
            image_width=self.fig_img_width,
            line_width=self.fig_line_width,
            probe_width=self.fig_probe_width,
        )

    def save_plots(self, save_path=None) -> None:
        """
        Saves all plots from the GUI into folder
        """
        self.desktop_workbench.save_plots(save_path)

    """
    Plot functions
    """

    def _histology_available(self) -> bool:
        """Return whether histology runtime data is loaded."""
        return self.app.queries.workspace.histology_data_loaded()

    def plot_histology(self, fig=None, ax="left", movable=True) -> None:
        """Compatibility wrapper for aligned histology rendering."""
        if not self._histology_available():
            return
        self.desktop_workbench.render_active_aligned_histology(fig, movable=movable)

    def plot_histology_ref(self, fig=None, ax="right", movable=False) -> None:
        """Compatibility wrapper for reference histology rendering."""
        if not self._histology_available():
            return
        self.desktop_workbench.render_active_reference_histology(
            fig,
            movable=movable,
        )

    def plot_histology_nearby(self, fig=None, ax="right", movable=False) -> None:
        """Compatibility wrapper for nearby histology boundary rendering."""
        self.desktop_workbench.render_active_nearby_histology(fig, movable=movable)

    def _scale_factor_y_range(self) -> tuple[float, float]:
        y_min, y_max = self.fig_img.viewRange()[1]
        return float(y_min), float(y_max)

    def plot_scale_factor(self) -> None:
        """
        Plots the scale factor applied to brain regions along probe track, displayed
        alongside histology figure
        """

        # If no histology we can't do alignment
        if not self._histology_available():
            return

        self.desktop_workbench.render_active_scale_factor()

    def plot_fit(self) -> None:
        """
        Plots the scale factor and offset applied to channels along depth of probe track
        relative to orignal position of channels
        """

        # If no histology we can't do alignment
        if not self._histology_available():
            return

        self.desktop_workbench.render_active_fit()

    ### --------- interaction functions --------- ###
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

    def load_heavy_data(self) -> bool:
        """Load all heavy data - ephys, atlas, histology. Called once per session."""
        return self.desktop_workbench.load_heavy_data()

    def load_existing_alignments(self) -> bool:
        return self.desktop_workbench.load_existing_alignments()

    def set_mouse_root(self, mouse_root: Path) -> bool:
        """Point the GUI at a preprocessed mouse-root directory.

        Loads ``datapackage.json``, populates the session dropdown, and clears
        probe/shank state. The user then picks a session + probe, at which
        point channel info is read from the corresponding ephys ALF.

        :param mouse_root: Directory containing ``datapackage.json``.
        :return: ``True`` on success.
        """
        return self.desktop_workbench.set_mouse_root(mouse_root)

    def on_mouse_root_selected(self) -> bool:
        """Prompt for the mouse-root directory."""
        return self.desktop_workbench.select_mouse_root()

    def on_mouse_root_edited(self) -> None:
        """Triggered when the user finishes editing the mouse-root text field."""
        self.desktop_workbench.mouse_root_edited()

    def on_session_combobox_activated(self, _idx: int) -> None:
        """Populate the probe dropdown for the selected session."""
        self.desktop_workbench.session_selected()

    def on_probe_combobox_activated(self, _idx: int) -> None:
        """Select a probe: load channel info, populate shank list, derive output dir."""
        self.desktop_workbench.probe_selected()

    def on_use_docdb_changed(self, state) -> None:
        """Handler for Use DocDB checkbox state changes"""
        use_docdb = state == QtCore.Qt.Checked
        logger.info(f"Use DocDB: {use_docdb}")

    def on_load_data_button_pressed(self) -> None:
        """Triggered when user clicks 'Load Data' button"""
        self.desktop_workbench.load_data_button_pressed()

    def _ensure_output_directory_for_save(
        self, requirement: Requirement | None = None
    ) -> bool:
        """Require a save location before writing alignment outputs."""
        return self.desktop_workbench.ensure_output_directory_for_save(requirement)

    def set_save_root(self, save_root: Path) -> bool:
        """Set the save-root directory. Per-probe output lands under it."""
        return self.desktop_workbench.set_save_root(save_root)

    def on_output_folder_selected(self) -> bool:
        """Prompt the user for a save-root directory."""
        return self.desktop_workbench.select_output_root()

    def on_output_folder_edited(self) -> None:
        """Triggered when user finishes editing output_folder_line text field."""
        self.desktop_workbench.output_folder_edited()

    def on_shank_selected(self, idx) -> None:
        """Triggered when selecting shank from dropdown"""
        self.desktop_workbench.shank_selected(idx)

    def on_alignment_selected(self, idx) -> None:
        """Triggered when selecting alignment from dropdown"""
        self.desktop_workbench.alignment_selected(idx)

    def toggle_histology_button_pressed(self) -> None:
        self.desktop_workbench.toggle_histology_boundaries()

    def toggle_histology_map_button_pressed(self) -> None:
        self.desktop_workbench.toggle_region_annotation_source()

    def fit_button_pressed(self) -> None:
        """
        Triggered when fit button or Enter key pressed, applies scaling factor to brain regions
        according to locations of reference lines on ephys and histology plots. Updates all plots
        and indices after scaling has been applied
        """
        self.desktop_workbench.fit_button_pressed()

    def offset_button_pressed(
        self, _checked: bool = False, *, track_shift_m: float = 0.0
    ) -> None:
        """
        Triggered when offset button or o key pressed, applies offset to brain regions according to
        locations of probe tip line on histology plot. Updates all plots and indices after offset
        has been applied
        """
        self.desktop_workbench.offset_button_pressed(track_shift_m=track_shift_m)

    def movedown_button_pressed(self) -> None:
        """
        Triggered when Shift+down key pressed. Moves probe tip down by 50um and offsets data
        """
        self.desktop_workbench.movedown_button_pressed()

    def moveup_button_pressed(self) -> None:
        """
        Triggered when Shift+down key pressed. Moves probe tip up by 50um and offsets data
        """
        self.desktop_workbench.moveup_button_pressed()

    def toggle_labels_button_pressed(self) -> None:
        """
        Triggered when Shift+A key pressed. Shows/hides labels Allen atlas labels on brain regions
        in histology plots
        """
        self.desktop_workbench.toggle_labels()

    def toggle_line_button_pressed(self) -> None:
        """
        Triggered when Shift+L key pressed. Shows/hides reference lines on ephys and histology
        plots
        """
        self.desktop_workbench.toggle_reference_lines()

    def toggle_channel_button_pressed(self) -> None:
        """
        Triggered when Shift+C key pressed. Shows/hides channels, tip, and trajectory on slice image
        and perpendicular slice image
        """
        self.desktop_workbench.toggle_channels()

    def delete_line_button_pressed(self) -> None:
        """
        Triggered when mouse hovers over a reference line and shift+D keys are pressed.
        Deletes a reference line from the ephys and histology plots
        """

        self.desktop_workbench.delete_selected_reference_line()

    def describe_labels_pressed(self) -> None:
        self.desktop_workbench.describe_labels_pressed()

    def label_closed(self, popup) -> None:
        self.desktop_workbench.label_closed(popup)

    def label_moved(self) -> None:
        self.desktop_workbench.label_moved()

    def label_pressed(self, item) -> None:
        self.desktop_workbench.label_pressed(item)

    def next_button_pressed(self) -> None:
        """
        Triggered when right key pressed. Updates all plots and indices with next move. Ensures
        user cannot go past latest move
        """
        self.desktop_workbench.next_button_pressed()

    def prev_button_pressed(self) -> None:
        """
        Triggered when left key pressed. Updates all plots and indices with previous move.
        Ensures user cannot go back past the active edit-history buffer.
        """
        self.desktop_workbench.prev_button_pressed()

    def reset_button_pressed(self) -> None:
        """
        Triggered when reset button or Shift+R key pressed. Resets channel locations to orignal
        location
        """
        self.desktop_workbench.reset_button_pressed()

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
        self.desktop_workbench.save_alignment_outputs()

    def display_qc_options(self) -> None:
        self.desktop_workbench.display_qc_options()

    def qc_button_clicked(self) -> None:
        self.desktop_workbench.qc_button_clicked()

    def _selected_qc_descriptions(self) -> list[str]:
        """Return selected QC description labels."""
        ephys_desc = []
        if not hasattr(self, "desc_buttons"):
            return ephys_desc
        for button in self.desc_buttons.buttons():
            if button.isChecked():
                ephys_desc.append(button.text())
        return ephys_desc

    def reset_axis_button_pressed(self) -> None:
        self.desktop_workbench.reset_axis()

    def display_session_notes(self) -> None:
        self.desktop_workbench.display_session_notes()

    def display_nearby_sessions(self) -> None:
        self._show_one_unsupported("Nearby sessions")

    def popup_closed(self, popup) -> None:
        self.desktop_workbench.popup_closed(popup)

    def popup_moved(self) -> None:
        self.desktop_workbench.popup_moved()

    def close_popups(self) -> None:
        self.desktop_workbench.close_popups()

    def minimise_popups(self) -> None:
        self.desktop_workbench.minimise_popups()

    def lin_fit_option_changed(self, state) -> None:
        """
        Triggered when Linear fit checkbox state changes.
        Updates the flag and recomputes alignment by calling
        fit_button_pressed.
        """
        self.desktop_workbench.set_linear_fit_enabled(state != 0)

    def cluster_clicked(self, item, point):
        return self.desktop_workbench.cluster_clicked(item, point)

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
        self.desktop_workbench.on_mouse_double_clicked(event)

    def on_mouse_hover(self, items) -> None:
        """
        Returns the pyqtgraph items that the mouse is hovering over. Used to identify reference
        lines so that they can be deleted
        """
        self.desktop_workbench.on_mouse_hover(items)

    def tip_line_moved(self) -> None:
        """
        Triggered when dotted line indicating probe tip on self.fig_hist moved. Gets the y pos of
        probe tip line and ensures the probe top line is set to probe tip line y pos + 3840
        """
        self.desktop_workbench.sync_histology_top_to_tip()

    def top_line_moved(self) -> None:
        """
        Triggered when dotted line indicating probe top on self.fig_hist moved. Gets the y pos of
        probe top line and ensures the probe tip line is set to probe top line y pos - 3840
        """
        self.desktop_workbench.sync_histology_tip_to_top()


def viewer(probe_id, one=None, histology=False, spike_collection=None, title=None):
    """ """
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
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
