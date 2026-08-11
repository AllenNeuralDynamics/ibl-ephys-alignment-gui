import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore, QtGui, QtWidgets

from ephys_alignment_gui.desktop.menu_setup import build_menu_bar
from ephys_alignment_gui.desktop.plot_elements import replace_axis

try:
    from ibllib.qc.base import CriticalInsertionNote
except Exception:  # pragma: no cover - online QC path is not supported here.

    class CriticalInsertionNote:
        """Fallback for the unsupported legacy online-QC dialog path."""

        descriptions_gui: list[str] = []


pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


def initialize_layout(window, offline=False) -> None:
    window.resize(1600, 800)
    window.setWindowTitle("IBL Ephys Alignment GUI")
    window.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
    )
    window.offline = offline
    main_widget = QtWidgets.QWidget()
    window.setCentralWidget(main_widget)

    initialize_interaction_features(window)
    initialize_figures(window)

    main_layout = QtWidgets.QGridLayout()
    main_layout.addWidget(window.fig_data_area, 0, 0, 10, 1)
    main_layout.addWidget(window.fig_hist_area, 0, 1, 10, 1)
    main_layout.addLayout(window.interaction_layout1, 0, 2, 2, 1)
    main_layout.addWidget(window.fig_slice_area, 2, 2, 2, 1)
    main_layout.addLayout(window.interaction_layout2, 4, 2, 2, 1)
    main_layout.addWidget(window.fig_fit, 6, 2, 2, 1)
    main_layout.addLayout(window.interaction_layout3, 8, 2, 2, 1)
    main_layout.setColumnStretch(0, 4)
    main_layout.setColumnStretch(1, 3)
    main_layout.setColumnStretch(2, 3)

    main_widget.setLayout(main_layout)


def initialize_menubar(window) -> None:
    build_menu_bar(window)


def initialize_interaction_features(window) -> None:
    """
    Create all interaction widgets that will be added to the GUI
    """
    # Button to apply interpolation
    window.fit_button = QtWidgets.QPushButton("Fit")
    window.fit_button.clicked.connect(window.fit_button_pressed)
    # Button to apply offset
    window.offset_button = QtWidgets.QPushButton("Offset")
    window.offset_button.clicked.connect(window.offset_button_pressed)
    # Button to go to next move
    window.next_button = QtWidgets.QPushButton("Next")
    window.next_button.clicked.connect(window.next_button_pressed)
    # Button to go to previous move
    window.prev_button = QtWidgets.QPushButton("Previous")
    window.prev_button.clicked.connect(window.prev_button_pressed)
    # String to display current move index
    window.idx_string = QtWidgets.QLabel()
    # String to display total number of moves
    window.tot_idx_string = QtWidgets.QLabel()
    # Button to reset GUI to initial state
    window.reset_button = QtWidgets.QPushButton("Reset")
    window.reset_button.clicked.connect(window.reset_button_pressed)
    # Button to upload final state to Alyx/ to local file
    window.complete_button = QtWidgets.QPushButton("Save")
    if not window.offline:
        window.complete_button.clicked.connect(window.display_qc_options)
    else:
        window.complete_button.clicked.connect(window.complete_button_pressed_offline)

    if not window.offline:
        # If offline mode is False, read in Subject and Session options from Alyx
        # Drop down list to choose subject
        window.subj_list = QtGui.QStandardItemModel()
        window.subj_combobox = QtWidgets.QComboBox()
        # Add line edit and completer to be able to search for subject
        window.subj_combobox.setLineEdit(QtWidgets.QLineEdit())
        subj_completer = QtWidgets.QCompleter()
        subj_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        window.subj_combobox.setCompleter(subj_completer)
        window.subj_combobox.setModel(window.subj_list)
        window.subj_combobox.completer().setModel(window.subj_list)
        window.subj_combobox.activated.connect(window.on_subject_selected)

        # Drop down list to choose session
        window.sess_list = QtGui.QStandardItemModel()
        window.sess_combobox = QtWidgets.QComboBox()
        window.sess_combobox.setModel(window.sess_list)
        window.sess_combobox.activated.connect(window.on_session_selected)
    else:
        # Offline mode: user points the GUI at a preprocessed mouse-root
        # directory (containing ``datapackage.json``) and selects the
        # session + probe from dropdowns populated from the manifest.
        window.mouse_root_line = QtWidgets.QLineEdit()
        window.mouse_root_button = QtWidgets.QToolButton()
        window.mouse_root_button.setText("Mouse Root")
        window.mouse_root_button.clicked.connect(window.on_mouse_root_selected)
        window.mouse_root_line.editingFinished.connect(window.on_mouse_root_edited)

        window.session_list = QtGui.QStandardItemModel()
        window.session_combobox = QtWidgets.QComboBox()
        window.session_combobox.setModel(window.session_list)
        window.session_combobox.activated.connect(window.on_session_combobox_activated)

        window.probe_list = QtGui.QStandardItemModel()
        window.probe_combobox = QtWidgets.QComboBox()
        window.probe_combobox.setModel(window.probe_list)
        window.probe_combobox.activated.connect(window.on_probe_combobox_activated)

        window.reload_folder_line = QtWidgets.QLineEdit()
        window.reload_folder_button = QtWidgets.QToolButton()
        window.reload_folder_button.setText("Load Alignments")
        window.reload_folder_button.clicked.connect(window.load_existing_alignments)

    # Drop down list to select shank
    window.shank_list = QtGui.QStandardItemModel()
    window.shank_combobox = QtWidgets.QComboBox()
    window.shank_combobox.setModel(window.shank_list)
    window.shank_combobox.activated.connect(window.on_shank_selected)

    # Drop down list to select previous alignment (NEW)
    window.align_list = QtGui.QStandardItemModel()
    window.align_combobox = QtWidgets.QComboBox()
    window.align_combobox.setModel(window.align_list)
    window.align_combobox.activated.connect(window.on_alignment_selected)

    window.output_folder_line = QtWidgets.QLineEdit()
    window.output_folder_button = QtWidgets.QToolButton()
    window.output_folder_button.setText("Output Directory")
    window.output_folder_button.clicked.connect(window.on_output_folder_selected)
    window.output_folder_line.editingFinished.connect(window.on_output_folder_edited)

    # After output_folder_button creation (around line 573):
    window.use_docdb_checkbox = QtWidgets.QCheckBox("DocDB")
    window.use_docdb_checkbox.setChecked(True)  # Default: try DocDB
    window.use_docdb_checkbox.stateChanged.connect(window.on_use_docdb_changed)

    window.load_data_button = QtWidgets.QToolButton()
    window.load_data_button.setText("Load Data")
    window.load_data_button.setEnabled(False)  # Disabled until input path is set
    window.load_data_button.clicked.connect(window.on_load_data_button_pressed)

    # Arrange interaction features into three different layout groups
    # Group 1 -- loading data
    if not window.offline:
        window.interaction_layout1 = QtWidgets.QHBoxLayout()
        window.interaction_layout1.addWidget(window.subj_combobox, stretch=1)
        window.interaction_layout1.addWidget(window.sess_combobox, stretch=2)
        window.interaction_layout1.addWidget(window.align_combobox, stretch=2)
        window.interaction_layout1.addWidget(window.data_button, stretch=1)
    else:
        interact_1_h_1 = QtWidgets.QHBoxLayout()
        interact_1_h_1.addWidget(window.mouse_root_button, stretch=0)
        interact_1_h_1.addWidget(window.mouse_root_line, stretch=2)

        interact_1_h_2 = QtWidgets.QHBoxLayout()
        interact_1_h_2.addWidget(window.session_combobox, stretch=2)
        interact_1_h_2.addWidget(window.probe_combobox, stretch=2)
        interact_1_h_2.addWidget(window.shank_combobox, stretch=1)
        interact_1_h_2.addWidget(window.load_data_button, stretch=0)

        interact_1_h_3 = QtWidgets.QHBoxLayout()
        interact_1_h_3.addWidget(window.align_combobox, stretch=1)
        interact_1_h_3.addWidget(window.use_docdb_checkbox, stretch=0)
        interact_1_h_3.addWidget(window.reload_folder_button, stretch=0)
        interact_1_h_3.addWidget(window.reload_folder_line, stretch=2)

        window.interaction_layout1 = QtWidgets.QVBoxLayout()
        window.interaction_layout1.addLayout(interact_1_h_1)
        window.interaction_layout1.addLayout(interact_1_h_2)
        window.interaction_layout1.addLayout(interact_1_h_3)

    # Group 2 -- fitting and navigation
    interact_2_h_1 = QtWidgets.QHBoxLayout()
    interact_2_h_1.addWidget(window.fit_button, stretch=1)
    interact_2_h_1.addWidget(window.offset_button, stretch=1)
    interact_2_h_1.addWidget(window.tot_idx_string, stretch=2)
    interact_2_h_2 = QtWidgets.QHBoxLayout()
    interact_2_h_2.addWidget(window.prev_button, stretch=1)
    interact_2_h_2.addWidget(window.next_button, stretch=1)
    interact_2_h_2.addWidget(window.idx_string, stretch=2)
    window.interaction_layout2 = QtWidgets.QVBoxLayout()
    window.interaction_layout2.addLayout(interact_2_h_1)
    window.interaction_layout2.addLayout(interact_2_h_2)

    # Group 3 -- saving data
    interact_3_h_1 = QtWidgets.QHBoxLayout()
    interact_3_h_1.addWidget(window.output_folder_button, stretch=0)
    interact_3_h_1.addWidget(window.output_folder_line, stretch=1)
    interact_3_h_2 = QtWidgets.QHBoxLayout()
    interact_3_h_2.addWidget(window.reset_button, stretch=0)
    interact_3_h_2.addWidget(window.complete_button, stretch=0)
    window.interaction_layout3 = QtWidgets.QVBoxLayout()
    window.interaction_layout3.addLayout(interact_3_h_1)
    window.interaction_layout3.addLayout(interact_3_h_2)

    # Pop up dialog for qc results to datajoint, only for online mode
    if not window.offline:
        align_qc_label = QtWidgets.QLabel("Confidence of alignment")
        window.align_qc = QtWidgets.QComboBox()
        window.align_qc.addItems(["High", "Medium", "Low"])
        ephys_qc_label = QtWidgets.QLabel("QC for ephys recording")
        window.ephys_qc = QtWidgets.QComboBox()
        window.ephys_qc.addItems(["Pass", "Warning", "Critical"])

        window.desc_buttons = QtWidgets.QButtonGroup()
        window.desc_group = QtWidgets.QGroupBox("Describe problem with recording")
        window.desc_layout = QtWidgets.QVBoxLayout()
        window.desc_layout.setSpacing(5)
        window.desc_buttons.setExclusive(False)
        options = CriticalInsertionNote.descriptions_gui
        for i, val in enumerate(options):
            button = QtWidgets.QCheckBox(val)
            button.setCheckState(QtCore.Qt.Unchecked)

            window.desc_buttons.addButton(button, id=i)
            window.desc_layout.addWidget(button)

        window.desc_group.setLayout(window.desc_layout)

        window.qc_dialog = QtWidgets.QDialog(window)
        window.qc_dialog.setWindowTitle("QC assessment")
        window.qc_dialog.resize(300, 150)
        window.qc_dialog.accepted.connect(window.qc_button_clicked)
        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttonBox.accepted.connect(window.qc_dialog.accept)
        buttonBox.rejected.connect(window.qc_dialog.reject)
        #
        dialog_layout = QtWidgets.QVBoxLayout()
        dialog_layout.addWidget(align_qc_label)
        dialog_layout.addWidget(window.align_qc)
        dialog_layout.addWidget(ephys_qc_label)
        dialog_layout.addWidget(window.ephys_qc)
        dialog_layout.addWidget(window.desc_group)
        dialog_layout.addWidget(buttonBox)
        window.qc_dialog.setLayout(dialog_layout)


def initialize_region_lookup(window, allen) -> None:
    """
    Create Allen Atlas structure tree
    """

    # Remove the first row which corresponds to 'Void'
    allen = allen.drop([0]).reset_index(drop=True)

    # Find the parent path of each structure by removing the structure id from path
    def parent_path(struct_path):
        return struct_path.rsplit("/", 2)[0] + "/"

    allen["parent_path"] = allen["structure_id_path"].apply(parent_path)

    # Create standard model view
    window.struct_list = QtGui.QStandardItemModel()
    window.struct_view = QtWidgets.QTreeView()
    window.struct_view.setModel(window.struct_list)
    window.struct_view.clicked.connect(window.label_pressed)

    # Defin
    window.struct_view.setHeaderHidden(True)
    unique_levels = np.unique(allen["depth"]).astype(int)
    parent_info = {}
    idx = np.where(allen["depth"] == unique_levels[0])[0]
    item = QtGui.QStandardItem(allen["acronym"][idx[0]] + ": " + allen["name"][idx[0]])
    icon = QtGui.QPixmap(20, 20)
    icon.fill(QtGui.QColor("#" + allen["color_hex_triplet"][idx[0]]))
    item.setIcon(QtGui.QIcon(icon))
    item.setAccessibleText(str(allen["id"][idx[0]]))
    item.setEditable(False)
    window.struct_list.appendRow(item)
    parent_info.update({allen["structure_id_path"][idx[0]]: item})

    for level in unique_levels[1:]:
        idx_levels = np.where(allen["depth"] == level)[0]
        for idx in idx_levels:
            parent = allen["parent_path"][idx]
            parent_item = parent_info[parent]
            item = QtGui.QStandardItem(
                allen["acronym"][idx] + ": " + allen["name"][idx]
            )
            icon.fill(QtGui.QColor("#" + allen["color_hex_triplet"][idx]))
            item.setIcon(QtGui.QIcon(icon))
            item.setAccessibleText(str(allen["id"][idx]))
            item.setEditable(False)
            parent_item.appendRow(item)
            parent_info.update({allen["structure_id_path"][idx]: item})

    window.struct_description = QtWidgets.QTextEdit()


def initialize_figures(window) -> None:
    """
    Create all figures that will be added to the GUI
    """
    depth_view = window.app.queries.workspace.depth_view_settings()
    y_min, y_max = depth_view.plot_y_range_um
    # Lists to store the position of probe top and tip
    window.probe_top_lines = []
    window.probe_tip_lines = []

    # Figures to show ephys data
    # 2D scatter/ image plot
    window.fig_img = pg.PlotItem()
    window.fig_img.setYRange(min=y_min, max=y_max, padding=window.pad)
    window.fig_img.setMouseEnabled(x=False, y=True)
    window.probe_tip_lines.append(
        window.fig_img.addLine(y=depth_view.probe_tip_um, pen=window.kpen_dot, z=50)
    )
    window.probe_top_lines.append(
        window.fig_img.addLine(y=depth_view.probe_top_um, pen=window.kpen_dot, z=50)
    )
    window.set_axis(window.fig_img, "bottom")
    window.fig_data_ax = window.set_axis(
        window.fig_img, "left", label="Distance from probe tip (uV)"
    )

    window.fig_img_cb = pg.PlotItem()
    window.fig_img_cb.setMaximumHeight(70)
    window.fig_img_cb.setMouseEnabled(x=False, y=False)
    window.set_axis(window.fig_img_cb, "bottom", show=False)
    window.set_axis(window.fig_img_cb, "left", pen="w")
    window.set_axis(window.fig_img_cb, "top", pen="w")

    # 1D line plot
    window.fig_line = pg.PlotItem()
    window.fig_line.setMouseEnabled(x=False, y=True)
    window.fig_line.setYRange(min=y_min, max=y_max, padding=window.pad)
    window.probe_tip_lines.append(
        window.fig_line.addLine(y=depth_view.probe_tip_um, pen=window.kpen_dot, z=50)
    )
    window.probe_top_lines.append(
        window.fig_line.addLine(y=depth_view.probe_top_um, pen=window.kpen_dot, z=50)
    )
    window.set_axis(window.fig_line, "bottom")
    window.set_axis(window.fig_line, "left", show=False)

    # 2D probe plot
    window.fig_probe = pg.PlotItem()
    window.fig_probe.setMouseEnabled(x=False, y=False)
    window.fig_probe.setMaximumWidth(50)
    window.fig_probe.setYRange(min=y_min, max=y_max, padding=window.pad)
    window.probe_tip_lines.append(
        window.fig_probe.addLine(y=depth_view.probe_tip_um, pen=window.kpen_dot, z=50)
    )
    window.probe_top_lines.append(
        window.fig_probe.addLine(y=depth_view.probe_top_um, pen=window.kpen_dot, z=50)
    )
    window.set_axis(window.fig_probe, "bottom", pen="w")
    window.set_axis(window.fig_probe, "left", show=False)

    window.fig_probe_cb = pg.PlotItem()
    window.fig_probe_cb.setMouseEnabled(x=False, y=False)
    window.fig_probe_cb.setMaximumHeight(70)
    window.set_axis(window.fig_probe_cb, "bottom", show=False)
    window.set_axis(window.fig_probe_cb, "left", show=False)
    window.set_axis(window.fig_probe_cb, "top", pen="w")

    # Add img plot, line plot, probe plot, img colourbar and probe colourbar to a graphics
    # layout widget so plots can be arranged and moved easily
    window.fig_data_area = pg.GraphicsLayoutWidget()
    window.fig_data_area.scene().sigMouseClicked.connect(window.on_mouse_double_clicked)
    window.fig_data_area.scene().sigMouseHover.connect(window.on_mouse_hover)
    window.fig_data_layout = pg.GraphicsLayout()

    window.fig_data_layout.addItem(window.fig_img_cb, 0, 0)
    window.fig_data_layout.addItem(window.fig_probe_cb, 0, 1, 1, 2)
    window.fig_data_layout.addItem(window.fig_img, 1, 0)
    window.fig_data_layout.addItem(window.fig_line, 1, 1)
    window.fig_data_layout.addItem(window.fig_probe, 1, 2)
    window.fig_data_layout.layout.setColumnStretchFactor(0, 6)
    window.fig_data_layout.layout.setColumnStretchFactor(1, 1)
    window.fig_data_layout.layout.setColumnStretchFactor(2, 1)
    window.fig_data_layout.layout.setRowStretchFactor(0, 1)
    window.fig_data_layout.layout.setRowStretchFactor(1, 10)

    window.fig_data_area.addItem(window.fig_data_layout)

    # Figures to show histology data
    # Histology figure that will be updated with user input
    window.fig_hist = pg.PlotItem()
    window.fig_hist.setContentsMargins(0, 0, 0, 0)
    window.fig_hist.setMouseEnabled(x=False)
    window.fig_hist.setYRange(min=y_min, max=y_max, padding=window.pad)
    window.set_axis(window.fig_hist, "bottom", pen="w")

    window.fig_img.setYLink(window.fig_line)
    window.fig_img.setYLink(window.fig_hist)
    window.fig_line.setYLink(window.fig_hist)
    window.fig_probe.setYLink(window.fig_img)

    replace_axis(window.fig_hist)
    window.ax_hist = window.set_axis(window.fig_hist, "left", pen=None)
    window.ax_hist.setWidth(0)
    # Region labels will be added as TextItems in plot_histology()

    window.fig_scale = pg.PlotItem()
    window.fig_scale.setMaximumWidth(50)
    window.fig_scale.setMouseEnabled(x=False)
    window.scale_label = pg.LabelItem(color="k")
    window.set_axis(window.fig_scale, "bottom", pen="w")
    window.set_axis(window.fig_scale, "left", show=False)
    (window.fig_scale).setYLink(window.fig_hist)

    # Figure that will show scale factor of histology boundaries
    window.fig_scale_cb = pg.PlotItem()
    window.fig_scale_cb.setMouseEnabled(x=False, y=False)
    window.fig_scale_cb.setMaximumHeight(70)
    window.set_axis(window.fig_scale_cb, "bottom", show=False)
    window.set_axis(window.fig_scale_cb, "left", show=False)
    window.fig_scale_ax = window.set_axis(window.fig_scale_cb, "top", pen="w")
    window.set_axis(window.fig_scale_cb, "right", show=False)

    # Histology figure that will remain at initial state for reference
    window.fig_hist_ref = pg.PlotItem()
    window.fig_hist_ref.setMouseEnabled(x=False)
    window.fig_hist_ref.setYRange(min=y_min, max=y_max, padding=window.pad)
    # Y-link to fig_hist so scrolling/zooming the track-space view stays
    # synchronised with the feature-space plots.
    window.fig_hist_ref.setYLink(window.fig_hist)
    window.set_axis(window.fig_hist_ref, "bottom", pen="w")
    window.set_axis(window.fig_hist_ref, "left", show=False)
    replace_axis(window.fig_hist_ref, orientation="right", pos=(2, 2))
    window.ax_hist_ref = window.set_axis(window.fig_hist_ref, "right", pen=None)
    window.ax_hist_ref.setWidth(0)
    # Region labels will be added as TextItems in plot_histology_ref()

    # Perpendicular histology slice plot
    window.fig_hist_perp = pg.PlotItem()
    window.fig_hist_perp.setContentsMargins(0, 0, 0, 0)
    window.fig_hist_perp.setMouseEnabled(x=False)
    window.fig_hist_perp.setYRange(min=y_min, max=y_max, padding=window.pad)
    window.set_axis(window.fig_hist_perp, "bottom", pen="w")
    window.set_axis(window.fig_hist_perp, "left", show=False)
    window.fig_hist_perp.setYLink(window.fig_hist)

    window.fig_hist_area = pg.GraphicsLayoutWidget()
    window.fig_hist_area.setMouseTracking(True)
    window.fig_hist_area.scene().sigMouseClicked.connect(window.on_mouse_double_clicked)
    window.fig_hist_area.scene().sigMouseHover.connect(window.on_mouse_hover)

    window.fig_hist_extra_yaxis = pg.PlotItem()
    window.fig_hist_extra_yaxis.setMouseEnabled(x=False, y=False)
    window.fig_hist_extra_yaxis.setMaximumWidth(2)
    window.fig_hist_extra_yaxis.setYRange(min=y_min, max=y_max, padding=window.pad)

    window.set_axis(window.fig_hist_extra_yaxis, "bottom", pen="w")
    window.ax_hist2 = window.set_axis(window.fig_hist_extra_yaxis, "left", pen=None)
    window.ax_hist2.setWidth(10)

    window.fig_hist_layout = pg.GraphicsLayout()
    window.fig_hist_layout.addItem(
        window.fig_scale_cb, 0, 0, 1, 5
    )  # Span all 5 columns
    window.fig_hist_layout.addItem(window.fig_hist_extra_yaxis, 1, 0)
    window.fig_hist_layout.addItem(window.fig_hist, 1, 1)
    window.fig_hist_layout.addItem(
        window.fig_hist_perp, 1, 2
    )  # NEW: Perpendicular slice
    window.fig_hist_layout.addItem(window.fig_scale, 1, 3)  # Moved from column 2
    window.fig_hist_layout.addItem(window.fig_hist_ref, 1, 4)  # Moved from column 3
    window.fig_hist_layout.layout.setColumnStretchFactor(0, 1)
    window.fig_hist_layout.layout.setColumnStretchFactor(1, 4)
    window.fig_hist_layout.layout.setColumnStretchFactor(2, 5)  # Perpendicular slice
    window.fig_hist_layout.layout.setColumnStretchFactor(3, 1)  # Scale
    window.fig_hist_layout.layout.setColumnStretchFactor(4, 4)  # Ref
    window.fig_hist_layout.layout.setRowStretchFactor(0, 1)
    window.fig_hist_layout.layout.setRowStretchFactor(1, 10)
    window.fig_hist_area.addItem(window.fig_hist_layout)

    # Figure to show coronal slice through the brain
    window.fig_slice_area = pg.GraphicsLayoutWidget()
    window.fig_slice_layout = pg.GraphicsLayout()
    window.fig_slice_hist_alt = pg.ViewBox()
    window.fig_slice = pg.ViewBox()
    window.fig_slice_layout.addItem(window.fig_slice, 0, 0)
    window.fig_slice_layout.addItem(window.fig_slice_hist_alt, 0, 1)
    window.fig_slice_layout.layout.setColumnStretchFactor(0, 3)
    window.fig_slice_layout.layout.setColumnStretchFactor(1, 1)
    window.fig_slice_area.addItem(window.fig_slice_layout)
    window.slice_item = window.fig_slice_hist_alt

    # Figure to show fit and offset applied by user
    window.fig_fit = pg.PlotWidget(background="w")
    window.fig_fit.setMouseEnabled(x=False, y=False)
    window.fig_fit_exporter = pg.exporters.ImageExporter(window.fig_fit.plotItem)
    window.fig_fit.sigDeviceRangeChanged.connect(
        lambda *args: position_linear_fit_checkbox(window)
    )
    view_min, view_max = depth_view.view_range_um
    window.fig_fit.setXRange(min=view_min, max=view_max)
    window.fig_fit.setYRange(min=view_min, max=view_max)
    # Each point on the fit plot is one user-placed reference line (plus
    # the two implicit endpoints at probe tip / top). X = where the line
    # was placed on the ephys side, Y = where it was placed on the
    # histology side; both are depths along the probe track measured
    # from the first electrode.
    window.set_axis(window.fig_fit, "bottom", label="Ephys reference depth (μm)")
    window.set_axis(window.fig_fit, "left", label="Atlas reference depth (μm)")
    plot = pg.PlotCurveItem()
    plot.setData(
        x=depth_view.fit_depth_um,
        y=depth_view.fit_depth_um,
        pen=window.kpen_dot,
    )
    window.fit_plot = pg.PlotCurveItem(pen=window.bpen_solid)
    window.fit_scatter = pg.ScatterPlotItem(size=7, symbol="o", brush="w", pen="b")
    window.fit_plot_lin = pg.PlotCurveItem(pen=window.rpen_dot)
    window.fig_fit.addItem(plot)
    window.fig_fit.addItem(window.fit_plot)
    window.fig_fit.addItem(window.fit_plot_lin)
    window.fig_fit.addItem(window.fit_scatter)

    window.lin_fit_option = QtWidgets.QCheckBox("Linear fit", window.fig_fit)
    window.lin_fit_option.setChecked(window.app.queries.workspace.linear_fit_enabled())
    window.lin_fit_option.stateChanged.connect(window.lin_fit_option_changed)
    position_linear_fit_checkbox(window)


def position_linear_fit_checkbox(window) -> None:
    # fig_width = window.fig_fit_exporter.getTargetRect().width()
    # fig_height = window.fig_fit_exporter.getTargetRect().width()
    window.lin_fit_option.move(70, 10)
