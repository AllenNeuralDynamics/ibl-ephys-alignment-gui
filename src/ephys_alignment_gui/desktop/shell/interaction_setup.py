"""Desktop interaction control construction."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from ibllib.qc.base import CriticalInsertionNote
except Exception:  # pragma: no cover - online QC path is not supported here.

    class CriticalInsertionNote:
        """Fallback for the unsupported legacy online-QC dialog path."""

        descriptions_gui: list[str] = []


def initialize_interaction_features(window) -> None:
    """Create interaction widgets and layout groups for the main window."""
    _initialize_alignment_buttons(window)
    _initialize_input_controls(window)
    _initialize_alignment_selection_controls(window)
    _initialize_output_controls(window)
    _initialize_interaction_layouts(window)
    _initialize_qc_dialog(window)


def _initialize_alignment_buttons(window) -> None:
    actions = window.shell_actions

    window.fit_button = QtWidgets.QPushButton("Fit")
    window.fit_button.clicked.connect(actions.fit_button_pressed)

    window.offset_button = QtWidgets.QPushButton("Offset")
    window.offset_button.clicked.connect(actions.offset_button_pressed)

    window.next_button = QtWidgets.QPushButton("Next")
    window.next_button.clicked.connect(actions.next_button_pressed)

    window.prev_button = QtWidgets.QPushButton("Previous")
    window.prev_button.clicked.connect(actions.prev_button_pressed)

    window.idx_string = QtWidgets.QLabel()
    window.tot_idx_string = QtWidgets.QLabel()

    window.reset_button = QtWidgets.QPushButton("Reset")
    window.reset_button.clicked.connect(actions.reset_button_pressed)

    window.complete_button = QtWidgets.QPushButton("Save")
    if not window.offline:
        window.complete_button.clicked.connect(actions.display_qc_options)
    else:
        window.complete_button.clicked.connect(actions.complete_button_pressed_offline)


def _initialize_input_controls(window) -> None:
    actions = window.shell_actions

    if not window.offline:
        window.subj_list = QtGui.QStandardItemModel()
        window.subj_combobox = QtWidgets.QComboBox()
        window.subj_combobox.setLineEdit(QtWidgets.QLineEdit())
        subj_completer = QtWidgets.QCompleter()
        subj_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        window.subj_combobox.setCompleter(subj_completer)
        window.subj_combobox.setModel(window.subj_list)
        window.subj_combobox.completer().setModel(window.subj_list)
        window.subj_combobox.activated.connect(actions.on_subject_selected)

        window.sess_list = QtGui.QStandardItemModel()
        window.sess_combobox = QtWidgets.QComboBox()
        window.sess_combobox.setModel(window.sess_list)
        window.sess_combobox.activated.connect(actions.on_session_selected)
        return

    window.mouse_root_line = QtWidgets.QLineEdit()
    window.mouse_root_button = QtWidgets.QToolButton()
    window.mouse_root_button.setText("Mouse Root")
    window.mouse_root_button.clicked.connect(actions.on_mouse_root_selected)
    window.mouse_root_line.editingFinished.connect(actions.on_mouse_root_edited)

    window.session_list = QtGui.QStandardItemModel()
    window.session_combobox = QtWidgets.QComboBox()
    window.session_combobox.setModel(window.session_list)
    window.session_combobox.activated.connect(actions.on_session_combobox_activated)

    window.probe_list = QtGui.QStandardItemModel()
    window.probe_combobox = QtWidgets.QComboBox()
    window.probe_combobox.setModel(window.probe_list)
    window.probe_combobox.activated.connect(actions.on_probe_combobox_activated)

    window.reload_folder_line = QtWidgets.QLineEdit()
    window.reload_folder_button = QtWidgets.QToolButton()
    window.reload_folder_button.setText("Load Alignments")
    window.reload_folder_button.clicked.connect(actions.load_existing_alignments)


def _initialize_alignment_selection_controls(window) -> None:
    actions = window.shell_actions

    window.shank_list = QtGui.QStandardItemModel()
    window.shank_combobox = QtWidgets.QComboBox()
    window.shank_combobox.setModel(window.shank_list)
    window.shank_combobox.activated.connect(actions.on_shank_selected)

    window.align_list = QtGui.QStandardItemModel()
    window.align_combobox = QtWidgets.QComboBox()
    window.align_combobox.setModel(window.align_list)
    window.align_combobox.activated.connect(actions.on_alignment_selected)


def _initialize_output_controls(window) -> None:
    actions = window.shell_actions

    window.output_folder_line = QtWidgets.QLineEdit()
    window.output_folder_button = QtWidgets.QToolButton()
    window.output_folder_button.setText("Output Directory")
    window.output_folder_button.clicked.connect(actions.on_output_folder_selected)
    window.output_folder_line.editingFinished.connect(actions.on_output_folder_edited)

    window.use_docdb_checkbox = QtWidgets.QCheckBox("DocDB")
    window.use_docdb_checkbox.setChecked(True)
    window.use_docdb_checkbox.stateChanged.connect(actions.on_use_docdb_changed)


def _initialize_interaction_layouts(window) -> None:
    if not window.offline:
        window.interaction_layout1 = QtWidgets.QHBoxLayout()
        window.interaction_layout1.addWidget(window.subj_combobox, stretch=1)
        window.interaction_layout1.addWidget(window.sess_combobox, stretch=2)
        window.interaction_layout1.addWidget(window.align_combobox, stretch=2)
    else:
        window.interaction_layout1 = _build_offline_input_layout(window)

    window.interaction_layout2 = _build_fit_navigation_layout(window)
    window.interaction_layout3 = _build_save_layout(window)


def _build_offline_input_layout(window) -> QtWidgets.QVBoxLayout:
    root_layout = QtWidgets.QHBoxLayout()
    root_layout.addWidget(window.mouse_root_button, stretch=0)
    root_layout.addWidget(window.mouse_root_line, stretch=2)

    selection_layout = QtWidgets.QHBoxLayout()
    selection_layout.addWidget(window.session_combobox, stretch=2)
    selection_layout.addWidget(window.probe_combobox, stretch=2)
    selection_layout.addWidget(window.shank_combobox, stretch=1)

    alignment_layout = QtWidgets.QHBoxLayout()
    alignment_layout.addWidget(window.align_combobox, stretch=1)
    alignment_layout.addWidget(window.use_docdb_checkbox, stretch=0)
    alignment_layout.addWidget(window.reload_folder_button, stretch=0)
    alignment_layout.addWidget(window.reload_folder_line, stretch=2)

    layout = QtWidgets.QVBoxLayout()
    layout.addLayout(root_layout)
    layout.addLayout(selection_layout)
    layout.addLayout(alignment_layout)
    return layout


def _build_fit_navigation_layout(window) -> QtWidgets.QVBoxLayout:
    fit_layout = QtWidgets.QHBoxLayout()
    fit_layout.addWidget(window.fit_button, stretch=1)
    fit_layout.addWidget(window.offset_button, stretch=1)
    fit_layout.addWidget(window.tot_idx_string, stretch=2)

    navigation_layout = QtWidgets.QHBoxLayout()
    navigation_layout.addWidget(window.prev_button, stretch=1)
    navigation_layout.addWidget(window.next_button, stretch=1)
    navigation_layout.addWidget(window.idx_string, stretch=2)

    layout = QtWidgets.QVBoxLayout()
    layout.addLayout(fit_layout)
    layout.addLayout(navigation_layout)
    return layout


def _build_save_layout(window) -> QtWidgets.QVBoxLayout:
    output_layout = QtWidgets.QHBoxLayout()
    output_layout.addWidget(window.output_folder_button, stretch=0)
    output_layout.addWidget(window.output_folder_line, stretch=1)

    action_layout = QtWidgets.QHBoxLayout()
    action_layout.addWidget(window.reset_button, stretch=0)
    action_layout.addWidget(window.complete_button, stretch=0)

    layout = QtWidgets.QVBoxLayout()
    layout.addLayout(output_layout)
    layout.addLayout(action_layout)
    return layout


def _initialize_qc_dialog(window) -> None:
    if window.offline:
        return

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
    for i, val in enumerate(CriticalInsertionNote.descriptions_gui):
        button = QtWidgets.QCheckBox(val)
        button.setCheckState(QtCore.Qt.Unchecked)
        window.desc_buttons.addButton(button, id=i)
        window.desc_layout.addWidget(button)

    window.desc_group.setLayout(window.desc_layout)

    window.qc_dialog = QtWidgets.QDialog(window)
    window.qc_dialog.setWindowTitle("QC assessment")
    window.qc_dialog.resize(300, 150)
    window.qc_dialog.accepted.connect(window.shell_actions.qc_button_clicked)
    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(window.qc_dialog.accept)
    button_box.rejected.connect(window.qc_dialog.reject)

    dialog_layout = QtWidgets.QVBoxLayout()
    dialog_layout.addWidget(align_qc_label)
    dialog_layout.addWidget(window.align_qc)
    dialog_layout.addWidget(ephys_qc_label)
    dialog_layout.addWidget(window.ephys_qc)
    dialog_layout.addWidget(window.desc_group)
    dialog_layout.addWidget(button_box)
    window.qc_dialog.setLayout(dialog_layout)
