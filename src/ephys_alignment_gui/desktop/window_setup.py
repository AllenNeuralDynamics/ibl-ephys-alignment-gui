"""Top-level desktop window construction."""

from __future__ import annotations

from PyQt5 import QtWidgets

from ephys_alignment_gui.desktop.figure_setup import (
    configure_pyqtgraph,
    initialize_figures,
)
from ephys_alignment_gui.desktop.interaction_setup import (
    initialize_interaction_features,
)


def initialize_layout(window, offline=False) -> None:
    """Create the main window layout from focused desktop setup helpers."""
    configure_pyqtgraph()

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
