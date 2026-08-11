"""Top-level desktop window construction."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt5 import QtWidgets

from ephys_alignment_gui.desktop.shell.interaction_setup import (
    initialize_interaction_features,
)


def initialize_shell(window, offline=False) -> None:
    """Create the top-level Qt shell and interaction controls."""
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    window.resize(1600, 800)
    window.setWindowTitle("IBL Ephys Alignment GUI")
    window.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
    )
    window.offline = offline
    main_widget = QtWidgets.QWidget()
    window.setCentralWidget(main_widget)

    initialize_interaction_features(window)


def install_main_layout(window, *, displays) -> None:
    """Install display-owned widgets into the main window grid."""
    main_widget = window.centralWidget()
    if main_widget is None:
        main_widget = QtWidgets.QWidget()
        window.setCentralWidget(main_widget)

    main_layout = QtWidgets.QGridLayout()
    main_layout.addWidget(displays.ephys.area, 0, 0, 10, 1)
    main_layout.addWidget(displays.histology.area, 0, 1, 10, 1)
    main_layout.addLayout(window.interaction_layout1, 0, 2, 2, 1)
    main_layout.addWidget(displays.slice.area, 2, 2, 2, 1)
    main_layout.addLayout(window.interaction_layout2, 4, 2, 2, 1)
    main_layout.addWidget(displays.histology.fit_plot, 6, 2, 2, 1)
    main_layout.addLayout(window.interaction_layout3, 8, 2, 2, 1)
    main_layout.setColumnStretch(0, 4)
    main_layout.setColumnStretch(1, 3)
    main_layout.setColumnStretch(2, 3)

    main_widget.setLayout(main_layout)
