"""Reusable Qt widgets for the desktop GUI."""

from __future__ import annotations

from random import randrange

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets


class PopupWindow(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal(QtWidgets.QMainWindow)
    moved = QtCore.pyqtSignal()

    def __init__(self, title, parent=None, size=(300, 300), graphics=True) -> None:
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.resize(size[0], size[1])
        self.move(randrange(30) + 1000, randrange(30) + 200)
        if graphics:
            self.popup_widget = pg.GraphicsLayoutWidget()
        else:
            self.popup_widget = QtWidgets.QWidget()
            self.layout = QtWidgets.QGridLayout()
            self.popup_widget.setLayout(self.layout)
        self.setCentralWidget(self.popup_widget)
        self.setWindowTitle(title)
        self.show()

    def closeEvent(self, event) -> None:
        self.closed.emit(self)
        self.close()

    def leaveEvent(self, event) -> None:
        self.moved.emit()


class CheckableComboBox(QtWidgets.QComboBox):
    def __init__(self) -> None:
        super().__init__()
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QtGui.QStandardItemModel(self))

    def handleItemPressed(self, index) -> None:
        item = self.model().itemFromIndex(index)
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)
