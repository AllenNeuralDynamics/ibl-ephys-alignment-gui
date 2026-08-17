"""Modal progress dialog shown while background work settles on close."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

if TYPE_CHECKING:
    from PyQt5.QtGui import QCloseEvent


class DesktopShutdownDialog(QtWidgets.QDialog):
    """Application-modal, non-blocking shutdown progress dialog."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._close_allowed = False
        self.setWindowTitle("Closing")
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.resize(420, 140)

        self._summary_label = QtWidgets.QLabel("Cancelling background work...")
        self._summary_label.setWordWrap(True)
        self._detail_label = QtWidgets.QLabel(
            "The window will close when active load, preload, plot warmup, or save "
            "workers reach a cancellation checkpoint."
        )
        self._detail_label.setWordWrap(True)
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 0)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._summary_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._detail_label)
        self.setLayout(layout)

    def set_detail(self, detail: str) -> None:
        """Update the shutdown detail text."""
        self._detail_label.setText(detail)

    def close_dialog(self) -> None:
        """Allow and close the dialog during final teardown."""
        self._close_allowed = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Keep shutdown modal visible until workers have settled."""
        if self._close_allowed:
            event.accept()
            return
        event.ignore()
