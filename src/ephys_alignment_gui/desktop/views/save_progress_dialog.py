"""Desktop dialog for edited-alignment save progress."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from ephys_alignment_gui.core.document import AlignmentKey

if TYPE_CHECKING:
    from PyQt5.QtGui import QCloseEvent


class DesktopSaveProgressDialog(QtWidgets.QDialog):
    """Modeless progress dialog for save-all alignment output transactions."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        cancel_requested: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._cancel_requested = cancel_requested
        self._cancel_enabled = False
        self._finished = False
        self._items_by_key: dict[AlignmentKey, QtWidgets.QListWidgetItem] = {}

        self.setWindowTitle("Saving Alignments")
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(560, 360)

        self._summary_label = QtWidgets.QLabel("Preparing save...")
        self._summary_label.setWordWrap(True)
        self._current_label = QtWidgets.QLabel("")
        self._current_label.setWordWrap(True)
        self._progress_bar = QtWidgets.QProgressBar()
        self._target_list = QtWidgets.QListWidget()
        self._target_list.setAlternatingRowColors(True)
        self._action_button = QtWidgets.QPushButton("Cancel")
        self._action_button.clicked.connect(self._action_clicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._summary_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._current_label)
        layout.addWidget(self._target_list, stretch=1)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self._action_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.set_cancel_enabled(False)

    def set_cancel_callback(self, callback: Callable[[], None] | None) -> None:
        """Set the callback invoked when the user requests cancellation."""
        self._cancel_requested = callback

    def set_cancel_enabled(self, enabled: bool) -> None:
        """Enable or disable user cancellation for the active phase."""
        self._cancel_enabled = enabled
        if self._finished:
            self._action_button.setText("Close")
            self._action_button.setEnabled(True)
            return
        self._action_button.setText("Cancel")
        self._action_button.setEnabled(enabled)

    def show_started(
        self,
        targets: Sequence[AlignmentKey],
        *,
        message: str,
        cancel_enabled: bool = False,
    ) -> None:
        """Show a fresh save-progress target list."""
        self._finished = False
        self._items_by_key.clear()
        self._target_list.clear()
        self._summary_label.setText(message)
        self._current_label.setText("")
        self._progress_bar.setRange(0, max(len(targets), 1))
        self._progress_bar.setValue(0)
        for key in targets:
            item = QtWidgets.QListWidgetItem(f"Pending - {_describe_key(key)}")
            self._items_by_key[key] = item
            self._target_list.addItem(item)
        self.set_cancel_enabled(cancel_enabled)
        self.show()
        self.raise_()
        self._process_events()

    def update_progress(
        self,
        *,
        key: AlignmentKey | None,
        phase_label: str,
        status_label: str,
        completed: int,
        total: int,
        message: str,
    ) -> None:
        """Update dialog labels and one target row."""
        if key is None and status_label == "Running":
            self._progress_bar.setRange(0, 0)
        elif total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(max(0, min(completed, total)))
        if key is None:
            self._summary_label.setText(f"{phase_label}: {status_label}")
        else:
            self._summary_label.setText(f"{phase_label}: {completed}/{max(total, 1)}")
        self._current_label.setText(message)
        if key is not None:
            item = self._items_by_key.get(key)
            if item is None:
                item = QtWidgets.QListWidgetItem("")
                self._items_by_key[key] = item
                self._target_list.addItem(item)
            item.setText(f"{status_label} - {_describe_key(key)}")
            self._target_list.scrollToItem(item)
        self._process_events()

    def show_finished(self, message: str, *, success: bool) -> None:
        """Render terminal save state and leave the dialog closable."""
        self._finished = True
        total = max(self._progress_bar.maximum(), 1)
        self._progress_bar.setRange(0, total)
        if success:
            self._progress_bar.setValue(total)
        self._summary_label.setText("Save complete" if success else "Save failed")
        self._current_label.setText(message)
        self.set_cancel_enabled(False)
        self.show()
        self.raise_()
        self._process_events()

    def close_dialog(self) -> None:
        """Close the dialog regardless of active state."""
        self._finished = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Prevent accidental close while non-cancelable save work is running."""
        if self._finished:
            event.accept()
            return
        if self._cancel_enabled and self._cancel_requested is not None:
            self._cancel_requested()
        event.ignore()

    def _action_clicked(self) -> None:
        if self._finished:
            self.close()
            return
        if self._cancel_enabled and self._cancel_requested is not None:
            self._cancel_requested()

    @staticmethod
    def _process_events() -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()


def _describe_key(key: AlignmentKey) -> str:
    return (
        f"{key.recording_id} / {key.ephys_collection} / shank {key.shank_idx + 1}"
    )
