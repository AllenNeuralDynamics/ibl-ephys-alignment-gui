"""Desktop context manager for long-running operations."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = Any


class BusyContext:
    """Context manager for long-running operations with visual feedback."""

    def __init__(
        self,
        window: Any,
        message: str | None = None,
        success_message: str | None = None,
        error_message: str | None = None,
        disable_widgets: list[Any] | Any | None = None,
        success_timeout_ms: int = 3000,
        error_timeout_ms: int = 5000,
    ) -> None:
        self.window = window
        self.message = message
        self.success_message = success_message
        self.error_message = error_message
        self.success_timeout_ms = success_timeout_ms
        self.error_timeout_ms = error_timeout_ms

        if disable_widgets is None:
            self.disable_widgets = []
        elif isinstance(disable_widgets, list):
            self.disable_widgets = disable_widgets
        else:
            self.disable_widgets = [disable_widgets]

        self.widget_states: dict[Any, bool] = {}

    def __enter__(self) -> Self:
        """Enter busy state: set cursor, show message, disable widgets."""
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if self.message:
            self.window.statusBar().showMessage(self.message)

        for widget in self.disable_widgets:
            self.widget_states[widget] = widget.isEnabled()
            widget.setEnabled(False)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit busy state: restore cursor, widgets, and status message."""
        QApplication.restoreOverrideCursor()

        for widget, was_enabled in self.widget_states.items():
            widget.setEnabled(was_enabled)

        if exc_type is not None:
            if self.error_message is None:
                error_msg = f"Error: {exc_val}"
            else:
                error_msg = self.error_message
            self.window.statusBar().showMessage(error_msg, self.error_timeout_ms)
        elif self.success_message:
            self.window.statusBar().showMessage(
                self.success_message,
                self.success_timeout_ms,
            )
        else:
            self.window.statusBar().clearMessage()

        return False

    def update_message(self, new_message: str) -> None:
        """Update the status message during a long operation."""
        if new_message:
            self.window.statusBar().showMessage(new_message)
            QApplication.processEvents()
