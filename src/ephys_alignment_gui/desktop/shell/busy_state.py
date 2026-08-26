"""Shared Qt busy-state ownership for desktop operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any

from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QApplication

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = Any


@dataclass
class _WidgetBusyState:
    desired_enabled: bool
    lease_count: int = 1


class BusyStateManager:
    """Issue overlap-safe leases for shared desktop busy presentation."""

    def __init__(
        self,
        window: Any,
        *,
        set_wait_cursor: Callable[[], None] | None = None,
        restore_wait_cursor: Callable[[], None] | None = None,
    ) -> None:
        self._window = window
        self._set_wait_cursor = set_wait_cursor or (
            lambda: QApplication.setOverrideCursor(Qt.WaitCursor)
        )
        self._restore_wait_cursor = restore_wait_cursor or (
            QApplication.restoreOverrideCursor
        )
        self._active_leases: dict[object, BusyContext] = {}
        self._widget_states: dict[Any, _WidgetBusyState] = {}

    def context(
        self,
        message: str | None = None,
        success_message: str | None = None,
        error_message: str | None = None,
        disable_widgets: list[Any] | Any | None = None,
        success_timeout_ms: int = 3000,
        error_timeout_ms: int = 5000,
    ) -> BusyContext:
        """Return a context manager backed by a shared busy-state lease."""
        return BusyContext(
            manager=self,
            message=message,
            success_message=success_message,
            error_message=error_message,
            disable_widgets=disable_widgets,
            success_timeout_ms=success_timeout_ms,
            error_timeout_ms=error_timeout_ms,
        )

    def acquire(self, lease: BusyContext) -> None:
        """Acquire one busy lease on the GUI thread."""
        self._assert_gui_thread()
        if lease.token in self._active_leases:
            raise RuntimeError("Busy-state lease is already active.")
        if not self._active_leases:
            self._set_wait_cursor()
        self._active_leases[lease.token] = lease
        for widget in lease.disable_widgets:
            self._disable_widget(widget)
        self._render_active_message()

    def release(
        self,
        lease: BusyContext,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
    ) -> None:
        """Release a busy lease safely, including out-of-order completion."""
        self._assert_gui_thread()
        if self._active_leases.pop(lease.token, None) is None:
            return
        for widget in lease.disable_widgets:
            self._release_widget(widget)
        if self._active_leases:
            self._render_active_message()
            return

        self._restore_wait_cursor()
        status_bar = self._status_bar()
        if status_bar is None:
            return
        if exc_type is not None:
            message = lease.error_message or f"Error: {exc_val}"
            status_bar.showMessage(message, lease.error_timeout_ms)
        elif lease.success_message:
            status_bar.showMessage(
                lease.success_message,
                lease.success_timeout_ms,
            )
        else:
            status_bar.clearMessage()

    def update_message(self, lease: BusyContext, message: str) -> None:
        """Update an active lease's status text on the GUI thread."""
        self._assert_gui_thread()
        if lease.token not in self._active_leases:
            return
        lease.message = message
        self._render_active_message()

    def _disable_widget(self, widget: Any) -> None:
        state = self._widget_states.get(widget)
        if state is None:
            self._widget_states[widget] = _WidgetBusyState(
                desired_enabled=_locally_enabled(widget)
            )
            widget.setEnabled(False)
            return
        state.lease_count += 1

    def _release_widget(self, widget: Any) -> None:
        state = self._widget_states.get(widget)
        if state is None:
            return
        state.lease_count -= 1
        if state.lease_count > 0:
            return
        del self._widget_states[widget]
        try:
            widget.setEnabled(state.desired_enabled)
        except RuntimeError:
            # The widget may have been deleted during top-level Qt teardown.
            pass

    def _render_active_message(self) -> None:
        status_bar = self._status_bar()
        if status_bar is None:
            return
        for lease in reversed(tuple(self._active_leases.values())):
            if lease.message:
                status_bar.showMessage(lease.message)
                return
        status_bar.clearMessage()

    def _status_bar(self) -> Any | None:
        try:
            return self._window.statusBar()
        except RuntimeError:
            return None

    @staticmethod
    def _assert_gui_thread() -> None:
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            raise RuntimeError("Busy state may only be changed on the Qt GUI thread.")


class BusyContext:
    """Context-manager adapter around a :class:`BusyStateManager` lease."""

    def __init__(
        self,
        manager: BusyStateManager,
        message: str | None = None,
        success_message: str | None = None,
        error_message: str | None = None,
        disable_widgets: list[Any] | Any | None = None,
        success_timeout_ms: int = 3000,
        error_timeout_ms: int = 5000,
    ) -> None:
        self.manager = manager
        self.message = message
        self.success_message = success_message
        self.error_message = error_message
        self.disable_widgets = _unique_widgets(disable_widgets)
        self.success_timeout_ms = success_timeout_ms
        self.error_timeout_ms = error_timeout_ms
        self.token = object()
        self._active = False

    def __enter__(self) -> Self:
        """Acquire the context's busy-state lease."""
        if self._active:
            raise RuntimeError("Busy context cannot be entered more than once.")
        self.manager.acquire(self)
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> bool:
        """Release the context's lease without suppressing exceptions."""
        if not self._active:
            return False
        self.manager.release(self, exc_type, exc_val)
        self._active = False
        return False

    def update_message(self, new_message: str) -> None:
        """Update this lease's status-bar message while active."""
        self.manager.update_message(self, new_message)


def _unique_widgets(widgets: list[Any] | Any | None) -> tuple[Any, ...]:
    if widgets is None:
        return ()
    values: Iterable[Any] = (
        widgets if isinstance(widgets, (list, tuple)) else (widgets,)
    )
    return tuple(dict.fromkeys(values))


def _locally_enabled(widget: Any) -> bool:
    """Return desired local state without manager-disabled ancestors."""
    test_attribute = getattr(widget, "testAttribute", None)
    if callable(test_attribute):
        return not bool(test_attribute(Qt.WA_ForceDisabled))
    return bool(widget.isEnabled())
