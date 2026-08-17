"""Qt event-loop helpers for desktop worker tests."""

from __future__ import annotations

import time
from collections.abc import Callable

from PyQt5 import QtCore

_QT_APP: QtCore.QCoreApplication | None = None


def qt_app() -> QtCore.QCoreApplication:
    """Return a process-stable Qt core application for worker tests."""
    global _QT_APP
    app = QtCore.QCoreApplication.instance()
    if app is None:
        _QT_APP = QtCore.QCoreApplication([])
        app = _QT_APP
    return app


def wait_for_qt(
    condition: Callable[[], bool],
    *,
    timeout_s: float = 3.0,
) -> bool:
    """Pump Qt events until a condition is met or a deadline expires."""
    app = qt_app()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if condition():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            return True
        time.sleep(0.01)
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    return condition()
