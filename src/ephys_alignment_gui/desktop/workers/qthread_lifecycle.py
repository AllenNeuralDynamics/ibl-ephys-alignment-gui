"""Utilities for safe PyQt QThread lifetime cleanup."""

from __future__ import annotations

from collections.abc import Callable

from PyQt5 import QtCore


def defer_cleanup_until_thread_stopped(
    thread: QtCore.QThread,
    cleanup: Callable[[], None],
    *,
    retry_ms: int = 10,
) -> None:
    """Run cleanup only after ``thread`` is fully stopped.

    PyQt can deadlock if the final Python reference to a ``QThread`` is released
    from the ``finished`` signal path while Qt is still unwinding the native
    thread finish. Polling with a zero-timeout wait keeps ownership alive until
    the destructor can run without blocking the GUI thread.
    """

    def maybe_cleanup() -> None:
        if thread.wait(0):
            cleanup()
            return
        QtCore.QTimer.singleShot(retry_ms, maybe_cleanup)

    QtCore.QTimer.singleShot(0, maybe_cleanup)
