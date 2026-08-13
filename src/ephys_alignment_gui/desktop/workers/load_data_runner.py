"""Qt worker runner for fresh load-data jobs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PyQt5 import QtCore

from ephys_alignment_gui.application.results import (
    FreshLoadExecution,
    FreshLoadJobInvocation,
)
from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.io.load_data_job import (
    LoadDataJobCancelled,
    LoadDataJobCompleted,
    LoadDataJobProgress,
    LoadDataProgressCallback,
)

logger = logging.getLogger(__name__)

FreshLoadJobResult = LoadDataJobCompleted | LoadDataJobCancelled | Failed
FreshLoadProgressHandler = Callable[[FreshLoadExecution, LoadDataJobProgress], None]
FreshLoadFinishedHandler = Callable[[FreshLoadExecution, FreshLoadJobResult], None]


class RunFreshLoadJob(Protocol):
    """Callable that runs fresh-load IO without publishing app events."""

    def __call__(
        self,
        invocation: FreshLoadJobInvocation,
        *,
        progress: LoadDataProgressCallback | None = None,
    ) -> FreshLoadJobResult:
        """Run one fresh-load job."""
        ...


class FreshLoadRunner(Protocol):
    """Execution adapter for one foreground fresh-load job."""

    @property
    def is_running(self) -> bool:
        """Whether a foreground load worker is still running."""
        ...

    def start(
        self,
        *,
        execution: FreshLoadExecution,
        invocation: FreshLoadJobInvocation,
        run_job: RunFreshLoadJob,
        on_progress: FreshLoadProgressHandler,
        on_finished: FreshLoadFinishedHandler,
    ) -> None:
        """Start a fresh-load job."""
        ...

    def cancel(self, reason: str) -> None:
        """Request cancellation of the active job."""
        ...

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        """Request cancellation and wait for the active job to stop."""
        ...


class FreshLoadWorker(QtCore.QObject):
    """QObject that runs one fresh-load job in its owning thread."""

    progress = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(object)

    def __init__(
        self,
        *,
        invocation: FreshLoadJobInvocation,
        run_job: RunFreshLoadJob,
    ) -> None:
        super().__init__()
        self._invocation = invocation
        self._run_job = run_job

    @QtCore.pyqtSlot()
    def run(self) -> None:
        """Run the load job and emit its terminal result."""
        try:
            result = self._run_job(self._invocation, progress=self.progress.emit)
        except Exception as exc:
            logger.exception("Fresh load worker failed")
            result = Failed(f"Fresh load worker failed: {exc}")
        self.finished.emit(result)


@dataclass
class _ActiveFreshLoadWorker:
    execution: FreshLoadExecution
    invocation: FreshLoadJobInvocation
    thread: QtCore.QThread
    worker: FreshLoadWorker
    on_progress: FreshLoadProgressHandler
    on_finished: FreshLoadFinishedHandler


class QtFreshLoadRunner(QtCore.QObject):
    """Run fresh-load jobs on a background QThread."""

    def __init__(self) -> None:
        super().__init__()
        self._active: _ActiveFreshLoadWorker | None = None

    @property
    def is_running(self) -> bool:
        """Whether a foreground load worker is still running."""
        active = self._active
        return active is not None and active.thread.isRunning()

    def start(
        self,
        *,
        execution: FreshLoadExecution,
        invocation: FreshLoadJobInvocation,
        run_job: RunFreshLoadJob,
        on_progress: FreshLoadProgressHandler,
        on_finished: FreshLoadFinishedHandler,
    ) -> None:
        """Start a fresh-load job on a background QThread."""
        if self.is_running:
            raise RuntimeError("Fresh load worker is already running")

        thread = QtCore.QThread()
        worker = FreshLoadWorker(invocation=invocation, run_job=run_job)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_progress)
        worker.finished.connect(self._handle_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda this_worker=worker: self._clear_if_active(this_worker)
        )

        self._active = _ActiveFreshLoadWorker(
            execution=execution,
            invocation=invocation,
            thread=thread,
            worker=worker,
            on_progress=on_progress,
            on_finished=on_finished,
        )
        thread.start()

    def cancel(self, reason: str) -> None:
        """Request cooperative cancellation of the active load worker."""
        active = self._active
        if active is None:
            return
        active.invocation.cancel_token.cancel(reason)
        active.thread.requestInterruption()

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        """Request cancellation and wait for the active load worker to stop.

        Cancellation is cooperative: this method does not terminate the worker
        thread. If heavy IO is still inside a non-interruptible call after the
        timeout, callers should keep the desktop lifecycle alive.
        """
        active = self._active
        if active is None:
            return True

        self.cancel(reason)
        active.thread.quit()
        stopped = active.thread.wait(timeout_ms)
        if not stopped:
            logger.warning(
                "Fresh load worker did not stop within %s ms after cancellation",
                timeout_ms,
            )
            return False

        if self._active is active:
            self._active = None
        return True

    def _clear_if_active(self, worker: FreshLoadWorker) -> None:
        active = self._active
        if active is not None and active.worker is worker:
            self._active = None

    @QtCore.pyqtSlot(object)
    def _handle_progress(self, event: LoadDataJobProgress) -> None:
        active = self._active
        if active is None:
            return
        active.on_progress(active.execution, event)

    @QtCore.pyqtSlot(object)
    def _handle_finished(self, result: FreshLoadJobResult) -> None:
        active = self._active
        if active is None:
            return
        active.on_finished(active.execution, result)
