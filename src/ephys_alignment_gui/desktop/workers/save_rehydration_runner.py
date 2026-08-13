"""Qt worker runner for save-time runtime rehydration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PyQt5 import QtCore

from ephys_alignment_gui.application.save_runtime_rehydration import (
    SaveRuntimeRehydrated,
    SaveRuntimeRehydrationCancelled,
    SaveRuntimeRehydrationPlan,
)
from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.io.load_data_job import (
    LoadDataCancelToken,
    LoadDataJobProgress,
    LoadDataProgressCallback,
)

logger = logging.getLogger(__name__)

SaveRuntimeRehydrationResult = (
    SaveRuntimeRehydrated | SaveRuntimeRehydrationCancelled | Failed
)
SaveRuntimeRehydrationProgressHandler = Callable[[LoadDataJobProgress], None]
SaveRuntimeRehydrationFinishedHandler = Callable[[SaveRuntimeRehydrationResult], None]


class RunSaveRuntimeRehydrationJob(Protocol):
    """Callable that reloads save-runtime dependencies without publishing events."""

    def __call__(
        self,
        plan: SaveRuntimeRehydrationPlan,
        *,
        progress: LoadDataProgressCallback | None = None,
        cancel_token: LoadDataCancelToken | None = None,
    ) -> SaveRuntimeRehydrationResult:
        """Run save-runtime rehydration."""
        ...


class SaveRuntimeRehydrationRunner(Protocol):
    """Execution adapter for one save-runtime rehydration job."""

    @property
    def is_running(self) -> bool:
        """Whether a save-runtime rehydration worker is still running."""
        ...

    def start(
        self,
        *,
        plan: SaveRuntimeRehydrationPlan,
        run_job: RunSaveRuntimeRehydrationJob,
        on_progress: SaveRuntimeRehydrationProgressHandler,
        on_finished: SaveRuntimeRehydrationFinishedHandler,
    ) -> None:
        """Start save-runtime rehydration."""
        ...

    def cancel(self, reason: str) -> None:
        """Request cancellation of the active job."""
        ...

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        """Request cancellation and wait for the active job to stop."""
        ...


class SaveRuntimeRehydrationWorker(QtCore.QObject):
    """QObject that rehydrates save-runtime dependencies in its owning thread."""

    progress = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(object)

    def __init__(
        self,
        *,
        plan: SaveRuntimeRehydrationPlan,
        run_job: RunSaveRuntimeRehydrationJob,
        cancel_token: LoadDataCancelToken,
    ) -> None:
        super().__init__()
        self._plan = plan
        self._run_job = run_job
        self._cancel_token = cancel_token

    @QtCore.pyqtSlot()
    def run(self) -> None:
        """Run the save-runtime rehydration job and emit its terminal result."""
        try:
            result = self._run_job(
                self._plan,
                progress=self.progress.emit,
                cancel_token=self._cancel_token,
            )
        except Exception as exc:
            logger.exception("Save-runtime rehydration worker failed")
            result = Failed(f"Save-runtime rehydration worker failed: {exc}")
        self.finished.emit(result)


@dataclass
class _ActiveSaveRuntimeRehydrationWorker:
    plan: SaveRuntimeRehydrationPlan
    cancel_token: LoadDataCancelToken
    thread: QtCore.QThread
    worker: SaveRuntimeRehydrationWorker
    on_progress: SaveRuntimeRehydrationProgressHandler
    on_finished: SaveRuntimeRehydrationFinishedHandler


class QtSaveRuntimeRehydrationRunner(QtCore.QObject):
    """Run save-runtime rehydration on a background QThread."""

    def __init__(self) -> None:
        super().__init__()
        self._active: _ActiveSaveRuntimeRehydrationWorker | None = None

    @property
    def is_running(self) -> bool:
        """Whether a save-runtime rehydration worker is still running."""
        active = self._active
        return active is not None and active.thread.isRunning()

    def start(
        self,
        *,
        plan: SaveRuntimeRehydrationPlan,
        run_job: RunSaveRuntimeRehydrationJob,
        on_progress: SaveRuntimeRehydrationProgressHandler,
        on_finished: SaveRuntimeRehydrationFinishedHandler,
    ) -> None:
        """Start save-runtime rehydration on a background QThread."""
        if self.is_running:
            raise RuntimeError("Save-runtime rehydration worker is already running")

        cancel_token = LoadDataCancelToken()
        thread = QtCore.QThread()
        worker = SaveRuntimeRehydrationWorker(
            plan=plan,
            run_job=run_job,
            cancel_token=cancel_token,
        )
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

        self._active = _ActiveSaveRuntimeRehydrationWorker(
            plan=plan,
            cancel_token=cancel_token,
            thread=thread,
            worker=worker,
            on_progress=on_progress,
            on_finished=on_finished,
        )
        thread.start()

    def cancel(self, reason: str) -> None:
        """Request cooperative cancellation of the active rehydration worker."""
        active = self._active
        if active is None:
            return
        active.cancel_token.cancel(reason)
        active.thread.requestInterruption()

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        """Request cancellation and wait for the active rehydration worker."""
        active = self._active
        if active is None:
            return True

        self.cancel(reason)
        active.thread.quit()
        stopped = active.thread.wait(timeout_ms)
        if not stopped:
            logger.warning(
                "Save-runtime rehydration worker did not stop within %s ms",
                timeout_ms,
            )
            return False

        if self._active is active:
            self._active = None
        return True

    def _clear_if_active(self, worker: SaveRuntimeRehydrationWorker) -> None:
        active = self._active
        if active is not None and active.worker is worker:
            self._active = None

    @QtCore.pyqtSlot(object)
    def _handle_progress(self, event: LoadDataJobProgress) -> None:
        active = self._active
        if active is None:
            return
        active.on_progress(event)

    @QtCore.pyqtSlot(object)
    def _handle_finished(self, result: SaveRuntimeRehydrationResult) -> None:
        active = self._active
        if active is None:
            return
        active.on_finished(result)
