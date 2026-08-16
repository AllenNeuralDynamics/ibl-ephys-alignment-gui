"""Qt worker runner for final edited-alignment save jobs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PyQt5 import QtCore

from ephys_alignment_gui.application.alignment_save_job import (
    AlignmentSaveJobCompleted,
    PreparedAlignmentSave,
)
from ephys_alignment_gui.core.alignment_events import SaveProgressUpdated
from ephys_alignment_gui.core.workflow import Failed

logger = logging.getLogger(__name__)

AlignmentSaveJobResult = AlignmentSaveJobCompleted | Failed
AlignmentSaveProgressHandler = Callable[[SaveProgressUpdated], None]
AlignmentSaveFinishedHandler = Callable[
    [PreparedAlignmentSave, AlignmentSaveJobResult],
    None,
]


class RunPreparedAlignmentSaveJob(Protocol):
    """Callable that builds and writes prepared alignment outputs."""

    def __call__(
        self,
        prepared: PreparedAlignmentSave,
        *,
        progress: AlignmentSaveProgressHandler | None = None,
    ) -> AlignmentSaveJobResult:
        """Run one prepared alignment save job."""
        ...


class AlignmentSaveRunner(Protocol):
    """Execution adapter for one prepared alignment save job."""

    @property
    def is_running(self) -> bool:
        """Whether a save worker is still running."""
        ...

    def start(
        self,
        *,
        prepared: PreparedAlignmentSave,
        run_job: RunPreparedAlignmentSaveJob,
        on_progress: AlignmentSaveProgressHandler,
        on_finished: AlignmentSaveFinishedHandler,
    ) -> None:
        """Start a prepared alignment save job."""
        ...

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        """Wait for the active save job to stop."""
        ...


class AlignmentSaveWorker(QtCore.QObject):
    """QObject that runs one prepared save job in its owning thread."""

    progress = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(object)

    def __init__(
        self,
        *,
        prepared: PreparedAlignmentSave,
        run_job: RunPreparedAlignmentSaveJob,
    ) -> None:
        super().__init__()
        self._prepared = prepared
        self._run_job = run_job

    @QtCore.pyqtSlot()
    def run(self) -> None:
        """Run the save job and emit its terminal result."""
        try:
            result = self._run_job(self._prepared, progress=self.progress.emit)
        except Exception as exc:
            logger.exception("Prepared alignment save worker failed")
            result = Failed(f"Prepared alignment save worker failed: {exc}")
        self.finished.emit(result)


@dataclass
class _ActiveAlignmentSaveWorker:
    prepared: PreparedAlignmentSave
    thread: QtCore.QThread
    worker: AlignmentSaveWorker
    on_progress: AlignmentSaveProgressHandler
    on_finished: AlignmentSaveFinishedHandler


class QtAlignmentSaveRunner(QtCore.QObject):
    """Run final alignment save work on a background QThread."""

    def __init__(self) -> None:
        super().__init__()
        self._active: _ActiveAlignmentSaveWorker | None = None

    @property
    def is_running(self) -> bool:
        """Whether a save worker is still running."""
        active = self._active
        return active is not None and active.thread.isRunning()

    def start(
        self,
        *,
        prepared: PreparedAlignmentSave,
        run_job: RunPreparedAlignmentSaveJob,
        on_progress: AlignmentSaveProgressHandler,
        on_finished: AlignmentSaveFinishedHandler,
    ) -> None:
        """Start a prepared save job on a background QThread."""
        if self.is_running:
            raise RuntimeError("Alignment save worker is already running")

        thread = QtCore.QThread()
        worker = AlignmentSaveWorker(prepared=prepared, run_job=run_job)
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

        self._active = _ActiveAlignmentSaveWorker(
            prepared=prepared,
            thread=thread,
            worker=worker,
            on_progress=on_progress,
            on_finished=on_finished,
        )
        thread.start()

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        """Wait for the active save worker to stop.

        Final output writes and ANTs transforms are not currently cancellable.
        During desktop teardown, wait for them to finish so Qt does not destroy
        a running thread.
        """
        active = self._active
        if active is None:
            return True

        active.thread.requestInterruption()
        stopped = active.thread.wait(timeout_ms)
        if not stopped:
            logger.warning(
                "Alignment save worker did not stop within %s ms after %s",
                timeout_ms,
                reason,
            )
            return False

        if self._active is active:
            self._active = None
        return True

    def _clear_if_active(self, worker: AlignmentSaveWorker) -> None:
        active = self._active
        if active is not None and active.worker is worker:
            self._active = None

    @QtCore.pyqtSlot(object)
    def _handle_progress(self, event: SaveProgressUpdated) -> None:
        active = self._active
        if active is None:
            return
        active.on_progress(event)

    @QtCore.pyqtSlot(object)
    def _handle_finished(self, result: AlignmentSaveJobResult) -> None:
        active = self._active
        if active is None:
            return
        active.on_finished(active.prepared, result)
