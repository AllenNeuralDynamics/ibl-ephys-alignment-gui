"""Qt worker runner for plot payload cache warmup jobs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PyQt5 import QtCore

from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.desktop.workers.qthread_lifecycle import (
    defer_cleanup_until_thread_stopped,
)
from ephys_alignment_gui.plotting.payload_warmup import (
    PlotPayloadCacheWarmed,
    PlotPayloadWarmupRequest,
)

logger = logging.getLogger(__name__)

PlotPayloadWarmupResult = PlotPayloadCacheWarmed | Failed
PlotPayloadWarmupFinishedHandler = Callable[
    [PlotPayloadWarmupRequest, PlotPayloadWarmupResult],
    None,
]


class RunPlotPayloadWarmupJob(Protocol):
    """Callable that warms one plot payload cache."""

    def __call__(
        self,
        request: PlotPayloadWarmupRequest,
    ) -> PlotPayloadWarmupResult:
        """Run one plot payload warmup job."""
        ...


class PlotPayloadWarmupRunner(Protocol):
    """Execution adapter for one plot payload warmup job."""

    @property
    def is_running(self) -> bool:
        """Whether a warmup worker is still running."""
        ...

    def start(
        self,
        *,
        request: PlotPayloadWarmupRequest,
        run_job: RunPlotPayloadWarmupJob,
        on_finished: PlotPayloadWarmupFinishedHandler,
    ) -> None:
        """Start a plot payload warmup job."""
        ...

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        """Wait for the active warmup job to stop."""
        ...


class PlotPayloadWarmupWorker(QtCore.QObject):
    """QObject that runs one plot payload warmup job in its owning thread."""

    finished = QtCore.pyqtSignal(object)

    def __init__(
        self,
        *,
        request: PlotPayloadWarmupRequest,
        run_job: RunPlotPayloadWarmupJob,
    ) -> None:
        super().__init__()
        self._request = request
        self._run_job = run_job

    @QtCore.pyqtSlot()
    def run(self) -> None:
        """Run the warmup job and emit its terminal result."""
        try:
            result = self._run_job(self._request)
        except Exception as exc:
            logger.exception("Plot payload warmup worker failed")
            result = Failed(f"Plot payload warmup worker failed: {exc}")
        self.finished.emit(result)


@dataclass
class _ActivePlotPayloadWarmupWorker:
    request: PlotPayloadWarmupRequest
    thread: QtCore.QThread
    worker: PlotPayloadWarmupWorker
    on_finished: PlotPayloadWarmupFinishedHandler


class QtPlotPayloadWarmupRunner(QtCore.QObject):
    """Run plot payload warmup jobs on a background QThread."""

    def __init__(self) -> None:
        super().__init__()
        self._active: _ActivePlotPayloadWarmupWorker | None = None

    @property
    def is_running(self) -> bool:
        """Whether a warmup worker is still running."""
        active = self._active
        return active is not None and active.thread.isRunning()

    def start(
        self,
        *,
        request: PlotPayloadWarmupRequest,
        run_job: RunPlotPayloadWarmupJob,
        on_finished: PlotPayloadWarmupFinishedHandler,
    ) -> None:
        """Start a plot payload warmup job on a background QThread."""
        if self.is_running:
            raise RuntimeError("Plot payload warmup worker is already running")

        thread = QtCore.QThread()
        worker = PlotPayloadWarmupWorker(request=request, run_job=run_job)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(
            lambda this_thread=thread, this_worker=worker: (
                defer_cleanup_until_thread_stopped(
                    this_thread,
                    lambda: self._clear_if_active(this_worker),
                )
            )
        )

        self._active = _ActivePlotPayloadWarmupWorker(
            request=request,
            thread=thread,
            worker=worker,
            on_finished=on_finished,
        )
        thread.start()

    def shutdown(self, reason: str, *, timeout_ms: int = 5000) -> bool:
        """Wait for the active warmup worker to stop."""
        active = self._active
        if active is None:
            return True

        active.thread.requestInterruption()
        stopped = active.thread.wait(timeout_ms)
        if not stopped:
            logger.warning(
                "Plot payload warmup worker did not stop within %s ms after %s",
                timeout_ms,
                reason,
            )
            return False

        if self._active is active:
            self._active = None
        return True

    def _clear_if_active(self, worker: PlotPayloadWarmupWorker) -> None:
        active = self._active
        if active is not None and active.worker is worker:
            self._active = None

    @QtCore.pyqtSlot(object)
    def _handle_finished(self, result: PlotPayloadWarmupResult) -> None:
        active = self._active
        if active is None:
            return
        active.on_finished(active.request, result)
