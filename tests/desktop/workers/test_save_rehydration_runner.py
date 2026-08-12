"""Tests for desktop save-runtime rehydration worker execution."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from PyQt5 import QtCore

from ephys_alignment_gui.application.save_runtime_rehydration import (
    SaveRuntimeRehydrated,
    SaveRuntimeRehydrationPlan,
)
from ephys_alignment_gui.desktop.workers.save_rehydration_runner import (
    QtSaveRuntimeRehydrationRunner,
)
from ephys_alignment_gui.io.load_data_job import LoadDataJobProgress


def test_qt_save_rehydration_runner_delivers_callbacks_on_main_thread() -> None:
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    _ = app
    main_thread_id = threading.get_ident()
    plan = _rehydration_plan()
    worker_thread_ids: list[int] = []
    progress_thread_ids: list[int] = []
    finished_thread_ids: list[int] = []
    results: list[SaveRuntimeRehydrated] = []
    loop = QtCore.QEventLoop()

    def run_job(plan, *, progress=None, cancel_token=None):
        worker_thread_ids.append(threading.get_ident())
        if progress is not None:
            progress(
                LoadDataJobProgress(
                    target=plan.dependencies[0].load_target,
                    phase="ephys",
                    status="started",
                    message="Reloading runtime data...",
                )
            )
        return SaveRuntimeRehydrated(1)

    def on_progress(_event):
        progress_thread_ids.append(threading.get_ident())

    def on_finished(result):
        finished_thread_ids.append(threading.get_ident())
        results.append(result)

    runner = QtSaveRuntimeRehydrationRunner()

    def quit_when_done() -> None:
        if results and not runner.is_running:
            loop.quit()
            return
        QtCore.QTimer.singleShot(10, quit_when_done)

    runner.start(
        plan=plan,
        run_job=run_job,
        on_progress=on_progress,
        on_finished=on_finished,
    )
    QtCore.QTimer.singleShot(0, quit_when_done)
    QtCore.QTimer.singleShot(3000, loop.quit)
    loop.exec_()

    assert results
    assert worker_thread_ids and worker_thread_ids[0] != main_thread_id
    assert progress_thread_ids == [main_thread_id]
    assert finished_thread_ids == [main_thread_id]


def test_qt_save_rehydration_runner_shutdown_cancels_and_waits_for_worker() -> None:
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    _ = app
    plan = _rehydration_plan()
    started = threading.Event()
    cancellation_reasons: list[str | None] = []

    def run_job(_plan, *, progress=None, cancel_token=None):
        started.set()
        while not cancel_token.cancelled:
            time.sleep(0.01)
        cancellation_reasons.append(cancel_token.reason)
        return SaveRuntimeRehydrated(0)

    runner = QtSaveRuntimeRehydrationRunner()
    runner.start(
        plan=plan,
        run_job=run_job,
        on_progress=lambda _event: None,
        on_finished=lambda _result: None,
    )

    assert started.wait(timeout=3)
    assert runner.is_running

    assert runner.shutdown("closing", timeout_ms=3000)

    assert cancellation_reasons == ["closing"]
    assert not runner.is_running


def _rehydration_plan() -> SaveRuntimeRehydrationPlan:
    target = SimpleNamespace(
        recording_id="rec",
        probe_name="probeA",
        stream_key=("rec", "stream"),
        shank_idx=0,
    )
    dependency = SimpleNamespace(load_target=target)
    return SaveRuntimeRehydrationPlan((dependency,))
