"""Tests for desktop load-data worker execution."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from PyQt5 import QtCore

from ephys_alignment_gui.application.results import (
    FreshLoadExecution,
    FreshLoadJobInvocation,
    LoadDataFreshPrepared,
)
from ephys_alignment_gui.desktop.workers.load_data_runner import QtFreshLoadRunner
from ephys_alignment_gui.io.load_data_job import (
    LoadDataCancelToken,
    LoadDataJobCompleted,
    LoadDataJobProgress,
    LoadDataJobRequest,
)


def test_qt_fresh_load_runner_delivers_callbacks_on_main_thread() -> None:
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    _ = app
    main_thread_id = threading.get_ident()
    target = SimpleNamespace(
        recording_id="rec",
        probe_name="probeA",
        stream_key=("rec", "stream"),
        shank_idx=0,
    )
    prepared = LoadDataFreshPrepared(
        stream_key=("rec", "stream"),
        shank_idx=0,
        preserve_plot_selection=True,
        target=target,
    )
    execution = FreshLoadExecution(load_id=1, prepared=prepared)
    invocation = FreshLoadJobInvocation(
        execution=execution,
        request=LoadDataJobRequest(target, load_id=1),
        cancel_token=LoadDataCancelToken(),
    )
    worker_thread_ids: list[int] = []
    progress_thread_ids: list[int] = []
    finished_thread_ids: list[int] = []
    results: list[LoadDataJobCompleted] = []
    loop = QtCore.QEventLoop()

    def run_job(invocation, *, progress=None):
        worker_thread_ids.append(threading.get_ident())
        if progress is not None:
            progress(
                LoadDataJobProgress(
                    target=invocation.request.target,
                    phase="ephys",
                    status="started",
                    message="Loading ephys data...",
                    load_id=invocation.request.load_id,
                )
            )
        return LoadDataJobCompleted(
            target=invocation.request.target,
            ephys=SimpleNamespace(stream=SimpleNamespace(ephys_dir="/tmp/ephys")),
            histology=SimpleNamespace(),
        )

    def on_progress(_execution, _event):
        progress_thread_ids.append(threading.get_ident())

    def on_finished(_execution, result):
        finished_thread_ids.append(threading.get_ident())
        results.append(result)

    runner = QtFreshLoadRunner()

    def quit_when_done() -> None:
        if results and not runner.is_running:
            loop.quit()
            return
        QtCore.QTimer.singleShot(10, quit_when_done)

    runner.start(
        execution=execution,
        invocation=invocation,
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
