"""Tests for desktop load-data worker execution."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from qt_helpers import qt_app, wait_for_qt

from ephys_alignment_gui.application.results import (
    FreshLoadExecution,
    FreshLoadJobInvocation,
    LoadDataFreshPrepared,
)
from ephys_alignment_gui.desktop.workers.load_data_runner import QtFreshLoadRunner
from ephys_alignment_gui.io.load_data_job import (
    LoadDataCancelToken,
    LoadDataJobCancelled,
    LoadDataJobCompleted,
    LoadDataJobProgress,
    LoadDataJobRequest,
)


def test_qt_fresh_load_runner_delivers_callbacks_on_main_thread() -> None:
    qt_app()
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
    try:
        runner.start(
            execution=execution,
            invocation=invocation,
            run_job=run_job,
            on_progress=on_progress,
            on_finished=on_finished,
        )
        assert wait_for_qt(lambda: bool(results) and not runner.is_running)
    finally:
        runner.shutdown("test cleanup", timeout_ms=3000)

    assert results
    assert worker_thread_ids and worker_thread_ids[0] != main_thread_id
    assert progress_thread_ids == [main_thread_id]
    assert finished_thread_ids == [main_thread_id]


def test_qt_fresh_load_runner_shutdown_cancels_and_waits_for_worker() -> None:
    qt_app()
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
    cancel_token = LoadDataCancelToken()
    invocation = FreshLoadJobInvocation(
        execution=execution,
        request=LoadDataJobRequest(target, load_id=1),
        cancel_token=cancel_token,
    )
    started = threading.Event()

    def run_job(invocation, *, progress=None):
        started.set()
        while not invocation.cancel_token.cancelled:
            time.sleep(0.01)
        return LoadDataJobCancelled(
            target=invocation.request.target,
            reason=invocation.cancel_token.reason or "cancelled",
        )

    runner = QtFreshLoadRunner()
    runner.start(
        execution=execution,
        invocation=invocation,
        run_job=run_job,
        on_progress=lambda _execution, _event: None,
        on_finished=lambda _execution, _result: None,
    )

    assert started.wait(timeout=3)
    assert runner.is_running

    assert runner.shutdown("closing", timeout_ms=3000)

    assert cancel_token.reason == "closing"
    assert not runner.is_running


def test_qt_fresh_load_runner_rejects_start_while_worker_running() -> None:
    qt_app()
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
    cancel_token = LoadDataCancelToken()
    invocation = FreshLoadJobInvocation(
        execution=FreshLoadExecution(load_id=1, prepared=prepared),
        request=LoadDataJobRequest(target, load_id=1),
        cancel_token=cancel_token,
    )
    started = threading.Event()

    def run_job(invocation, *, progress=None):
        started.set()
        while not invocation.cancel_token.cancelled:
            time.sleep(0.01)
        return LoadDataJobCancelled(
            target=invocation.request.target,
            reason=invocation.cancel_token.reason or "cancelled",
        )

    runner = QtFreshLoadRunner()
    runner.start(
        execution=invocation.execution,
        invocation=invocation,
        run_job=run_job,
        on_progress=lambda _execution, _event: None,
        on_finished=lambda _execution, _result: None,
    )

    try:
        assert started.wait(timeout=3)
        with pytest.raises(RuntimeError, match="already running"):
            runner.start(
                execution=invocation.execution,
                invocation=invocation,
                run_job=run_job,
                on_progress=lambda _execution, _event: None,
                on_finished=lambda _execution, _result: None,
            )
    finally:
        assert runner.shutdown("closing", timeout_ms=3000)
