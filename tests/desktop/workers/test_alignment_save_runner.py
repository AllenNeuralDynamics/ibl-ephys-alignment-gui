"""Tests for desktop prepared-alignment save worker execution."""

from __future__ import annotations

import threading
import time

import pytest
from qt_helpers import qt_app, wait_for_qt

from ephys_alignment_gui.application.alignment_save_job import (
    AlignmentSaveJobCancelled,
    AlignmentSaveJobCompleted,
    PreparedAlignmentSave,
)
from ephys_alignment_gui.core.alignment_events import SaveProgressUpdated
from ephys_alignment_gui.desktop.workers.alignment_save_runner import (
    QtAlignmentSaveRunner,
)


def test_qt_alignment_save_runner_delivers_callbacks_on_main_thread() -> None:
    qt_app()
    main_thread_id = threading.get_ident()
    prepared = PreparedAlignmentSave((), use_docdb=False)
    worker_thread_ids: list[int] = []
    progress_thread_ids: list[int] = []
    finished_thread_ids: list[int] = []
    results: list[AlignmentSaveJobCompleted] = []

    def run_job(prepared, *, progress=None, cancel_token=None):
        worker_thread_ids.append(threading.get_ident())
        if progress is not None:
            progress(
                SaveProgressUpdated(
                    key=None,
                    phase="building_outputs",
                    status="started",
                    completed=0,
                    total=len(prepared.targets),
                    message="Transforming CCF...",
                )
            )
        return AlignmentSaveJobCompleted(saved_outputs={})

    def on_progress(_event):
        progress_thread_ids.append(threading.get_ident())

    def on_finished(_prepared, result):
        finished_thread_ids.append(threading.get_ident())
        results.append(result)

    runner = QtAlignmentSaveRunner()
    try:
        runner.start(
            prepared=prepared,
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


def test_qt_alignment_save_runner_shutdown_cancels_and_waits_for_worker() -> None:
    qt_app()
    prepared = PreparedAlignmentSave((), use_docdb=False)
    started = threading.Event()
    cancellation_reasons: list[str | None] = []

    def run_job(_prepared, *, progress=None, cancel_token=None):
        started.set()
        while not cancel_token.cancelled:
            time.sleep(0.01)
        cancellation_reasons.append(cancel_token.reason)
        return AlignmentSaveJobCancelled(
            reason=cancel_token.reason or "cancelled",
        )

    runner = QtAlignmentSaveRunner()
    runner.start(
        prepared=prepared,
        run_job=run_job,
        on_progress=lambda _event: None,
        on_finished=lambda _prepared, _result: None,
    )

    assert started.wait(timeout=3)
    assert runner.is_running

    assert runner.shutdown("closing", timeout_ms=3000)

    assert cancellation_reasons == ["closing"]
    assert not runner.is_running


def test_qt_alignment_save_runner_rejects_start_while_worker_running() -> None:
    qt_app()
    prepared = PreparedAlignmentSave((), use_docdb=False)
    started = threading.Event()

    def run_job(_prepared, *, progress=None, cancel_token=None):
        started.set()
        while not cancel_token.cancelled:
            time.sleep(0.01)
        return AlignmentSaveJobCancelled(
            reason=cancel_token.reason or "cancelled",
        )

    runner = QtAlignmentSaveRunner()
    runner.start(
        prepared=prepared,
        run_job=run_job,
        on_progress=lambda _event: None,
        on_finished=lambda _prepared, _result: None,
    )

    try:
        assert started.wait(timeout=3)
        with pytest.raises(RuntimeError, match="already running"):
            runner.start(
                prepared=prepared,
                run_job=run_job,
                on_progress=lambda _event: None,
                on_finished=lambda _prepared, _result: None,
            )
    finally:
        assert runner.shutdown("closing", timeout_ms=3000)
