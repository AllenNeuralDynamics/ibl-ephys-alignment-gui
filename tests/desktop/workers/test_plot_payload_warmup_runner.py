"""Tests for desktop plot-payload warmup worker execution."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from qt_helpers import qt_app, wait_for_qt

from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.desktop.workers.plot_payload_warmup_runner import (
    QtPlotPayloadWarmupRunner,
)
from ephys_alignment_gui.plotting.payload_warmup import (
    PlotPayloadWarmupCancelToken,
    PlotPayloadWarmupRequest,
)


def test_qt_plot_payload_warmup_runner_delivers_callback_on_main_thread() -> None:
    qt_app()
    main_thread_id = threading.get_ident()
    request = PlotPayloadWarmupRequest(
        stream_key=("rec", "stream"),
        stream=SimpleNamespace(),
        shank_idx=0,
        unit_filter="unitrefine_neural",
        spec_keys=("line.fr",),
    )
    worker_thread_ids: list[int] = []
    finished_thread_ids: list[int] = []
    results: list[Failed] = []

    seen_tokens: list[PlotPayloadWarmupCancelToken | None] = []

    def run_job(_request, *, cancel_token=None):
        worker_thread_ids.append(threading.get_ident())
        seen_tokens.append(cancel_token)
        return Failed("expected test result")

    def on_finished(_request, result):
        finished_thread_ids.append(threading.get_ident())
        results.append(result)

    runner = QtPlotPayloadWarmupRunner()
    try:
        runner.start(
            request=request,
            run_job=run_job,
            on_finished=on_finished,
        )
        assert wait_for_qt(lambda: bool(results) and not runner.is_running)
    finally:
        runner.shutdown("test cleanup", timeout_ms=3000)

    assert results
    assert worker_thread_ids and worker_thread_ids[0] != main_thread_id
    assert finished_thread_ids == [main_thread_id]
    assert isinstance(seen_tokens[0], PlotPayloadWarmupCancelToken)
