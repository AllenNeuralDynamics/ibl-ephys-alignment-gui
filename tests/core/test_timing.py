"""Tests for opt-in timing diagnostics."""

from __future__ import annotations

import logging

from ephys_alignment_gui.core.timing import (
    TIMING_ENV_VAR,
    TimingSession,
    current_timing_session,
    timing_enabled,
)


def test_timing_enabled_requires_truthy_environment_value() -> None:
    assert not timing_enabled({})
    assert not timing_enabled({TIMING_ENV_VAR: ""})
    assert not timing_enabled({TIMING_ENV_VAR: "0"})
    assert not timing_enabled({TIMING_ENV_VAR: "false"})
    assert timing_enabled({TIMING_ENV_VAR: "1"})
    assert timing_enabled({TIMING_ENV_VAR: "yes"})


def test_timing_session_logs_steps_and_restores_context(caplog) -> None:
    caplog.set_level(logging.INFO, logger="ephys_alignment_gui.timing")
    timer = TimingSession(
        workflow="test_workflow",
        enabled=True,
        fields={"probe_name": "probeA"},
    )

    assert current_timing_session() is None
    with timer.activate():
        assert current_timing_session() is timer
        with timer.step("step_a", shank_idx=2):
            pass
        timer.mark("marker")
    timer.finish("completed")

    assert current_timing_session() is None
    messages = [record.getMessage() for record in caplog.records]
    assert any("workflow=test_workflow event=start" in message for message in messages)
    assert any("event=step_a status=completed" in message for message in messages)
    assert any("event=marker status=mark" in message for message in messages)
    assert any("event=finish status=completed" in message for message in messages)
