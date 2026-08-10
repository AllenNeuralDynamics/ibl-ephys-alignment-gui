"""Tests for desktop alignment edit action adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.app_results import AlignmentEditApplied, AlignmentEditNoop
from ephys_alignment_gui.desktop.alignment_edit_actions import (
    NUDGE_STEP_M,
    DesktopAlignmentEditActionCallbacks,
    DesktopAlignmentEditActions,
)
from ephys_alignment_gui.workflow import Failed


def _applied() -> AlignmentEditApplied:
    return AlignmentEditApplied(
        ActiveAlignment(
            np.array([0.0, 1.0]),
            np.array([2.0, 3.0]),
        )
    )


class FakeCommands:
    def __init__(self, result: Any | None = None) -> None:
        self.result = _applied() if result is None else result
        self.calls: list[Any] = []

    def fit_active_alignment_from_pending_reference_lines(self) -> Any:
        self.calls.append("fit")
        return self.result

    def offset_active_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        track_shift_m: float = 0.0,
    ) -> Any:
        self.calls.append(("offset", tip_position_um, track_shift_m))
        return self.result

    def nudge_active_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        track_shift_m: float,
    ) -> Any:
        self.calls.append(("nudge", tip_position_um, track_shift_m))
        return self.result

    def go_next_alignment(self) -> Any:
        self.calls.append("next")
        return self.result

    def go_previous_alignment(self) -> Any:
        self.calls.append("prev")
        return self.result

    def reset_active_alignment_to_initial(self) -> Any:
        self.calls.append("reset")
        return self.result


class FakeCallbacks:
    def __init__(
        self,
        *,
        histology_available: bool = True,
        capture_result: bool = True,
        tip_position_um: float | None = 125.0,
    ) -> None:
        self.histology_available_value = histology_available
        self.capture_result = capture_result
        self.tip_position_um_value = tip_position_um
        self.capture_count = 0

    def ports(self) -> DesktopAlignmentEditActionCallbacks:
        return DesktopAlignmentEditActionCallbacks(
            histology_available=lambda: self.histology_available_value,
            capture_pending_reference_lines=self.capture,
            tip_position_um=lambda: self.tip_position_um_value,
        )

    def capture(self) -> bool:
        self.capture_count += 1
        return self.capture_result


def test_fit_captures_reference_lines_before_command() -> None:
    commands = FakeCommands()
    callbacks = FakeCallbacks()
    actions = DesktopAlignmentEditActions(commands, callbacks.ports())

    assert actions.fit_button_pressed()

    assert callbacks.capture_count == 1
    assert commands.calls == ["fit"]


def test_fit_stops_when_histology_or_capture_is_unavailable() -> None:
    commands = FakeCommands()
    no_histology = FakeCallbacks(histology_available=False)
    failed_capture = FakeCallbacks(capture_result=False)

    no_histology_actions = DesktopAlignmentEditActions(
        commands,
        no_histology.ports(),
    )
    failed_capture_actions = DesktopAlignmentEditActions(
        commands,
        failed_capture.ports(),
    )

    assert not no_histology_actions.fit_button_pressed()
    assert not failed_capture_actions.fit_button_pressed()

    assert no_histology.capture_count == 0
    assert failed_capture.capture_count == 1
    assert commands.calls == []


def test_offset_uses_probe_tip_line_from_desktop() -> None:
    commands = FakeCommands()
    callbacks = FakeCallbacks(tip_position_um=321.0)
    actions = DesktopAlignmentEditActions(commands, callbacks.ports())

    assert actions.offset_button_pressed(track_shift_m=0.25)

    assert commands.calls == [("offset", 321.0, 0.25)]


def test_offset_stops_when_tip_line_is_missing() -> None:
    commands = FakeCommands()
    callbacks = FakeCallbacks(tip_position_um=None)
    actions = DesktopAlignmentEditActions(commands, callbacks.ports())

    assert not actions.offset_button_pressed()

    assert commands.calls == []


def test_move_buttons_delegate_to_bounded_nudge_command() -> None:
    commands = FakeCommands()
    actions = DesktopAlignmentEditActions(commands, FakeCallbacks().ports())

    assert actions.movedown_button_pressed()
    assert actions.moveup_button_pressed()

    assert commands.calls == [
        ("nudge", 125.0, -NUDGE_STEP_M),
        ("nudge", 125.0, NUDGE_STEP_M),
    ]


def test_history_and_reset_buttons_delegate_to_app_commands() -> None:
    commands = FakeCommands()
    actions = DesktopAlignmentEditActions(commands, FakeCallbacks().ports())

    assert actions.next_button_pressed()
    assert actions.prev_button_pressed()
    assert actions.reset_button_pressed()

    assert commands.calls == ["next", "prev", "reset"]


def test_failed_and_noop_results_are_not_applied() -> None:
    failed_actions = DesktopAlignmentEditActions(
        FakeCommands(result=Failed("bad edit")),
        FakeCallbacks().ports(),
    )
    noop_actions = DesktopAlignmentEditActions(
        FakeCommands(result=AlignmentEditNoop()),
        FakeCallbacks().ports(),
    )

    assert not failed_actions.next_button_pressed()
    assert not noop_actions.next_button_pressed()
