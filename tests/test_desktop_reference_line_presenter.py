"""Tests for desktop reference-line capture presentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop_reference_line_presenter import (
    DesktopReferenceLinePresenter,
)
from ephys_alignment_gui.workflow import Failed, Ok


class FakeCommands:
    def __init__(self, result: Any | None = None) -> None:
        self.result = Ok() if result is None else result
        self.captures: list[Any] = []

    def capture_active_reference_lines(self, positions: Any) -> Any:
        self.captures.append(positions)
        return self.result


class FakeReferenceLineDisplay:
    def __init__(self, positions: Any) -> None:
        self._positions = positions

    def positions(self) -> Any:
        return self._positions


def test_capture_pending_reference_lines_sends_display_positions_to_app() -> None:
    commands = FakeCommands()
    display = FakeReferenceLineDisplay(([1.0], [2.0]))
    presenter = DesktopReferenceLinePresenter(
        app=SimpleNamespace(commands=commands),
        reference_line_display=display,
    )

    assert presenter.capture_pending_reference_lines()

    assert commands.captures == [([1.0], [2.0])]


def test_capture_pending_reference_lines_reports_app_failure() -> None:
    commands = FakeCommands(result=Failed("capture failed"))
    display = FakeReferenceLineDisplay(None)
    presenter = DesktopReferenceLinePresenter(
        app=SimpleNamespace(commands=commands),
        reference_line_display=display,
    )

    assert not presenter.capture_pending_reference_lines()

    assert commands.captures == [None]
