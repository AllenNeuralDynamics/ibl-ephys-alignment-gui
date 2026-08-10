"""Tests for desktop shank-selection action adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.controller import ShankSelected
from ephys_alignment_gui.desktop_shank_selection_actions import (
    DesktopShankSelectionActions,
)
from ephys_alignment_gui.workflow import Failed


class FakeSelectionView:
    def __init__(self, shank_idx: int | None) -> None:
        self.shank_idx = shank_idx

    def current_shank_index(self) -> int | None:
        return self.shank_idx


class FakeReferenceLineDisplay:
    def __init__(self) -> None:
        self.positions_count = 0
        self.current_positions = ([1.0], [2.0])

    def positions(self) -> Any:
        self.positions_count += 1
        return self.current_positions


class FakeCommands:
    def __init__(self, result: Any | None = None) -> None:
        self.result = result
        self.calls: list[tuple[Any, Any, str]] = []
        self.shanks = self

    def select_shank(
        self,
        shank_idx: int,
        *,
        outgoing_reference_lines: Any,
        source: str,
    ) -> Any:
        self.calls.append((shank_idx, outgoing_reference_lines, source))
        return self.result or ShankSelected(
            previous_key=None,
            selected_key=None,
            previous_shank_idx=0,
            shank_idx=shank_idx,
            data_loaded=True,
        )


def _actions(
    *,
    requested_shank_idx: int | None = 1,
    active_shank_idx: int = 0,
    command_result: Any | None = None,
) -> tuple[DesktopShankSelectionActions, FakeCommands, FakeReferenceLineDisplay]:
    commands = FakeCommands(command_result)
    reference_lines = FakeReferenceLineDisplay()
    app = SimpleNamespace(
        queries=SimpleNamespace(
            workspace=SimpleNamespace(
                active_shank_selection=lambda: SimpleNamespace(
                    shank_idx=active_shank_idx
                )
            )
        ),
        commands=commands,
    )
    return (
        DesktopShankSelectionActions(
            app=app,
            selection_view=FakeSelectionView(requested_shank_idx),
            reference_line_display=reference_lines,
        ),
        commands,
        reference_lines,
    )


def test_shank_selection_noops_for_invalid_or_current_shank() -> None:
    invalid_actions, invalid_commands, invalid_reference_lines = _actions(
        requested_shank_idx=None
    )
    current_actions, current_commands, current_reference_lines = _actions(
        requested_shank_idx=1,
        active_shank_idx=1,
    )

    assert not invalid_actions.shank_selected()
    assert current_actions.shank_selected()

    assert invalid_commands.calls == []
    assert current_commands.calls == []
    assert invalid_reference_lines.positions_count == 0
    assert current_reference_lines.positions_count == 0


def test_shank_selection_captures_lines_and_selects_requested_shank() -> None:
    actions, commands, reference_lines = _actions(requested_shank_idx=2)

    assert actions.shank_selected()

    assert reference_lines.positions_count == 1
    assert commands.calls == [(2, ([1.0], [2.0]), "dropdown")]


def test_shank_selection_reports_command_failure() -> None:
    actions, commands, reference_lines = _actions(command_result=Failed("bad shank"))

    assert not actions.shank_selected()

    assert reference_lines.positions_count == 1
    assert commands.calls == [(1, ([1.0], [2.0]), "dropdown")]
