"""Tests for desktop alignment-selection action adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.app_results import (
    LoadedShankPrepared,
    PreviousAlignmentSelected,
)
from ephys_alignment_gui.desktop_alignment_selection_actions import (
    DesktopAlignmentSelectionActions,
    DesktopAlignmentSelectionCallbacks,
)
from ephys_alignment_gui.workflow import Failed


def _selected() -> PreviousAlignmentSelected:
    return PreviousAlignmentSelected(
        feature_prev=[1.0],
        track_prev=[2.0],
        choice="previous",
        choices=["previous"],
    )


class FakeCommands:
    def __init__(
        self,
        *,
        select_result: Any | None = None,
        prepare_result: Any | None = None,
    ) -> None:
        self.select_result = _selected() if select_result is None else select_result
        self.prepare_result = (
            LoadedShankPrepared(
                shank_idx=1,
                n_channels=384,
                histology_available=True,
                alignment_choices=["previous"],
            )
            if prepare_result is None
            else prepare_result
        )
        self.select_calls: list[int] = []
        self.prepare_calls: list[tuple[int, bool]] = []
        self.persistence = self
        self.loaded_shank = self

    def select_previous_alignment(self, idx: int) -> Any:
        self.select_calls.append(idx)
        return self.select_result

    def prepare_loaded_shank(
        self,
        shank_idx: int,
        *,
        select_default_alignment_if_empty: bool,
    ) -> Any:
        self.prepare_calls.append((shank_idx, select_default_alignment_if_empty))
        return self.prepare_result


class FakeCallbacks:
    def __init__(self) -> None:
        self.render_count = 0

    def render_loaded_shank_histology(self) -> bool:
        self.render_count += 1
        return True

    def ports(self) -> DesktopAlignmentSelectionCallbacks:
        return DesktopAlignmentSelectionCallbacks(
            render_loaded_shank_histology=self.render_loaded_shank_histology
        )


def _actions(
    *,
    data_loaded: bool = True,
    command_result: Any | None = None,
    prepare_result: Any | None = None,
) -> tuple[DesktopAlignmentSelectionActions, FakeCommands, FakeCallbacks]:
    commands = FakeCommands(
        select_result=command_result,
        prepare_result=prepare_result,
    )
    callbacks = FakeCallbacks()
    workspace_queries = SimpleNamespace(
        active_shank_selection=lambda: SimpleNamespace(
            shank_idx=1,
            data_loaded=data_loaded,
        )
    )
    app = SimpleNamespace(
        queries=SimpleNamespace(workspace=workspace_queries),
        commands=commands,
    )
    return (
        DesktopAlignmentSelectionActions(
            app=app,
            callbacks=callbacks.ports(),
        ),
        commands,
        callbacks,
    )


def test_alignment_selection_before_data_load_only_updates_choice() -> None:
    actions, commands, callbacks = _actions(data_loaded=False)

    assert actions.alignment_selected(2)

    assert commands.select_calls == [2]
    assert commands.prepare_calls == []
    assert callbacks.render_count == 0


def test_alignment_selection_prepares_loaded_shank_and_renders_histology() -> None:
    actions, commands, callbacks = _actions(data_loaded=True)

    assert actions.alignment_selected(3)

    assert commands.select_calls == [3]
    assert commands.prepare_calls == [(1, False)]
    assert callbacks.render_count == 1


def test_alignment_selection_without_histology_skips_render() -> None:
    actions, commands, callbacks = _actions(
        prepare_result=LoadedShankPrepared(
            shank_idx=1,
            n_channels=384,
            histology_available=False,
        )
    )

    assert actions.alignment_selected(0)

    assert commands.prepare_calls == [(1, False)]
    assert callbacks.render_count == 0


def test_alignment_selection_reports_select_or_prepare_failure() -> None:
    select_actions, select_commands, select_callbacks = _actions(
        command_result=Failed("bad alignment")
    )
    prepare_actions, prepare_commands, prepare_callbacks = _actions(
        prepare_result=Failed("bad prepare")
    )

    assert not select_actions.alignment_selected(0)
    assert not prepare_actions.alignment_selected(1)

    assert select_commands.select_calls == [0]
    assert select_commands.prepare_calls == []
    assert select_callbacks.render_count == 0
    assert prepare_commands.select_calls == [1]
    assert prepare_commands.prepare_calls == [(1, False)]
    assert prepare_callbacks.render_count == 0
