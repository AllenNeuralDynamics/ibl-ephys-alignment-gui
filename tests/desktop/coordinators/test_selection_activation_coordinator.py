"""Tests for desktop selection-driven activation coordination."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.desktop.coordinators.selection_activation_coordinator import (
    DesktopSelectionActivationCoordinator,
)


class FakeSessionSelectionCoordinator:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.indices: list[int | None] = []

    def session_selected(self, idx: int | None = None) -> bool:
        self.indices.append(idx)
        return self.result


class FakeProbeSelectionCoordinator:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.indices: list[int | None] = []

    def probe_selected(self, idx: int | None = None) -> bool:
        self.indices.append(idx)
        return self.result


class FakeShankSelectionActions:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.count = 0

    def shank_selected(self) -> bool:
        self.count += 1
        return self.result


class FakeLoadPreflightCoordinator:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.count = 0

    def load_data_button_pressed(self) -> bool:
        self.count += 1
        return self.result


def _coordinator(
    *,
    session: Any | None = None,
    probe: Any | None = None,
    shank: Any | None = None,
    load: Any | None = None,
) -> DesktopSelectionActivationCoordinator:
    return DesktopSelectionActivationCoordinator(
        session_selection_coordinator=(session or FakeSessionSelectionCoordinator()),
        probe_selection_coordinator=probe or FakeProbeSelectionCoordinator(),
        shank_selection_actions=shank or FakeShankSelectionActions(),
        load_preflight_coordinator=load or FakeLoadPreflightCoordinator(),
    )


def test_session_selection_loads_after_selecting_metadata() -> None:
    session = FakeSessionSelectionCoordinator()
    load = FakeLoadPreflightCoordinator()
    coordinator = _coordinator(session=session, load=load)

    assert coordinator.session_selected(2)

    assert session.indices == [2]
    assert load.count == 1


def test_probe_selection_loads_after_selecting_metadata() -> None:
    probe = FakeProbeSelectionCoordinator()
    load = FakeLoadPreflightCoordinator()
    coordinator = _coordinator(probe=probe, load=load)

    assert coordinator.probe_selected(3)

    assert probe.indices == [3]
    assert load.count == 1


def test_shank_selection_loads_after_selecting_shank() -> None:
    shank = FakeShankSelectionActions()
    load = FakeLoadPreflightCoordinator()
    coordinator = _coordinator(shank=shank, load=load)

    assert coordinator.shank_selected(1)

    assert shank.count == 1
    assert load.count == 1


def test_failed_selection_does_not_load() -> None:
    session = FakeSessionSelectionCoordinator(result=False)
    load = FakeLoadPreflightCoordinator()
    coordinator = _coordinator(session=session, load=load)

    assert not coordinator.session_selected(2)

    assert session.indices == [2]
    assert load.count == 0


def test_explicit_load_uses_same_preflight_path() -> None:
    load = FakeLoadPreflightCoordinator()
    coordinator = _coordinator(load=load)

    assert coordinator.load_or_activate_selected_stream()

    assert load.count == 1


def test_load_preflight_failure_is_returned() -> None:
    load = FakeLoadPreflightCoordinator(result=False)
    coordinator = _coordinator(load=load)

    assert not coordinator.probe_selected(1)

    assert load.count == 1
