"""Tests for desktop session-selection coordination."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.application.results.metadata import RecordingSelected
from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.desktop.coordinators.session_selection_coordinator import (
    DesktopSessionSelectionCallbacks,
    DesktopSessionSelectionCoordinator,
)


class FakeSelectionView:
    def __init__(
        self,
        calls: list[tuple],
        *,
        session_name: str = "rec",
        sessions: list[str] | None = None,
    ) -> None:
        self.calls = calls
        self.session_name = session_name
        self.sessions = sessions or [session_name]

    def current_session(self) -> str:
        return self.session_name

    def session_at_index(self, idx: int) -> str | None:
        try:
            return self.sessions[idx]
        except IndexError:
            return None

    def populate_probes(self, probes: list[str]) -> None:
        self.calls.append(("populate-probes", probes))

    def clear_shanks(self) -> None:
        self.calls.append(("clear-shanks",))

    def set_load_data_enabled(self, enabled: bool) -> None:
        self.calls.append(("enable-load", enabled))

    def select_probe_index(self, idx: int) -> None:
        self.calls.append(("select-probe", idx))


class FakeCommands:
    def __init__(
        self,
        result: Any | None = None,
    ) -> None:
        self.result = result or RecordingSelected("rec", ["probeA", "probeB"])
        self.calls: list[str] = []
        self.ui_calls: list[tuple] | None = None
        self.metadata = self
        self.load = self

    def select_recording_metadata(self, recording_id: str):
        self.calls.append(recording_id)
        return self.result

    def detach_active_stream(self) -> None:
        if self.ui_calls is not None:
            self.ui_calls.append(("detach-app",))


def _coordinator(
    *,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
    mouse_root_loaded: bool = True,
    session_name: str = "rec",
    sessions: list[str] | None = None,
    active_session_name: str | None = None,
) -> tuple[DesktopSessionSelectionCoordinator, FakeCommands, list[tuple]]:
    calls = calls if calls is not None else []
    commands = commands or FakeCommands()
    commands.ui_calls = calls
    selection_view = FakeSelectionView(
        calls,
        session_name=session_name,
        sessions=sessions,
    )
    app = SimpleNamespace(
        commands=commands,
        queries=SimpleNamespace(
            workspace=SimpleNamespace(
                mouse_root_loaded=lambda: mouse_root_loaded,
                active_probe_selection_state=(
                    lambda: (
                        SimpleNamespace(
                            recording_id=active_session_name,
                            probe_name="probeA",
                        )
                        if active_session_name is not None
                        else None
                    )
                ),
            )
        ),
    )
    coordinator = DesktopSessionSelectionCoordinator(
        app=app,
        selection_view=selection_view,
        callbacks=DesktopSessionSelectionCallbacks(
            capture_pending_reference_lines=lambda: calls.append(("capture",)),
            show_empty_state=lambda: calls.append(("empty",)),
            select_first_probe=lambda: calls.append(("select-first-probe",)),
        ),
    )
    return coordinator, commands, calls


def test_session_selected_noops_without_mouse_root() -> None:
    coordinator, commands, calls = _coordinator(mouse_root_loaded=False)

    assert not coordinator.session_selected()

    assert commands.calls == []
    assert calls == []


def test_session_selected_noops_without_session_name() -> None:
    coordinator, commands, calls = _coordinator(session_name="")

    assert not coordinator.session_selected()

    assert commands.calls == []
    assert calls == []


def test_session_selected_noops_for_current_session() -> None:
    coordinator, commands, calls = _coordinator(active_session_name="rec")

    assert coordinator.session_selected()

    assert commands.calls == []
    assert calls == []


def test_session_selected_populates_probes_and_selects_first_probe() -> None:
    coordinator, commands, calls = _coordinator()

    assert coordinator.session_selected()

    assert commands.calls == ["rec"]
    assert calls == [
        ("capture",),
        ("detach-app",),
        ("empty",),
        ("populate-probes", ["probeA", "probeB"]),
        ("clear-shanks",),
        ("enable-load", False),
        ("select-probe", 0),
        ("select-first-probe",),
    ]


def test_session_selected_uses_activated_index_over_stale_current_text() -> None:
    commands = FakeCommands(
        result=RecordingSelected("rec2", ["probeC", "probeD"]),
    )
    coordinator, commands, calls = _coordinator(
        commands=commands,
        session_name="rec1",
        sessions=["rec1", "rec2"],
    )

    assert coordinator.session_selected(1)

    assert commands.calls == ["rec2"]
    assert ("populate-probes", ["probeC", "probeD"]) in calls


def test_session_selected_without_probes_does_not_select_first_probe() -> None:
    coordinator, _commands, calls = _coordinator(
        commands=FakeCommands(result=RecordingSelected("rec", []))
    )

    assert coordinator.session_selected()

    assert calls == [
        ("capture",),
        ("detach-app",),
        ("empty",),
        ("populate-probes", []),
        ("clear-shanks",),
        ("enable-load", False),
    ]


def test_session_selected_failure_does_not_mutate_selection_view() -> None:
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(result=Failed("recording failed"))
    )

    assert not coordinator.session_selected()

    assert commands.calls == ["rec"]
    assert calls == [
        ("capture",),
        ("detach-app",),
    ]


def test_session_selected_preserves_preload_and_stream_cache() -> None:
    coordinator, commands, calls = _coordinator()

    assert coordinator.session_selected()

    assert commands.calls == ["rec"]
    assert ("detach-app",) in calls
    assert not any(call[0] in {"cancel-preload", "evict-app"} for call in calls)
