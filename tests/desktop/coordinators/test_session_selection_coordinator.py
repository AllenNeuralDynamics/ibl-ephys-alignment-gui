"""Tests for desktop session-selection coordination."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.application.results.metadata import RecordingSelected
from ephys_alignment_gui.application.workflow import Failed
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
        *,
        evict_result: Any | None = None,
    ) -> None:
        self.result = result or RecordingSelected("rec", ["probeA", "probeB"])
        self.evict_result = evict_result
        self.calls: list[str] = []
        self.ui_calls: list[tuple] | None = None
        self.metadata = self
        self.load = self

    def select_recording_metadata(self, recording_id: str):
        self.calls.append(recording_id)
        return self.result

    def evict_stream_cache(self) -> Any:
        if self.ui_calls is not None:
            self.ui_calls.append(("evict-app",))
        return self.evict_result


def _coordinator(
    *,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
    mouse_root_loaded: bool = True,
    session_name: str = "rec",
    sessions: list[str] | None = None,
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
            )
        ),
    )
    coordinator = DesktopSessionSelectionCoordinator(
        app=app,
        selection_view=selection_view,
        callbacks=DesktopSessionSelectionCallbacks(
            capture_pending_reference_lines=lambda: calls.append(("capture",)),
            cancel_active_preload=lambda reason: calls.append(
                ("cancel-preload", reason)
            )
            or True,
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


def test_session_selected_populates_probes_and_selects_first_probe() -> None:
    coordinator, commands, calls = _coordinator()

    assert coordinator.session_selected()

    assert commands.calls == ["rec"]
    assert calls == [
        ("capture",),
        ("cancel-preload", "session changed"),
        ("evict-app",),
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
        ("cancel-preload", "session changed"),
        ("evict-app",),
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
        ("cancel-preload", "session changed"),
        ("evict-app",),
    ]


def test_session_selected_stops_when_cache_eviction_is_blocked() -> None:
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(evict_result=Failed("dirty runtime"))
    )

    assert not coordinator.session_selected()

    assert commands.calls == []
    assert calls == [
        ("capture",),
        ("cancel-preload", "session changed"),
        ("evict-app",),
    ]
