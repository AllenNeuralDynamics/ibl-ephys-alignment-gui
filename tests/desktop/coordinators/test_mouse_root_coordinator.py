"""Tests for desktop mouse-root coordination."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.application.results.metadata import MouseRootLoaded
from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.desktop.coordinators.mouse_root_coordinator import (
    DesktopMouseRootCallbacks,
    DesktopMouseRootCoordinator,
)


class FakeBusyContext:
    def __init__(
        self,
        calls: list[tuple],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.calls = calls
        self.calls.append(("busy", args, kwargs))

    def __enter__(self):
        self.calls.append(("busy-enter",))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.calls.append(("busy-exit", exc_type))
        return False


class FakePathView:
    def __init__(self, calls: list[tuple], *, text: str = "/data/mouse") -> None:
        self.calls = calls
        self.text = text

    def mouse_root_widgets(self) -> list[str]:
        return ["button", "line"]

    def set_mouse_root(self, mouse_root: Path) -> None:
        self.calls.append(("path", mouse_root))

    def mouse_root_text(self) -> str:
        return self.text


class FakeSelectionView:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def populate_sessions(self, sessions: list[str]) -> None:
        self.calls.append(("sessions", sessions))

    def clear_probes(self) -> None:
        self.calls.append(("clear-probes",))

    def clear_shanks(self) -> None:
        self.calls.append(("clear-shanks",))

    def select_session_index(self, idx: int) -> None:
        self.calls.append(("select-session", idx))


class FakeCommands:
    def __init__(self, result: Any | None = None) -> None:
        self.result = result or MouseRootLoaded(
            SimpleNamespace(
                root=Path("/data/mouse"),
                mouse_id="mouse",
                sessions=["rec1"],
                probes={"rec1": {"probeA": object()}},
            ),
            root_changed=True,
        )
        self.calls: list[Path] = []
        self.clear_histology_calls = 0

    def set_mouse_root(self, mouse_root: Path):
        self.calls.append(mouse_root)
        return self.result

    def clear_histology_context(self) -> None:
        self.clear_histology_calls += 1


def _coordinator(
    *,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
    path_text: str = "/data/mouse",
    evict_result: Any | None = None,
) -> tuple[DesktopMouseRootCoordinator, FakeCommands, list[tuple]]:
    calls = calls if calls is not None else []
    commands = commands or FakeCommands()
    coordinator = DesktopMouseRootCoordinator(
        commands=commands,
        path_view=FakePathView(calls, text=path_text),
        selection_view=FakeSelectionView(calls),
        callbacks=DesktopMouseRootCallbacks(
            busy_context=lambda *args, **kwargs: FakeBusyContext(
                calls,
                *args,
                **kwargs,
            ),
            cancel_active_preload=lambda reason: (
                calls.append(("cancel-preload", reason)) or True
            ),
            evict_stream_cache=lambda: calls.append(("evict-app",)) or evict_result,
        ),
    )
    return coordinator, commands, calls


def test_set_mouse_root_populates_sessions_without_activating_default_session() -> None:
    coordinator, commands, calls = _coordinator()

    assert coordinator.set_mouse_root(Path("/data/mouse"))

    assert commands.calls == [Path("/data/mouse")]
    assert commands.clear_histology_calls == 1
    assert calls == [
        (
            "busy",
            ("Loading datapackage...", "Mouse root loaded"),
            {"disable_widgets": ["button", "line"]},
        ),
        ("busy-enter",),
        ("cancel-preload", "mouse root changed"),
        ("evict-app",),
        ("path", Path("/data/mouse")),
        ("sessions", ["rec1"]),
        ("select-session", -1),
        ("clear-probes",),
        ("clear-shanks",),
        ("busy-exit", None),
    ]


def test_set_mouse_root_without_sessions_does_not_select_session() -> None:
    result = MouseRootLoaded(
        SimpleNamespace(
            root=Path("/data/mouse"),
            mouse_id="mouse",
            sessions=[],
            probes={},
        ),
        root_changed=False,
    )
    coordinator, _commands, calls = _coordinator(commands=FakeCommands(result=result))

    assert coordinator.set_mouse_root(Path("/data/mouse"))

    assert _commands.clear_histology_calls == 0
    assert ("cancel-preload", "mouse root changed") not in calls
    assert ("evict-app",) not in calls
    assert ("select-session", 0) not in calls


def test_set_same_mouse_root_preserves_preload_and_stream_cache() -> None:
    result = MouseRootLoaded(
        SimpleNamespace(
            root=Path("/data/mouse"),
            mouse_id="mouse",
            sessions=["rec1"],
            probes={"rec1": {"probeA": object()}},
        ),
        root_changed=False,
    )
    coordinator, _commands, calls = _coordinator(commands=FakeCommands(result=result))

    assert coordinator.set_mouse_root(Path("/data/mouse"))

    assert _commands.clear_histology_calls == 0
    assert ("cancel-preload", "mouse root changed") not in calls
    assert ("evict-app",) not in calls


def test_set_mouse_root_failure_does_not_update_views() -> None:
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(result=Failed("bad root"))
    )

    assert not coordinator.set_mouse_root(Path("/data/missing"))

    assert commands.calls == [Path("/data/missing")]
    assert calls == [
        (
            "busy",
            ("Loading datapackage...", "Mouse root loaded"),
            {"disable_widgets": ["button", "line"]},
        ),
        ("busy-enter",),
        ("busy-exit", None),
    ]


def test_set_mouse_root_stops_when_cache_eviction_is_blocked() -> None:
    coordinator, commands, calls = _coordinator(
        evict_result=Failed("dirty runtime"),
    )

    assert not coordinator.set_mouse_root(Path("/data/mouse"))

    assert commands.calls == [Path("/data/mouse")]
    assert commands.clear_histology_calls == 1
    assert calls == [
        (
            "busy",
            ("Loading datapackage...", "Mouse root loaded"),
            {"disable_widgets": ["button", "line"]},
        ),
        ("busy-enter",),
        ("cancel-preload", "mouse root changed"),
        ("evict-app",),
        ("busy-exit", None),
    ]


def test_mouse_root_edited_ignores_empty_text() -> None:
    coordinator, commands, calls = _coordinator(path_text=" ")

    assert not coordinator.mouse_root_edited()

    assert commands.calls == []
    assert calls == []
