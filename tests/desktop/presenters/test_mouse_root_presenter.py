"""Tests for desktop mouse-root presentation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.application.results.metadata import MouseRootLoaded
from ephys_alignment_gui.application.workflow import Failed
from ephys_alignment_gui.desktop.presenters.mouse_root_presenter import (
    DesktopMouseRootCallbacks,
    DesktopMouseRootPresenter,
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

    def set_load_data_enabled(self, enabled: bool) -> None:
        self.calls.append(("enable-load", enabled))

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


def _presenter(
    *,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
    path_text: str = "/data/mouse",
) -> tuple[DesktopMouseRootPresenter, FakeCommands, list[tuple]]:
    calls = calls if calls is not None else []
    commands = commands or FakeCommands()
    presenter = DesktopMouseRootPresenter(
        commands=commands,
        path_view=FakePathView(calls, text=path_text),
        selection_view=FakeSelectionView(calls),
        callbacks=DesktopMouseRootCallbacks(
            busy_context=lambda *args, **kwargs: FakeBusyContext(
                calls,
                *args,
                **kwargs,
            ),
            select_first_session=lambda: calls.append(("select-first-session",)),
        ),
    )
    return presenter, commands, calls


def test_set_mouse_root_populates_sessions_and_selects_first_session() -> None:
    presenter, commands, calls = _presenter()

    assert presenter.set_mouse_root(Path("/data/mouse"))

    assert commands.calls == [Path("/data/mouse")]
    assert commands.clear_histology_calls == 1
    assert calls == [
        (
            "busy",
            ("Loading datapackage...", "Mouse root loaded"),
            {"disable_widgets": ["button", "line"]},
        ),
        ("busy-enter",),
        ("path", Path("/data/mouse")),
        ("sessions", ["rec1"]),
        ("clear-probes",),
        ("clear-shanks",),
        ("enable-load", False),
        ("select-session", 0),
        ("select-first-session",),
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
    presenter, _commands, calls = _presenter(commands=FakeCommands(result=result))

    assert presenter.set_mouse_root(Path("/data/mouse"))

    assert _commands.clear_histology_calls == 0
    assert ("select-session", 0) not in calls
    assert ("select-first-session",) not in calls


def test_set_mouse_root_failure_does_not_update_views() -> None:
    presenter, commands, calls = _presenter(
        commands=FakeCommands(result=Failed("bad root"))
    )

    assert not presenter.set_mouse_root(Path("/data/missing"))

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


def test_mouse_root_edited_disables_load_data_for_empty_text() -> None:
    presenter, commands, calls = _presenter(path_text=" ")

    assert not presenter.mouse_root_edited()

    assert commands.calls == []
    assert calls == [("enable-load", False)]
