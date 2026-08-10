"""Tests for desktop session-selection presentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop_session_selection_presenter import (
    DesktopSessionSelectionCallbacks,
    DesktopSessionSelectionPresenter,
)
from ephys_alignment_gui.metadata_results import RecordingSelected
from ephys_alignment_gui.workflow import Failed


class FakeSelectionView:
    def __init__(
        self,
        calls: list[tuple],
        *,
        session_name: str = "rec",
    ) -> None:
        self.calls = calls
        self.session_name = session_name

    def current_session(self) -> str:
        return self.session_name

    def populate_probes(self, probes: list[str]) -> None:
        self.calls.append(("populate-probes", probes))

    def clear_shanks(self) -> None:
        self.calls.append(("clear-shanks",))

    def set_load_data_enabled(self, enabled: bool) -> None:
        self.calls.append(("enable-load", enabled))

    def select_probe_index(self, idx: int) -> None:
        self.calls.append(("select-probe", idx))


class FakeCommands:
    def __init__(self, result: Any | None = None) -> None:
        self.result = result or RecordingSelected("rec", ["probeA", "probeB"])
        self.calls: list[str] = []
        self.metadata = self

    def select_recording_metadata(self, recording_id: str):
        self.calls.append(recording_id)
        return self.result


def _presenter(
    *,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
    mouse_root_loaded: bool = True,
    session_name: str = "rec",
) -> tuple[DesktopSessionSelectionPresenter, FakeCommands, list[tuple]]:
    calls = calls if calls is not None else []
    commands = commands or FakeCommands()
    selection_view = FakeSelectionView(calls, session_name=session_name)
    app = SimpleNamespace(
        commands=commands,
        queries=SimpleNamespace(
            workspace=SimpleNamespace(
                mouse_root_loaded=lambda: mouse_root_loaded,
            )
        ),
    )
    presenter = DesktopSessionSelectionPresenter(
        app=app,
        selection_view=selection_view,
        callbacks=DesktopSessionSelectionCallbacks(
            capture_pending_reference_lines=lambda: calls.append(("capture",)),
            evict_stream_cache=lambda: calls.append(("evict",)),
            show_empty_state=lambda: calls.append(("empty",)),
            select_first_probe=lambda: calls.append(("select-first-probe",)),
        ),
    )
    return presenter, commands, calls


def test_session_selected_noops_without_mouse_root() -> None:
    presenter, commands, calls = _presenter(mouse_root_loaded=False)

    assert not presenter.session_selected()

    assert commands.calls == []
    assert calls == []


def test_session_selected_noops_without_session_name() -> None:
    presenter, commands, calls = _presenter(session_name="")

    assert not presenter.session_selected()

    assert commands.calls == []
    assert calls == []


def test_session_selected_populates_probes_and_selects_first_probe() -> None:
    presenter, commands, calls = _presenter()

    assert presenter.session_selected()

    assert commands.calls == ["rec"]
    assert calls == [
        ("capture",),
        ("evict",),
        ("empty",),
        ("populate-probes", ["probeA", "probeB"]),
        ("clear-shanks",),
        ("enable-load", False),
        ("select-probe", 0),
        ("select-first-probe",),
    ]


def test_session_selected_without_probes_does_not_select_first_probe() -> None:
    presenter, _commands, calls = _presenter(
        commands=FakeCommands(result=RecordingSelected("rec", []))
    )

    assert presenter.session_selected()

    assert calls == [
        ("capture",),
        ("evict",),
        ("empty",),
        ("populate-probes", []),
        ("clear-shanks",),
        ("enable-load", False),
    ]


def test_session_selected_failure_does_not_mutate_selection_view() -> None:
    presenter, commands, calls = _presenter(
        commands=FakeCommands(result=Failed("recording failed"))
    )

    assert not presenter.session_selected()

    assert commands.calls == ["rec"]
    assert calls == [("capture",), ("evict",)]
