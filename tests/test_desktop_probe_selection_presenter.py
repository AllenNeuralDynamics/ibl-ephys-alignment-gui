"""Tests for desktop probe-selection presentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ephys_alignment_gui.controller import ProbeSelected
from ephys_alignment_gui.desktop_probe_selection_presenter import (
    DesktopProbeSelectionCallbacks,
    DesktopProbeSelectionPresenter,
)
from ephys_alignment_gui.workflow import Failed


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


class FakeCommands:
    def __init__(self, result: Any | None = None) -> None:
        self.result = result or ProbeSelected(
            recording_id="rec",
            probe_name="probeA",
            shanks=["1/2", "2/2"],
            n_shanks=2,
            output_directory=Path("/tmp/out"),
        )
        self.calls: list[tuple[str, str]] = []

    def select_probe_metadata(self, session_name: str, probe_name: str):
        self.calls.append((session_name, probe_name))
        return self.result


def _presenter(
    *,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
    mouse_root_loaded: bool = True,
    session_name: str = "rec",
    probe_name: str = "probeA",
    active_shank_idx: int = 1,
    cached: bool = False,
    selected_shank: int | None = 0,
) -> tuple[DesktopProbeSelectionPresenter, FakeCommands, list[tuple]]:
    calls = calls if calls is not None else []
    commands = commands or FakeCommands()
    presenter = DesktopProbeSelectionPresenter(
        commands=commands,
        callbacks=DesktopProbeSelectionCallbacks(
            mouse_root_loaded=lambda: mouse_root_loaded,
            session_name=lambda: session_name,
            probe_name=lambda: probe_name,
            active_shank_idx=lambda: active_shank_idx,
            capture_pending_reference_lines=lambda: calls.append(("capture",)),
            stash_and_detach_current=lambda: calls.append(("stash",)),
            present_cached_probe_selection=lambda session, probe, shank: (
                calls.append(("cached", session, probe, shank)) or cached
            ),
            show_empty_state=lambda: calls.append(("empty",)),
            busy_context=lambda *args, **kwargs: FakeBusyContext(
                calls,
                *args,
                **kwargs,
            ),
            selection_widgets=lambda: ["probe", "session"],
            populate_shanks=lambda shanks: calls.append(("populate", shanks)),
            init_session_variables=lambda: calls.append(("init",)),
            select_shank_for_view=lambda shank_idx, source: (
                calls.append(("select-shank", shank_idx, source)) or selected_shank
            ),
            display_output_directory=lambda path: calls.append(("output", path)),
            set_load_data_enabled=lambda enabled: calls.append(("enable", enabled)),
        ),
    )
    return presenter, commands, calls


def test_probe_selected_noops_without_mouse_root() -> None:
    presenter, commands, calls = _presenter(mouse_root_loaded=False)

    assert not presenter.probe_selected()

    assert commands.calls == []
    assert calls == []


def test_probe_selected_noops_without_session_or_probe() -> None:
    presenter, commands, calls = _presenter(session_name="")

    assert not presenter.probe_selected()

    assert commands.calls == []
    assert calls == []


def test_probe_selected_presents_cached_probe_without_channel_info_load() -> None:
    presenter, commands, calls = _presenter(cached=True)

    assert presenter.probe_selected()

    assert commands.calls == []
    assert calls == [
        ("capture",),
        ("stash",),
        ("cached", "rec", "probeA", 1),
    ]


def test_probe_selected_cache_miss_loads_channel_info_for_fresh_load() -> None:
    presenter, commands, calls = _presenter(cached=False)

    assert presenter.probe_selected()

    assert commands.calls == [("rec", "probeA")]
    assert ("empty",) in calls
    assert (
        "busy",
        ("Loading channel info...", "Ready"),
        {"disable_widgets": ["probe", "session"]},
    ) in calls
    assert ("populate", ["1/2", "2/2"]) in calls
    assert ("init",) in calls
    assert ("select-shank", 0, "probe-selected") in calls
    assert ("output", Path("/tmp/out")) in calls
    assert calls[-1] == ("enable", True)


def test_probe_selected_failure_disables_load_button() -> None:
    presenter, commands, calls = _presenter(
        commands=FakeCommands(result=Failed("channel info failed"))
    )

    assert not presenter.probe_selected()

    assert commands.calls == [("rec", "probeA")]
    assert ("enable", False) in calls
    assert ("init",) not in calls


def test_probe_selected_shank_selection_failure_disables_load_button() -> None:
    presenter, commands, calls = _presenter(selected_shank=None)

    assert not presenter.probe_selected()

    assert commands.calls == [("rec", "probeA")]
    assert ("select-shank", 0, "probe-selected") in calls
    assert ("enable", False) in calls
    assert ("output", Path("/tmp/out")) not in calls
