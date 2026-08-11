"""Tests for desktop probe-selection coordination."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.application.results import ShankSelected
from ephys_alignment_gui.application.results.metadata import ProbeSelected
from ephys_alignment_gui.application.workflow import Failed
from ephys_alignment_gui.desktop.coordinators.probe_selection_coordinator import (
    DesktopProbeSelectionCallbacks,
    DesktopProbeSelectionCoordinator,
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


class FakeCommands:
    def __init__(
        self,
        *,
        probe_result: Any | None = None,
        shank_result: Any | None = None,
    ) -> None:
        self.probe_result = probe_result or ProbeSelected(
            recording_id="rec",
            probe_name="probeA",
            shanks=["1/2", "2/2"],
            n_shanks=2,
            output_directory=Path("/tmp/out"),
        )
        self.shank_result = shank_result
        self.ui_calls: list[tuple] | None = None
        self.calls: list[tuple[str, str]] = []
        self.shank_calls: list[tuple[int, str]] = []
        self.metadata = self
        self.shanks = self
        self.load = self

    def select_probe_metadata(self, session_name: str, probe_name: str):
        self.calls.append((session_name, probe_name))
        return self.probe_result

    def select_shank(self, shank_idx: int, *, source: str) -> Any:
        self.shank_calls.append((shank_idx, source))
        return self.shank_result or ShankSelected(
            previous_key=None,
            selected_key=None,
            previous_shank_idx=1,
            shank_idx=shank_idx,
            data_loaded=False,
        )

    def detach_active_stream(self) -> None:
        if self.ui_calls is not None:
            self.ui_calls.append(("detach-app",))


class FakeSelectionView:
    def __init__(
        self,
        calls: list[tuple],
        *,
        session_name: str = "rec",
        probe_name: str = "probeA",
        probes: list[str] | None = None,
    ) -> None:
        self.calls = calls
        self.session_name = session_name
        self.probe_name = probe_name
        self.probes = probes or [probe_name]

    def current_session(self) -> str:
        return self.session_name

    def current_probe(self) -> str:
        return self.probe_name

    def probe_at_index(self, idx: int) -> str | None:
        try:
            return self.probes[idx]
        except IndexError:
            return None

    def selection_widgets(self) -> list[str]:
        return ["probe", "session"]

    def populate_probe_shanks(self, shanks: list[str]) -> None:
        self.calls.append(("populate", shanks))

    def set_load_data_enabled(self, enabled: bool) -> None:
        self.calls.append(("enable", enabled))


def _coordinator(
    *,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
    mouse_root_loaded: bool = True,
    session_name: str = "rec",
    probe_name: str = "probeA",
    probes: list[str] | None = None,
    active_shank_idx: int = 1,
    cached: bool = False,
) -> tuple[DesktopProbeSelectionCoordinator, FakeCommands, list[tuple]]:
    calls = calls if calls is not None else []
    commands = commands or FakeCommands()
    commands.ui_calls = calls
    selection_view = FakeSelectionView(
        calls,
        session_name=session_name,
        probe_name=probe_name,
        probes=probes,
    )
    app = SimpleNamespace(
        commands=commands,
        queries=SimpleNamespace(
            workspace=SimpleNamespace(
                mouse_root_loaded=lambda: mouse_root_loaded,
                active_shank_selection=lambda: SimpleNamespace(
                    shank_idx=active_shank_idx
                ),
            )
        ),
    )
    coordinator = DesktopProbeSelectionCoordinator(
        app=app,
        selection_view=selection_view,
        callbacks=DesktopProbeSelectionCallbacks(
            capture_pending_reference_lines=lambda: calls.append(("capture",)),
            present_cached_probe_selection=lambda session, probe, shank: (
                calls.append(("cached", session, probe, shank)) or cached
            ),
            show_empty_state=lambda: calls.append(("empty",)),
            busy_context=lambda *args, **kwargs: FakeBusyContext(
                calls,
                *args,
                **kwargs,
            ),
        ),
    )
    return coordinator, commands, calls


def test_probe_selected_noops_without_mouse_root() -> None:
    coordinator, commands, calls = _coordinator(mouse_root_loaded=False)

    assert not coordinator.probe_selected()

    assert commands.calls == []
    assert calls == []


def test_probe_selected_noops_without_session_or_probe() -> None:
    coordinator, commands, calls = _coordinator(session_name="")

    assert not coordinator.probe_selected()

    assert commands.calls == []
    assert calls == []


def test_probe_selected_presents_cached_probe_without_channel_info_load() -> None:
    coordinator, commands, calls = _coordinator(cached=True)

    assert coordinator.probe_selected()

    assert commands.calls == []
    assert calls == [
        ("capture",),
        ("cached", "rec", "probeA", 1),
    ]


def test_probe_selected_uses_activated_index_over_stale_current_text() -> None:
    coordinator, commands, calls = _coordinator(
        probe_name="probeA",
        probes=["probeA", "probeB"],
        cached=False,
    )

    assert coordinator.probe_selected(1)

    assert commands.calls == [("rec", "probeB")]
    assert ("cached", "rec", "probeB", 1) in calls


def test_probe_selected_cache_miss_loads_channel_info_for_fresh_load() -> None:
    coordinator, commands, calls = _coordinator(cached=False)

    assert coordinator.probe_selected()

    assert commands.calls == [("rec", "probeA")]
    assert ("detach-app",) in calls
    assert ("empty",) in calls
    assert (
        "busy",
        ("Loading channel info...", "Ready"),
        {"disable_widgets": ["probe", "session"]},
    ) in calls
    assert ("populate", ["1/2", "2/2"]) in calls
    assert commands.shank_calls == [(0, "probe-selected")]
    assert calls[-1] == ("enable", True)


def test_probe_selected_failure_disables_load_button() -> None:
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(probe_result=Failed("channel info failed"))
    )

    assert not coordinator.probe_selected()

    assert commands.calls == [("rec", "probeA")]
    assert ("enable", False) in calls
    assert ("detach-app",) in calls


def test_probe_selected_shank_selection_failure_disables_load_button() -> None:
    coordinator, commands, calls = _coordinator(
        commands=FakeCommands(shank_result=Failed("bad shank"))
    )

    assert not coordinator.probe_selected()

    assert commands.calls == [("rec", "probeA")]
    assert commands.shank_calls == [(0, "probe-selected")]
    assert ("enable", False) in calls
