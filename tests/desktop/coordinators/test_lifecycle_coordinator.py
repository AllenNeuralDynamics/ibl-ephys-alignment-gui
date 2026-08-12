"""Tests for desktop lifecycle coordination."""

from __future__ import annotations

from types import SimpleNamespace

from ephys_alignment_gui.core.alignment_events import (
    StreamCacheEvicted,
    StreamDetached,
)
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.desktop.coordinators.lifecycle_coordinator import (
    DesktopLifecycleCallbacks,
    DesktopLifecycleCoordinator,
)


class FakeCommands:
    def __init__(
        self,
        calls: list[tuple],
        events: EventBus,
        *,
        evict_result=None,
    ) -> None:
        self.calls = calls
        self.events = events
        self.evict_result = evict_result
        self.load = self

    def detach_active_stream(self) -> None:
        self.calls.append(("detach-app",))
        self.events.emit(StreamDetached(cached_stream_count=1))

    def evict_stream_cache(self) -> None:
        self.calls.append(("evict-app",))
        if self.evict_result is not None:
            return self.evict_result
        self.events.emit(StreamCacheEvicted(evicted_stream_count=2))


class FakeDisplaySection:
    def __init__(self, calls: list[tuple], name: str) -> None:
        self.calls = calls
        self.name = name

    def clear(self) -> None:
        self.calls.append(("clear", self.name))


def _coordinator(
    calls: list[tuple],
    *,
    evict_result=None,
) -> DesktopLifecycleCoordinator:
    events = EventBus()
    displays = SimpleNamespace(
        reference_lines=FakeDisplaySection(calls, "reference-lines"),
        ephys=FakeDisplaySection(calls, "ephys"),
        slice=FakeDisplaySection(calls, "slice"),
        histology=FakeDisplaySection(calls, "histology"),
    )
    callbacks = DesktopLifecycleCallbacks(
        close_popups=lambda: calls.append(("close-popups",)),
        reset_raw_image_payloads=lambda: calls.append(("reset-raw-images",)),
        show_empty_state=lambda: calls.append(("empty",)),
        collect_garbage=lambda: calls.append(("gc",)),
    )
    return DesktopLifecycleCoordinator(
        app=SimpleNamespace(
            events=events,
            commands=FakeCommands(calls, events, evict_result=evict_result),
        ),
        displays=displays,
        callbacks=callbacks,
    )


def test_detach_active_stream_clears_desktop_coordination_without_gc() -> None:
    calls: list[tuple] = []
    coordinator = _coordinator(calls)
    coordinator.connect_lifecycle_events()

    coordinator.detach_active_stream()

    assert calls == [
        ("detach-app",),
        ("clear", "reference-lines"),
        ("close-popups",),
        ("clear", "ephys"),
        ("clear", "slice"),
        ("clear", "histology"),
        ("reset-raw-images",),
    ]


def test_prepare_for_fresh_stream_load_clears_desktop_state_after_app_prepare() -> None:
    calls: list[tuple] = []
    coordinator = _coordinator(calls)

    coordinator.prepare_for_fresh_stream_load()

    assert calls == [
        ("clear", "reference-lines"),
        ("close-popups",),
        ("clear", "ephys"),
        ("clear", "slice"),
        ("clear", "histology"),
        ("reset-raw-images",),
        ("gc",),
    ]


def test_evict_stream_cache_clears_app_cache_and_desktop_state() -> None:
    calls: list[tuple] = []
    coordinator = _coordinator(calls)
    coordinator.connect_lifecycle_events()

    coordinator.evict_stream_cache()

    assert calls == [
        ("evict-app",),
        ("clear", "reference-lines"),
        ("close-popups",),
        ("clear", "ephys"),
        ("clear", "slice"),
        ("clear", "histology"),
        ("reset-raw-images",),
        ("gc",),
    ]


def test_evict_stream_cache_blocked_does_not_clear_desktop_state() -> None:
    calls: list[tuple] = []
    coordinator = _coordinator(calls, evict_result=Failed("dirty runtime"))
    coordinator.connect_lifecycle_events()

    coordinator.evict_stream_cache()

    assert calls == [("evict-app",)]
