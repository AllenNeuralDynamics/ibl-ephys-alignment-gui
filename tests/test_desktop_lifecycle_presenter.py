"""Tests for desktop lifecycle presentation."""

from __future__ import annotations

from types import SimpleNamespace

from ephys_alignment_gui.desktop_lifecycle_presenter import (
    DesktopLifecycleCallbacks,
    DesktopLifecyclePresenter,
)


class FakeCommands:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls
        self.load = self

    def detach_active_stream(self) -> None:
        self.calls.append(("detach-app",))

    def evict_stream_cache(self) -> None:
        self.calls.append(("evict-app",))


class FakeDisplaySection:
    def __init__(self, calls: list[tuple], name: str) -> None:
        self.calls = calls
        self.name = name

    def clear(self) -> None:
        self.calls.append(("clear", self.name))


def _presenter(calls: list[tuple]) -> DesktopLifecyclePresenter:
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
    return DesktopLifecyclePresenter(
        app=SimpleNamespace(commands=FakeCommands(calls)),
        displays=displays,
        callbacks=callbacks,
    )


def test_detach_active_stream_clears_desktop_presentation_without_gc() -> None:
    calls: list[tuple] = []
    presenter = _presenter(calls)

    presenter.detach_active_stream()

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
    presenter = _presenter(calls)

    presenter.prepare_for_fresh_stream_load()

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
    presenter = _presenter(calls)

    presenter.evict_stream_cache()

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
