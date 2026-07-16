"""Tests for the framework-agnostic event bus."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.event_bus import EventBus


@dataclass(frozen=True)
class FirstEvent:
    value: int


@dataclass(frozen=True)
class SecondEvent:
    value: int


def test_event_bus_dispatches_exact_event_type_in_order() -> None:
    bus = EventBus()
    calls: list[tuple[str, int]] = []

    bus.subscribe(FirstEvent, lambda event: calls.append(("a", event.value)))
    bus.subscribe(FirstEvent, lambda event: calls.append(("b", event.value)))
    bus.subscribe(SecondEvent, lambda event: calls.append(("second", event.value)))

    bus.emit(FirstEvent(3))

    assert calls == [("a", 3), ("b", 3)]


def test_event_subscription_disconnects_idempotently() -> None:
    bus = EventBus()
    calls: list[int] = []
    subscription = bus.subscribe(FirstEvent, lambda event: calls.append(event.value))

    bus.emit(FirstEvent(1))
    subscription.disconnect()
    subscription.disconnect()
    bus.emit(FirstEvent(2))

    assert calls == [1]
    assert not subscription.active


def test_emit_uses_subscription_snapshot() -> None:
    bus = EventBus()
    calls: list[str] = []

    def first_handler(event: FirstEvent) -> None:
        calls.append(f"first:{event.value}")
        second_subscription.disconnect()

    def second_handler(event: FirstEvent) -> None:
        calls.append(f"second:{event.value}")

    bus.subscribe(FirstEvent, first_handler)
    second_subscription = bus.subscribe(FirstEvent, second_handler)

    bus.emit(FirstEvent(1))
    bus.emit(FirstEvent(2))

    assert calls == ["first:1", "second:1", "first:2"]
