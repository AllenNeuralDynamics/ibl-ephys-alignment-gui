"""Small framework-agnostic event dispatcher for GUI workflow events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

EventT = TypeVar("EventT")
EventHandler = Callable[[EventT], None]
StoredHandler = Callable[[Any], None]


class EventSubscription:
    """Handle returned by :class:`EventBus` for explicit teardown."""

    def __init__(
        self,
        bus: EventBus,
        event_type: type[Any],
        handler: StoredHandler,
    ) -> None:
        self._bus = bus
        self._event_type = event_type
        self._handler = handler
        self._active = True

    @property
    def active(self) -> bool:
        """Whether this subscription is still registered."""
        return self._active

    def disconnect(self) -> None:
        """Remove this subscription if it is still active."""
        if not self._active:
            return
        self._bus.unsubscribe(self._event_type, self._handler)
        self._active = False


class EventBus:
    """Dispatch typed events without coupling document/runtime objects to Qt."""

    def __init__(self) -> None:
        self._handlers: dict[type[Any], list[StoredHandler]] = {}

    def subscribe(
        self,
        event_type: type[EventT],
        handler: EventHandler[EventT],
    ) -> EventSubscription:
        """Subscribe a handler to events with exactly ``event_type``."""
        stored_handler = cast(StoredHandler, handler)
        self._handlers.setdefault(event_type, []).append(stored_handler)
        return EventSubscription(self, event_type, stored_handler)

    def unsubscribe(
        self,
        event_type: type[Any],
        handler: StoredHandler,
    ) -> None:
        """Remove one handler subscription if present."""
        handlers = self._handlers.get(event_type)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            del self._handlers[event_type]

    def emit(self, event: Any) -> None:
        """Emit one event to subscribers in subscription order."""
        for handler in tuple(self._handlers.get(type(event), ())):
            handler(event)

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._handlers.clear()
