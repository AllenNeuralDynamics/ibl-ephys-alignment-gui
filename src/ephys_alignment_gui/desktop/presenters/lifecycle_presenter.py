"""Desktop lifecycle presentation for stream/session transitions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.application.workflow import Failed
from ephys_alignment_gui.core.alignment_events import (
    StreamCacheEvicted,
    StreamDetached,
)
from ephys_alignment_gui.core.event_bus import EventSubscription

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopLifecycleCallbacks:
    """Desktop-only lifecycle side effects."""

    close_popups: Callable[[], None]
    reset_raw_image_payloads: Callable[[], None]
    show_empty_state: Callable[[], None]
    collect_garbage: Callable[[], None]


@dataclass
class DesktopLifecyclePresenter:
    """Coordinate desktop cleanup around app-owned lifecycle transitions."""

    app: Any
    displays: Any
    callbacks: DesktopLifecycleCallbacks

    def connect_lifecycle_events(self) -> list[EventSubscription]:
        """Subscribe desktop lifecycle cleanup to stream lifecycle events."""
        return [
            self.app.events.subscribe(StreamDetached, self.on_stream_detached),
            self.app.events.subscribe(StreamCacheEvicted, self.on_stream_cache_evicted),
        ]

    def on_stream_detached(self, _event: StreamDetached) -> None:
        """Clear desktop presentation after the active stream is detached."""
        self.clear_active_stream_presentation()
        self.reset_desktop_stream_state()

    def on_stream_cache_evicted(self, _event: StreamCacheEvicted) -> None:
        """Clear desktop presentation after cached streams are evicted."""
        self.clear_active_stream_presentation()
        self.reset_desktop_stream_state()
        self.callbacks.collect_garbage()

    def reset_desktop_stream_state(self) -> None:
        """Reset desktop-owned per-stream state that is not in the app model."""
        self.callbacks.reset_raw_image_payloads()

    def initialize_startup_stream_state(self) -> None:
        """Initialize stream-dependent app and desktop state at startup."""
        self.callbacks.close_popups()
        self.callbacks.reset_raw_image_payloads()
        self.app.commands.load.detach_active_stream()

    def clear_active_stream_presentation(self) -> None:
        """Clear plot handles and popups for the active stream presentation."""
        self.displays.reference_lines.clear()
        self.callbacks.close_popups()
        self.displays.ephys.clear()
        self.displays.slice.clear()
        self.displays.histology.clear()

    def detach_active_stream(self) -> None:
        """Detach the active app stream; event subscribers clear desktop state."""
        self.app.commands.load.detach_active_stream()

    def prepare_for_fresh_stream_load(self) -> None:
        """Clear desktop presentation after the app prepared a fresh stream load."""
        self.clear_active_stream_presentation()
        self.reset_desktop_stream_state()
        self.callbacks.collect_garbage()

    def evict_stream_cache(self) -> None:
        """Evict app stream cache; event subscribers clear desktop state."""
        result = self.app.commands.load.evict_stream_cache()
        if isinstance(result, Failed):
            logger.error(result.message)

    def show_empty_state(self) -> None:
        """Show the desktop empty-state placeholder."""
        self.callbacks.show_empty_state()
