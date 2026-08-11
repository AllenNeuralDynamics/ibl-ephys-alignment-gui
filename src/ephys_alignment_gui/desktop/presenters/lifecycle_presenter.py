"""Desktop lifecycle presentation for stream/session transitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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
        """Detach the active app stream and clear its desktop presentation."""
        self.app.commands.load.detach_active_stream()
        self.clear_active_stream_presentation()
        self.reset_desktop_stream_state()

    def prepare_for_fresh_stream_load(self) -> None:
        """Clear desktop presentation after the app prepared a fresh stream load."""
        self.clear_active_stream_presentation()
        self.reset_desktop_stream_state()
        self.callbacks.collect_garbage()

    def evict_stream_cache(self) -> None:
        """Evict app stream cache and clear desktop presentation state."""
        self.app.commands.load.evict_stream_cache()
        self.clear_active_stream_presentation()
        self.reset_desktop_stream_state()
        self.callbacks.collect_garbage()

    def show_empty_state(self) -> None:
        """Show the desktop empty-state placeholder."""
        self.callbacks.show_empty_state()
