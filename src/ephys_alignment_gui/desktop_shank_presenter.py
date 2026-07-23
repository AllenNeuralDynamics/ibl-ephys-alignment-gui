"""Desktop presenter for active shank changes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from ephys_alignment_gui.alignment_events import ShankChanged
from ephys_alignment_gui.event_bus import EventBus, EventSubscription
from ephys_alignment_gui.slice_display_policy import SliceSelection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopShankSelectionState:
    """Desktop selections to preserve across shank redraws."""

    previous_slice_selection: SliceSelection | None = None
    previous_slice_label: str | None = None
    previous_ephys_plot_keys: dict[str, str | None] | None = None


@dataclass(frozen=True)
class DesktopShankRenderCallbacks:
    """Desktop callbacks used to apply a shank selection."""

    apply_shank_selection: Callable[[int], None]
    resolve_preserve_plot_selection: Callable[[bool | None], bool]
    capture_plot_selection: Callable[[bool], DesktopShankSelectionState]
    clear_reference_lines: Callable[[], None]
    prepare_runtime: Callable[[int], None]
    prepare_histology: Callable[[int], bool]
    prepare_plot_data: Callable[[int], None]
    prepare_slice_data: Callable[[], bool]
    refresh_plot_menus: Callable[[bool, dict[str, str | None] | None], None]
    render_ephys_plots: Callable[[bool], None]
    render_histology_plots: Callable[[int], None]
    restore_slice_selection: Callable[[SliceSelection | None, str | None], None]
    configure_view: Callable[[bool], None]


@dataclass
class DesktopShankPresenter:
    """Coordinate desktop shank presentation from semantic shank events."""

    events: EventBus
    callbacks: DesktopShankRenderCallbacks | None = None

    def configure(self, *, callbacks: DesktopShankRenderCallbacks) -> None:
        """Attach desktop callbacks after view construction."""
        self.callbacks = callbacks

    def connect_shank_events(self) -> list[EventSubscription]:
        """Subscribe presenter handlers for shank selection events."""
        return [self.events.subscribe(ShankChanged, self.on_shank_changed)]

    def on_shank_changed(self, event: ShankChanged) -> None:
        """Present a semantic shank selection in the desktop shell."""
        if not event.data_loaded:
            self._require_callbacks().apply_shank_selection(event.shank_idx)
            logger.info("Data not loaded yet, shank index updated")
            return

        self.render_loaded_shank(
            shank_idx=event.shank_idx,
            preserve_plot_selection=event.preserve_plot_selection,
        )

    def render_loaded_shank(
        self,
        *,
        shank_idx: int,
        preserve_plot_selection: bool | None = None,
    ) -> None:
        """Render the loaded desktop view for one active shank."""
        callbacks = self._require_callbacks()
        preserve = callbacks.resolve_preserve_plot_selection(preserve_plot_selection)

        logger.info("Setting up view for shank index %s", shank_idx)
        callbacks.apply_shank_selection(shank_idx)
        selections = callbacks.capture_plot_selection(preserve)
        callbacks.clear_reference_lines()
        callbacks.prepare_runtime(shank_idx)
        if not callbacks.prepare_histology(shank_idx):
            return
        callbacks.prepare_plot_data(shank_idx)
        if not callbacks.prepare_slice_data():
            return
        callbacks.refresh_plot_menus(preserve, selections.previous_ephys_plot_keys)
        callbacks.render_ephys_plots(preserve)
        callbacks.render_histology_plots(shank_idx)
        callbacks.restore_slice_selection(
            selections.previous_slice_selection,
            selections.previous_slice_label,
        )
        callbacks.configure_view(preserve)
        logger.info("Shank view setup complete")

    def _require_callbacks(self) -> DesktopShankRenderCallbacks:
        if self.callbacks is None:
            raise RuntimeError("DesktopShankPresenter callbacks are not configured")
        return self.callbacks
