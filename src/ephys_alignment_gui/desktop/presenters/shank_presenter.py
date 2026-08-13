"""Desktop presenter for active shank changes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.application.results import LoadedShankPrepared
from ephys_alignment_gui.core.alignment_events import ShankChanged
from ephys_alignment_gui.core.alignment_read_models import (
    ActiveShankPlotDataState,
    ActiveShankScreenState,
    ActiveSliceMenuState,
    PreparedActiveShankScreenState,
)
from ephys_alignment_gui.core.event_bus import EventSubscription
from ephys_alignment_gui.core.slice_display_policy import SliceSelection
from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.plotting.menu_state import PlotMenuState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopShankSelectionState:
    """Desktop selections to preserve across shank redraws."""

    previous_slice_selection: SliceSelection | None = None
    previous_slice_label: str | None = None
    previous_ephys_plot_keys: dict[str, str | None] | None = None


@dataclass(frozen=True)
class DesktopShankRenderCallbacks:
    """Desktop callbacks used to render a shank selection."""

    capture_plot_selection: Callable[[bool], DesktopShankSelectionState]
    clear_reference_lines: Callable[[], None]
    render_alignment_choices: Callable[[list[str]], None]
    apply_plot_data_state: Callable[[ActiveShankPlotDataState], None]
    raw_image_payloads: Callable[[], Mapping[Any, Any]]
    render_plot_menus: Callable[[PlotMenuState], None]
    render_ephys_plots: Callable[[ActiveShankScreenState], None]
    render_histology_plots: Callable[[int], None]
    restore_slice_selection: Callable[
        [ActiveSliceMenuState | None, SliceSelection | None, str | None],
        None,
    ]
    configure_view: Callable[[bool], None]
    offline: Callable[[], bool]


@dataclass
class DesktopShankPresenter:
    """Coordinate desktop shank presentation from semantic shank events."""

    app: Any
    callbacks: DesktopShankRenderCallbacks | None = None

    def configure(self, *, callbacks: DesktopShankRenderCallbacks) -> None:
        """Attach desktop callbacks after view construction."""
        self.callbacks = callbacks

    def connect_shank_events(self) -> list[EventSubscription]:
        """Subscribe presenter handlers for shank selection events."""
        return [self.app.events.subscribe(ShankChanged, self.on_shank_changed)]

    def on_shank_changed(self, event: ShankChanged) -> None:
        """Present a semantic shank selection in the desktop shell."""
        if not event.data_loaded:
            logger.info("Data not loaded yet; document shank selection updated")
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
        preserve = self.app.queries.workspace.resolve_shank_preserve_plot_selection(
            preserve_plot_selection
        )

        logger.info("Setting up view for shank index %s", shank_idx)
        selections = callbacks.capture_plot_selection(preserve)
        callbacks.clear_reference_lines()
        prepared = self.app.commands.loaded_shank.prepare_loaded_shank(shank_idx)
        if isinstance(prepared, Failed):
            logger.error(prepared.message)
            return
        assert isinstance(prepared, LoadedShankPrepared)
        logger.debug(
            "Selected %s channels for shank index %s",
            prepared.n_channels,
            prepared.shank_idx,
        )
        if prepared.alignment_choices is not None:
            callbacks.render_alignment_choices(prepared.alignment_choices)

        screen_preparation = (
            self.app.queries.active_shank.prepare_active_shank_screen_state(
                histology_available=prepared.histology_available,
                preserve_plot_selection=preserve,
                previous_ephys_plot_keys=selections.previous_ephys_plot_keys,
                raw_image_payloads=callbacks.raw_image_payloads(),
                previous_slice_selection=selections.previous_slice_selection,
                offline=callbacks.offline(),
            )
        )
        if screen_preparation.missing_plot_data:
            raise RuntimeError("No active stream runtime for shank plot data")
        assert screen_preparation.plot_data is not None
        self.app.commands.display.set_probe_limits(
            min(0.0, float(screen_preparation.plot_data.channel_min_um)),
            float(screen_preparation.plot_data.channel_max_um),
        )
        callbacks.apply_plot_data_state(screen_preparation.plot_data)
        if screen_preparation.missing_required_slice_data:
            raise RuntimeError("Could not build active slice data")
        screen_state = self._require_screen_state(screen_preparation)
        callbacks.render_plot_menus(screen_state.plot_menu)
        callbacks.render_ephys_plots(screen_state)
        callbacks.restore_slice_selection(
            screen_state.slice_menu,
            selections.previous_slice_selection,
            selections.previous_slice_label,
        )
        if prepared.histology_available:
            callbacks.render_histology_plots(shank_idx)
        callbacks.configure_view(preserve)
        logger.info("Shank view setup complete")

    def _require_callbacks(self) -> DesktopShankRenderCallbacks:
        if self.callbacks is None:
            raise RuntimeError("DesktopShankPresenter callbacks are not configured")
        return self.callbacks

    @staticmethod
    def _require_screen_state(
        screen_preparation: PreparedActiveShankScreenState,
    ) -> ActiveShankScreenState:
        if screen_preparation.screen is None:
            raise RuntimeError("Could not build active shank screen state")
        return screen_preparation.screen
