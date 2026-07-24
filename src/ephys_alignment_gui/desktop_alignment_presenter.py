"""Desktop presenter for alignment edit rendering."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from ephys_alignment_gui.alignment_events import (
    AlignmentEdited,
    AlignmentEditKind,
)
from ephys_alignment_gui.alignment_read_models import (
    ActiveAlignmentRenderState,
    FitPlotRenderState,
    HistologyPanelRenderState,
    ScaleFactorRenderState,
)
from ephys_alignment_gui.event_bus import EventBus, EventSubscription

logger = logging.getLogger(__name__)

LineUpdateMode = Literal[
    "none",
    "reattach",
    "sync_to_alignment",
    "reset_to_previous",
]


@dataclass(frozen=True)
class DesktopAlignmentPresentationOptions:
    """Desktop render behavior for one alignment edit kind."""

    line_update: LineUpdateMode = "none"
    reset_histology_range: bool = False
    refresh_perpendicular: bool = True
    preserve_depth_range: bool = False
    clear_reference_lines: bool = False


def desktop_presentation_options_for_edit(
    edit_kind: AlignmentEditKind,
) -> DesktopAlignmentPresentationOptions:
    """Return desktop presentation policy for an application alignment edit."""
    if edit_kind == "fit":
        return DesktopAlignmentPresentationOptions(
            line_update="sync_to_alignment",
            preserve_depth_range=True,
        )
    if edit_kind == "offset":
        return DesktopAlignmentPresentationOptions(line_update="sync_to_alignment")
    if edit_kind in {"next", "previous"}:
        return DesktopAlignmentPresentationOptions(line_update="reattach")
    return DesktopAlignmentPresentationOptions(
        line_update="reset_to_previous",
        reset_histology_range=True,
        clear_reference_lines=True,
    )


@dataclass(frozen=True)
class DesktopAlignmentRenderCallbacks:
    """Desktop callbacks used to apply alignment render updates."""

    restore_lin_fit: Callable[[bool | None], None]
    clear_reference_lines: Callable[[], None]
    capture_depth_plot_y_ranges: Callable[[], Any]
    restore_depth_plot_y_ranges: Callable[[Any], None]
    reattach_reference_lines: Callable[[], None]
    probe_extent_query_kwargs: Callable[[], dict[str, float]]
    fit_depth_um: Callable[[], Any]
    lin_fit_enabled: Callable[[], bool]
    scale_factor_y_range: Callable[[], tuple[float, float]]
    render_histology: Callable[[HistologyPanelRenderState], None]
    render_scale_factor: Callable[
        [ScaleFactorRenderState, tuple[float, float]],
        None,
    ]
    render_fit: Callable[[FitPlotRenderState], None]
    plot_channels: Callable[[Any], None]
    refresh_perpendicular_histology: Callable[[], None]
    update_reference_lines_to_alignment: Callable[[], None]
    create_reference_lines_for_previous_alignment: Callable[[], None]
    set_default_feature_y_range: Callable[[], None]
    update_status: Callable[[], None]


@dataclass
class DesktopAlignmentPresenter:
    """Coordinate desktop alignment presentation from app read models."""

    events: EventBus
    queries: Any | None = None
    callbacks: DesktopAlignmentRenderCallbacks | None = None

    def configure(
        self,
        *,
        queries: Any,
        callbacks: DesktopAlignmentRenderCallbacks,
    ) -> None:
        """Attach app queries and desktop callbacks after view construction."""
        self.queries = queries
        self.callbacks = callbacks

    def connect_alignment_events(self) -> list[EventSubscription]:
        """Subscribe presenter handlers for alignment edit/render events."""
        return [self.events.subscribe(AlignmentEdited, self.on_alignment_edited)]

    def on_alignment_edited(self, event: AlignmentEdited) -> None:
        """Present a semantic alignment edit in the desktop shell."""
        queries = self._require_queries()
        callbacks = self._require_callbacks()
        options = desktop_presentation_options_for_edit(event.edit_kind)
        callbacks.restore_lin_fit(event.lin_fit)
        render_state = queries.active_alignment_render_state()
        if render_state is None:
            logger.error(
                "Cannot refresh alignment: active alignment data is not loaded"
            )
            return
        if options.clear_reference_lines:
            callbacks.clear_reference_lines()
        depth_ranges = (
            callbacks.capture_depth_plot_y_ranges()
            if options.preserve_depth_range
            else {}
        )
        try:
            self.render_alignment_edit(
                render_state=render_state,
                options=options,
            )
        finally:
            if depth_ranges:
                callbacks.restore_depth_plot_y_ranges(depth_ranges)

    def render_alignment_edit(
        self,
        *,
        render_state: ActiveAlignmentRenderState,
        options: DesktopAlignmentPresentationOptions,
    ) -> None:
        """Apply one alignment edit to focused desktop render callbacks."""
        callbacks = self._require_callbacks()
        histology_state = self._histology_panel_state(render_state)
        scale_state = self._scale_factor_state(render_state, histology_state)
        fit_state = self._fit_plot_state()

        self._prepare_reference_lines_before_render(options)
        if histology_state is not None:
            callbacks.render_histology(histology_state)
        if scale_state is not None:
            callbacks.render_scale_factor(
                scale_state,
                callbacks.scale_factor_y_range(),
            )
        if fit_state is not None:
            callbacks.render_fit(fit_state)
        callbacks.plot_channels(render_state.projection)
        if options.refresh_perpendicular:
            callbacks.refresh_perpendicular_histology()
        self._update_reference_lines_after_render(options)
        if options.reset_histology_range:
            callbacks.set_default_feature_y_range()
        callbacks.update_status()

    def _histology_panel_state(
        self,
        render_state: ActiveAlignmentRenderState,
    ) -> HistologyPanelRenderState | None:
        """Build aligned-histology panel state from the active render DTO."""
        probe_extent = self._require_queries().probe_extent_render_state(
            render_state.active_alignment,
            **self._require_callbacks().probe_extent_query_kwargs(),
        )
        if probe_extent is None:
            logger.error("Cannot render histology: active probe extent is not loaded")
            return None
        return HistologyPanelRenderState(
            key=render_state.key,
            histology=render_state.histology,
            probe_extent=probe_extent,
        )

    def _scale_factor_state(
        self,
        render_state: ActiveAlignmentRenderState,
        histology_state: HistologyPanelRenderState | None,
    ) -> ScaleFactorRenderState | None:
        """Build scale-factor panel state from the active render DTO."""
        if histology_state is None:
            return None
        return ScaleFactorRenderState(
            key=render_state.key,
            region=render_state.histology.scale.region,
            scale=render_state.histology.scale.scale,
            probe_extent=histology_state.probe_extent,
        )

    def _fit_plot_state(self) -> FitPlotRenderState | None:
        """Return feature/track fit plot state for the active alignment."""
        callbacks = self._require_callbacks()
        state = self._require_queries().active_fit_plot_state(
            depth_um=callbacks.fit_depth_um(),
            lin_fit=callbacks.lin_fit_enabled(),
        )
        if state is None:
            logger.error("Cannot render fit: active alignment data is not loaded")
        return state

    def _prepare_reference_lines_before_render(
        self,
        options: DesktopAlignmentPresentationOptions,
    ) -> None:
        """Prepare reference-line handles before plot refreshes."""
        if options.line_update == "reattach":
            self._require_callbacks().reattach_reference_lines()

    def _update_reference_lines_after_render(
        self,
        options: DesktopAlignmentPresentationOptions,
    ) -> None:
        """Update desktop reference-line handles after plot refreshes."""
        callbacks = self._require_callbacks()
        if options.line_update == "reattach":
            callbacks.reattach_reference_lines()
        elif options.line_update == "sync_to_alignment":
            callbacks.reattach_reference_lines()
            callbacks.update_reference_lines_to_alignment()
        elif options.line_update == "reset_to_previous":
            callbacks.create_reference_lines_for_previous_alignment()

    def _require_queries(self) -> Any:
        if self.queries is None:
            raise RuntimeError("DesktopAlignmentPresenter queries are not configured")
        return self.queries

    def _require_callbacks(self) -> DesktopAlignmentRenderCallbacks:
        if self.callbacks is None:
            raise RuntimeError("DesktopAlignmentPresenter callbacks are not configured")
        return self.callbacks
