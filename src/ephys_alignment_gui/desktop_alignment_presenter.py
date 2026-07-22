"""Desktop presenter for alignment edit rendering."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.alignment_events import (
    AlignmentChanged,
    AlignmentEdited,
    AlignmentEditKind,
    LineUpdateMode,
)
from ephys_alignment_gui.alignment_read_models import ActiveAlignmentRenderState
from ephys_alignment_gui.event_bus import EventBus, EventSubscription

logger = logging.getLogger(__name__)


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
    apply_histology_data: Callable[[Any], None]
    apply_channel_projection: Callable[[Any], None]
    reattach_reference_lines: Callable[[], None]
    plot_histology: Callable[[], None]
    plot_scale_factor: Callable[[], None]
    plot_fit: Callable[[], None]
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
        return [
            self.events.subscribe(AlignmentEdited, self.on_alignment_edited),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_apply_data,
            ),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_prepare_lines,
            ),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_histology,
            ),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_scale,
            ),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_fit,
            ),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_channels,
            ),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_perpendicular,
            ),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_lines,
            ),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_range,
            ),
            self.events.subscribe(
                AlignmentChanged,
                self.on_alignment_changed_status,
            ),
        ]

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
            self.emit_legacy_alignment_changed(
                render_state=render_state,
                source=event.edit_kind,
                line_update=options.line_update,
                reset_histology_range=options.reset_histology_range,
                refresh_perpendicular=options.refresh_perpendicular,
            )
        finally:
            if depth_ranges:
                callbacks.restore_depth_plot_y_ranges(depth_ranges)

    def on_alignment_changed_apply_data(self, event: AlignmentChanged) -> None:
        """Apply derived alignment data to desktop compatibility state."""
        callbacks = self._require_callbacks()
        callbacks.apply_histology_data(event.histology)
        callbacks.apply_channel_projection(event.projection)

    def on_alignment_changed_prepare_lines(self, event: AlignmentChanged) -> None:
        """Prepare reference-line handles before plotting updates."""
        if event.line_update == "reattach":
            self._require_callbacks().reattach_reference_lines()

    def on_alignment_changed_histology(self, event: AlignmentChanged) -> None:
        """Render histology region overlays."""
        self._require_callbacks().plot_histology()

    def on_alignment_changed_scale(self, event: AlignmentChanged) -> None:
        """Render scale-factor plot."""
        self._require_callbacks().plot_scale_factor()

    def on_alignment_changed_fit(self, event: AlignmentChanged) -> None:
        """Render fit plot."""
        self._require_callbacks().plot_fit()

    def on_alignment_changed_channels(self, event: AlignmentChanged) -> None:
        """Render channel projection overlays."""
        self._require_callbacks().plot_channels(event.projection)

    def on_alignment_changed_perpendicular(self, event: AlignmentChanged) -> None:
        """Refresh perpendicular histology when required."""
        if event.refresh_perpendicular:
            self._require_callbacks().refresh_perpendicular_histology()

    def on_alignment_changed_lines(self, event: AlignmentChanged) -> None:
        """Update desktop reference-line handles after alignment changes."""
        callbacks = self._require_callbacks()
        if event.line_update == "reattach":
            callbacks.reattach_reference_lines()
        elif event.line_update == "sync_to_alignment":
            callbacks.reattach_reference_lines()
            callbacks.update_reference_lines_to_alignment()
        elif event.line_update == "reset_to_previous":
            callbacks.create_reference_lines_for_previous_alignment()

    def on_alignment_changed_range(self, event: AlignmentChanged) -> None:
        """Apply desktop plot range policy."""
        if event.reset_histology_range:
            self._require_callbacks().set_default_feature_y_range()

    def on_alignment_changed_status(self, event: AlignmentChanged) -> None:
        """Refresh desktop status text when required."""
        if event.update_status:
            self._require_callbacks().update_status()

    def emit_legacy_alignment_changed(
        self,
        *,
        render_state: ActiveAlignmentRenderState,
        source: str,
        line_update: LineUpdateMode = "none",
        reset_histology_range: bool = False,
        refresh_perpendicular: bool = True,
    ) -> None:
        """Publish the legacy desktop ``AlignmentChanged`` render packet."""
        self.events.emit(
            self.build_legacy_alignment_changed(
                render_state=render_state,
                source=source,
                line_update=line_update,
                reset_histology_range=reset_histology_range,
                refresh_perpendicular=refresh_perpendicular,
            )
        )

    def build_legacy_alignment_changed(
        self,
        *,
        render_state: ActiveAlignmentRenderState,
        source: str,
        line_update: LineUpdateMode = "none",
        reset_histology_range: bool = False,
        refresh_perpendicular: bool = True,
    ) -> AlignmentChanged:
        """Return the desktop compatibility refresh payload."""
        return AlignmentChanged(
            source=source,
            active_alignment=render_state.active_alignment,
            histology=render_state.histology,
            projection=render_state.projection,
            line_update=line_update,
            reset_histology_range=reset_histology_range,
            refresh_perpendicular=refresh_perpendicular,
        )

    def _require_queries(self) -> Any:
        if self.queries is None:
            raise RuntimeError("DesktopAlignmentPresenter queries are not configured")
        return self.queries

    def _require_callbacks(self) -> DesktopAlignmentRenderCallbacks:
        if self.callbacks is None:
            raise RuntimeError("DesktopAlignmentPresenter callbacks are not configured")
        return self.callbacks
