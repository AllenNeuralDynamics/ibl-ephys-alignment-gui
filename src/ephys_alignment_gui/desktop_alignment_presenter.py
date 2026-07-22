"""Desktop presenter for alignment edit rendering."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.alignment_events import (
    AlignmentChanged,
    AlignmentEditKind,
    LineUpdateMode,
)
from ephys_alignment_gui.alignment_read_models import ActiveAlignmentRenderState
from ephys_alignment_gui.event_bus import EventBus


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


@dataclass
class DesktopAlignmentPresenter:
    """Coordinate desktop alignment presentation from app read models."""

    events: EventBus

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
