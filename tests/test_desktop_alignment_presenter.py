"""Tests for desktop alignment presentation."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_events import AlignmentChanged
from ephys_alignment_gui.alignment_read_models import ActiveAlignmentRenderState
from ephys_alignment_gui.desktop_alignment_presenter import (
    DesktopAlignmentPresenter,
    desktop_presentation_options_for_edit,
)
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.event_bus import EventBus


def test_desktop_presenter_builds_and_emits_legacy_render_payload() -> None:
    events = EventBus()
    presenter = DesktopAlignmentPresenter(events)
    received: list[AlignmentChanged] = []
    events.subscribe(AlignmentChanged, received.append)
    active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    render_state = ActiveAlignmentRenderState(
        key=AlignmentKey("rec", "stream", 1),
        active_alignment=active_alignment,
        histology="histology",
        projection="projection",
    )

    presenter.emit_legacy_alignment_changed(
        render_state=render_state,
        source="fit",
        line_update="sync_to_alignment",
        reset_histology_range=True,
        refresh_perpendicular=False,
    )

    assert len(received) == 1
    event = received[0]
    assert event.source == "fit"
    assert event.active_alignment is active_alignment
    assert event.histology == "histology"
    assert event.projection == "projection"
    assert event.line_update == "sync_to_alignment"
    assert event.reset_histology_range
    assert not event.refresh_perpendicular


def test_desktop_presentation_options_are_derived_from_edit_kind() -> None:
    fit = desktop_presentation_options_for_edit("fit")
    reset = desktop_presentation_options_for_edit("reset")

    assert fit.line_update == "sync_to_alignment"
    assert fit.preserve_depth_range
    assert not fit.clear_reference_lines
    assert reset.line_update == "reset_to_previous"
    assert reset.reset_histology_range
    assert reset.clear_reference_lines
