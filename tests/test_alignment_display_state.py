"""Tests for Qt-free display state."""

from __future__ import annotations

from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState


def test_region_annotation_source_toggle_and_reset() -> None:
    state = AlignmentDisplayState()

    assert state.region_annotation_source == "Allen"
    assert state.toggle_region_annotation_source() == "FranklinPaxinos"
    assert state.region_annotation_source == "FranklinPaxinos"

    state.reset_region_annotation_source()

    assert state.region_annotation_source == "Allen"
