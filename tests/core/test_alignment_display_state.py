"""Tests for Qt-free display state."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState


def test_region_annotation_source_toggle_and_reset() -> None:
    state = AlignmentDisplayState()

    assert state.region_annotation_source == "Allen"
    assert state.toggle_region_annotation_source() == "FranklinPaxinos"
    assert state.region_annotation_source == "FranklinPaxinos"

    state.reset_region_annotation_source()

    assert state.region_annotation_source == "Allen"


def test_unit_filter_set_and_reset() -> None:
    state = AlignmentDisplayState()

    assert state.unit_filter == "all"


def test_depth_view_settings_provide_probe_ranges_and_fit_depth_grid() -> None:
    state = AlignmentDisplayState()

    assert state.depth_view.plot_y_range_um == (-100.0, 3940.0)
    assert state.depth_view.view_range_um == (-2000.0, 6000.0)
    np.testing.assert_allclose(
        state.depth_view.fit_depth_um[:3],
        [-2000.0, -1980.0, -1960.0],
    )

    state.depth_view.set_probe_limits(25, 3900)

    assert state.depth_view.plot_y_range_um == (-75.0, 4000.0)


def test_edit_settings_track_current_alignment_command_options() -> None:
    state = AlignmentDisplayState()

    assert state.edit_settings.lin_fit
    assert state.edit_settings.extend_feature == 1
    assert state.edit_settings.set_lin_fit(False) is False

    state.reset_edit_settings()

    assert state.edit_settings.lin_fit


def test_visibility_toggles_are_resettable_display_state() -> None:
    state = AlignmentDisplayState()

    assert state.reference_lines_visible
    assert state.histology_boundaries_visible
    assert state.toggle_reference_lines_visible() is False
    assert state.toggle_histology_boundaries_visible() is False

    state.reset_visibility_toggles()

    assert state.reference_lines_visible
    assert state.histology_boundaries_visible
    assert state.set_unit_filter("KS good") == "KS good"
    assert state.unit_filter == "KS good"

    state.reset_unit_filter()

    assert state.unit_filter == "all"
