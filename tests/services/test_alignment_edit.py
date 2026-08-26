"""Tests for Qt-free alignment edit commands."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_edit_history import AlignmentEditHistory
from ephys_alignment_gui.services.alignment_edit import AlignmentEditService


class FakeEphysAlignment:
    def __init__(self) -> None:
        self.feature_init = np.array([1.0, 4.0])
        self.track_init = np.array([3.0, 6.0])

    @staticmethod
    def feature2track(depths_track, feature_ref, track_ref):
        return np.asarray(depths_track, dtype=float) + 10.0

    @staticmethod
    def adjust_extremes_uniform(feature, track):
        return np.asarray(track, dtype=float) + 1.0

    @staticmethod
    def adjust_extremes_linear(feature, track, extend_feature=1):
        return (
            np.asarray(feature, dtype=float) + extend_feature,
            np.asarray(track, dtype=float) + extend_feature,
        )


class PiecewiseFakeEphysAlignment:
    def __init__(self) -> None:
        self.feature_init = np.array([0.0, 4.0])
        self.track_init = np.array([0.0, 4.0])

    @staticmethod
    def feature2track(feature_new, feature_ref, track_ref):
        return np.interp(feature_new, feature_ref, track_ref)

    @staticmethod
    def adjust_extremes_uniform(feature, track):
        track = np.array(track, dtype=float, copy=True)
        diff = np.diff(feature - track)
        track[0] -= diff[0]
        track[-1] += diff[-1]
        return track

    @staticmethod
    def adjust_extremes_linear(feature, track, extend_feature=1):
        return feature, track


def test_go_next_noops_when_cursor_is_at_latest_edit() -> None:
    history = AlignmentEditHistory(max_idx=10)

    result = AlignmentEditService().go_next(history)

    assert not result.changed
    assert history.current_idx == 0
    assert history.idx == 0


def test_go_next_advances_to_next_saved_edit() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.idx = 0
    history.set_current_alignment(
        ActiveAlignment(np.array([0.0, 1.0]), np.array([0.0, 2.0]), lin_fit=True)
    )
    history.idx = 1
    history.set_current_alignment(
        ActiveAlignment(np.array([2.0, 3.0]), np.array([4.0, 5.0]), lin_fit=False)
    )
    history.idx = 0
    history.current_idx = 0
    history.total_idx = 1

    result = AlignmentEditService().go_next(history)

    assert result.changed
    assert history.current_idx == 1
    assert history.idx == 1
    assert result.alignment is not None
    np.testing.assert_array_equal(result.alignment.feature, [2.0, 3.0])
    np.testing.assert_array_equal(result.alignment.track, [4.0, 5.0])
    assert result.lin_fit is False


def test_go_previous_tracks_latest_edit_and_moves_back() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.idx = 1
    history.set_current_alignment(
        ActiveAlignment(np.array([0.0, 1.0]), np.array([2.0, 3.0]), lin_fit=False)
    )
    history.idx = 2
    history.set_current_alignment(
        ActiveAlignment(np.array([4.0, 5.0]), np.array([6.0, 7.0]), lin_fit=True)
    )
    history.current_idx = 2
    history.total_idx = 2
    history.diff_idx = 9

    result = AlignmentEditService().go_previous(history)

    assert result.changed
    assert history.last_idx == 2
    assert history.current_idx == 1
    assert history.idx == 1
    assert result.alignment is not None
    np.testing.assert_array_equal(result.alignment.feature, [0.0, 1.0])
    np.testing.assert_array_equal(result.alignment.track, [2.0, 3.0])
    assert result.lin_fit is False


def test_go_previous_noops_at_earliest_available_edit() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.current_idx = 0
    history.total_idx = 2
    history.diff_idx = 9

    result = AlignmentEditService().go_previous(history)

    assert not result.changed
    assert history.current_idx == 0
    assert history.idx == 0


def test_reset_to_initial_appends_initial_alignment() -> None:
    history = AlignmentEditHistory(max_idx=10)

    result = AlignmentEditService().reset_to_initial(
        history,
        feature_init=np.array([1.0, 2.0]),
        track_init=np.array([3.0, 4.0]),
        lin_fit=False,
    )

    assert result.changed
    assert history.diff_idx == 9
    assert history.total_idx == 1
    assert history.current_idx == 1
    assert history.idx == 1
    assert result.alignment is not None
    np.testing.assert_array_equal(result.alignment.feature, [1.0, 2.0])
    np.testing.assert_array_equal(result.alignment.track, [3.0, 4.0])
    assert result.lin_fit is False


def test_reset_to_initial_discards_redo_span_after_undo() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.current_idx = 2
    history.total_idx = 5
    history.last_idx = 5

    AlignmentEditService().reset_to_initial(
        history,
        feature_init=np.array([1.0, 2.0]),
        track_init=np.array([3.0, 4.0]),
        lin_fit=True,
    )

    assert history.diff_idx == 7
    assert history.total_idx == 3
    assert history.current_idx == 3
    assert history.idx == 3


def test_fit_to_reference_lines_appends_fit_edit() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.set_current_alignment(
        ActiveAlignment(
            np.array([0.0, 4.0]),
            np.array([10.0, 14.0]),
            lin_fit=True,
        )
    )

    result = AlignmentEditService().fit_to_reference_lines(
        history,
        ephysalign=FakeEphysAlignment(),
        line_features_um=np.array([2_000_000.0]),
        line_tracks_um=np.array([12_000_000.0]),
        lin_fit=False,
        extend_feature=2,
    )

    assert result.changed
    assert history.idx_prev == 0
    assert history.idx == 1
    assert history.current_idx == 1
    assert history.total_idx == 1
    assert result.alignment is not None
    np.testing.assert_array_equal(result.alignment.feature, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(result.alignment.track, [11.0, 23.0, 15.0])
    assert result.lin_fit is False


def test_fit_to_reference_lines_uses_linear_extremes_when_enabled() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.set_current_alignment(
        ActiveAlignment(
            np.array([0.0, 4.0]),
            np.array([10.0, 14.0]),
        )
    )

    result = AlignmentEditService().fit_to_reference_lines(
        history,
        ephysalign=FakeEphysAlignment(),
        line_features_um=np.array([1_000_000.0, 3_000_000.0]),
        line_tracks_um=np.array([11_000_000.0, 13_000_000.0]),
        lin_fit=True,
        extend_feature=2,
    )

    assert result.alignment is not None
    np.testing.assert_array_equal(result.alignment.feature, [2.0, 3.0, 5.0, 6.0])
    np.testing.assert_array_equal(
        result.alignment.track,
        [12.0, 23.0, 25.0, 16.0],
    )
    assert result.lin_fit is True


def test_fit_to_reference_lines_requires_two_points_for_linear_extremes() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.set_current_alignment(
        ActiveAlignment(
            np.array([0.0, 4.0]),
            np.array([10.0, 14.0]),
        )
    )

    result = AlignmentEditService().fit_to_reference_lines(
        history,
        ephysalign=FakeEphysAlignment(),
        line_features_um=np.array([2_000_000.0]),
        line_tracks_um=np.array([12_000_000.0]),
        lin_fit=True,
        extend_feature=2,
    )

    assert result.alignment is not None
    np.testing.assert_array_equal(result.alignment.feature, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(result.alignment.track, [11.0, 23.0, 15.0])
    assert result.lin_fit is True


def test_fit_to_reference_lines_is_idempotent_for_same_display_pairs() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.set_current_alignment(
        ActiveAlignment(
            np.array([0.0, 1.0, 4.0]),
            np.array([0.0, 2.0, 4.0]),
        )
    )
    service = AlignmentEditService()
    ephysalign = PiecewiseFakeEphysAlignment()

    first = service.fit_to_reference_lines(
        history,
        ephysalign=ephysalign,
        line_features_um=np.array([3_000_000.0]),
        line_tracks_um=np.array([3_000_000.0]),
        lin_fit=False,
        extend_feature=2,
    )
    assert first.alignment is not None
    first_feature = np.array(first.alignment.feature, copy=True)
    first_track = np.array(first.alignment.track, copy=True)

    second = service.fit_to_reference_lines(
        history,
        ephysalign=ephysalign,
        line_features_um=np.array([3_000_000.0]),
        line_tracks_um=np.array([3_000_000.0]),
        lin_fit=False,
        extend_feature=2,
    )

    assert second.alignment is not None
    np.testing.assert_allclose(second.alignment.feature, first_feature)
    np.testing.assert_allclose(second.alignment.track, first_track)


def test_fit_to_reference_lines_without_points_resets_to_initial_alignment() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.set_current_alignment(
        ActiveAlignment(
            np.array([0.0, 4.0]),
            np.array([10.0, 14.0]),
            lin_fit=True,
        )
    )

    result = AlignmentEditService().fit_to_reference_lines(
        history,
        ephysalign=FakeEphysAlignment(),
        line_features_um=np.array([]),
        line_tracks_um=np.array([]),
        lin_fit=False,
        extend_feature=2,
    )

    assert result.changed
    assert history.idx == 1
    assert result.alignment is not None
    np.testing.assert_array_equal(result.alignment.feature, [1.0, 4.0])
    np.testing.assert_array_equal(result.alignment.track, [3.0, 6.0])
    assert result.lin_fit is False


def test_offset_from_tip_appends_offset_edit_without_mutating_previous_slot() -> None:
    history = AlignmentEditHistory(max_idx=10)
    history.set_current_alignment(
        ActiveAlignment(
            np.array([0.0, 2.0, 4.0]),
            np.array([10.0, 12.0, 14.0]),
            lin_fit=True,
        )
    )

    result = AlignmentEditService().offset_from_tip(
        history,
        tip_position_um=200.0,
        probe_tip_um=100.0,
        lin_fit=False,
        track_shift_m=0.5,
    )

    assert result.changed
    assert history.idx_prev == 0
    assert history.idx == 1
    assert result.alignment is not None
    np.testing.assert_array_equal(history.track[0], [10.0, 12.0, 14.0])
    np.testing.assert_array_equal(result.alignment.feature, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(result.alignment.track, [10.5001, 12.5, 14.5001])
    assert result.lin_fit is False
