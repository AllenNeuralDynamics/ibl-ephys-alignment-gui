"""Tests for Qt-free alignment edit commands."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_edit_history import AlignmentEditHistory
from ephys_alignment_gui.alignment_edit_service import AlignmentEditService


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
