"""Tests for transient alignment edit history."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_edit_history import AlignmentEditHistory


def test_edit_history_initializes_cyclic_buffers() -> None:
    history = AlignmentEditHistory(max_idx=3)

    assert history.max_idx == 3
    assert history.idx == 0
    assert history.current_idx == 0
    assert history.total_idx == 0
    assert history.last_idx == 0
    assert history.diff_idx == 0
    assert history.idx_prev == 0
    assert history.track == [0, 0, 0, 0]
    assert history.features == [0, 0, 0, 0]
    assert history.lin_fit_history == [True, True, True, True]


def test_edit_history_is_transient_per_instance() -> None:
    first = AlignmentEditHistory(max_idx=2)
    second = AlignmentEditHistory(max_idx=2)

    first.features[0] = "feature"
    first.track[0] = "track"

    assert second.features == [0, 0, 0]
    assert second.track == [0, 0, 0]


def test_current_alignment_returns_none_before_initialization() -> None:
    history = AlignmentEditHistory(max_idx=2)

    assert history.current_alignment is None


def test_set_current_alignment_updates_legacy_buffers_with_copies() -> None:
    history = AlignmentEditHistory(max_idx=2)
    history.idx = 1
    alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([0.0, 2.0]),
        lin_fit=False,
    )

    history.set_current_alignment(alignment)

    np.testing.assert_array_equal(history.features[1], [0.0, 1.0])
    np.testing.assert_array_equal(history.track[1], [0.0, 2.0])
    assert not history.lin_fit_history[1]
    current = history.current_alignment
    assert current is not None
    np.testing.assert_array_equal(current.feature, [0.0, 1.0])
    np.testing.assert_array_equal(current.track, [0.0, 2.0])
    assert not current.lin_fit
    assert current.feature is not alignment.feature


def test_clear_current_alignment_restores_blank_sentinel() -> None:
    history = AlignmentEditHistory(max_idx=2)
    history.set_current_alignment(
        ActiveAlignment(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    )

    history.clear_current_alignment()

    assert history.features[0] == 0
    assert history.track[0] == 0
    assert history.lin_fit_history[0]
    assert history.current_alignment is None
