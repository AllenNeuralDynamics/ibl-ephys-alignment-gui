"""Tests for transient alignment edit history."""

from __future__ import annotations

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
