"""Tests for pure editable alignment state."""

from __future__ import annotations

import datetime as _dt

import numpy as np

import ephys_alignment_gui.core.alignment_state as alignment_state
from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_edit_history import AlignmentEditHistory
from ephys_alignment_gui.core.alignment_state import (
    AlignmentState,
    PendingReferenceLines,
)


class _FixedDatetime:
    """datetime stand-in whose now() returns a fixed instant."""

    _fixed = _dt.datetime(2026, 7, 9, 12, 0, 0)

    @classmethod
    def now(cls):
        return cls._fixed


def test_alignment_state_owns_edit_history_and_active_alignment() -> None:
    state = AlignmentState(max_idx=3)

    assert isinstance(state.edit_history, AlignmentEditHistory)
    assert state.edit_history.max_idx == 3
    assert state.prev_align == ["original"]

    state.active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
        lin_fit=False,
    )

    assert state.active_alignment is not None
    np.testing.assert_array_equal(state.active_alignment.feature, [0.0, 1.0])
    assert not state.active_alignment.lin_fit
    assert not state.has_unsaved_alignment

    state.mark_alignment_changed()

    assert state.has_unsaved_alignment
    assert state.save_state.revision == 1
    assert state.save_state.saved_revision == 0

    state.mark_saved()

    assert not state.has_unsaved_alignment
    assert state.save_state.saved_revision == 1

    state.mark_alignment_changed()

    assert not state.has_unsaved_alignment
    assert state.save_state.revision == 2
    assert state.save_state.saved_revision == 1

    state.active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 4.0]),
        lin_fit=False,
    )
    state.mark_alignment_changed()

    assert state.has_unsaved_alignment

    state.active_alignment = None

    assert state.active_alignment is None


def test_alignment_state_history_roundtrip() -> None:
    state = AlignmentState()
    feature = np.array([0.0, 1.0, 2.0])
    track = np.array([0.0, 1.5, 3.0])

    key = state.add_alignment(feature, track)

    assert state.prev_align[0] == key
    assert state.prev_align[-1] == "original"
    f, t = state.get_alignment_idx(0)
    np.testing.assert_array_equal(f, feature)
    np.testing.assert_array_equal(t, track)


def test_alignment_state_can_prepare_history_without_mutating() -> None:
    state = AlignmentState()
    feature = np.array([0.0, 1.0, 2.0])
    track = np.array([0.0, 1.5, 3.0])

    key, alignments = state.with_alignment_added(feature, track)

    assert state.alignments == {}
    assert state.prev_align == ["original"]
    assert alignments[key] == [feature.tolist(), track.tolist()]


def test_alignment_state_save_history_represents_active_alignment_once() -> None:
    state = AlignmentState()
    state.set_alignments({"saved": [[0.0, 1.0], [2.0, 3.0]]})
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )

    alignments = state.alignment_history_for_save()

    assert alignments == state.alignments


def test_alignment_state_save_history_adds_new_active_alignment_without_mutating(
    monkeypatch,
) -> None:
    monkeypatch.setattr(alignment_state, "datetime", _FixedDatetime)
    state = AlignmentState()
    state.set_alignments({"saved": [[0.0, 1.0], [2.0, 3.0]]})
    state.active_alignment = ActiveAlignment(
        np.array([4.0, 5.0]),
        np.array([6.0, 7.0]),
    )

    alignments = state.alignment_history_for_save()

    assert state.alignments == {"saved": [[0.0, 1.0], [2.0, 3.0]]}
    assert alignments == {
        "saved": [[0.0, 1.0], [2.0, 3.0]],
        "2026-07-09T12:00:00": [[4.0, 5.0], [6.0, 7.0]],
    }


def test_alignment_state_same_second_keys_disambiguate(monkeypatch) -> None:
    monkeypatch.setattr(alignment_state, "datetime", _FixedDatetime)
    state = AlignmentState()

    k1 = state.add_alignment(np.array([0.0]), np.array([0.0]))
    k2 = state.add_alignment(np.array([1.0]), np.array([1.0]))

    assert k1 != k2
    assert k2.startswith(k1)
    assert len(state.alignments) == 2


def test_alignment_state_set_alignments_orders_newest_first() -> None:
    state = AlignmentState()

    state.set_alignments(
        {
            "2026-07-09T10:00:00": [[0.0], [0.0]],
            "2026-07-09T12:00:00": [[1.0], [1.0]],
            "2026-07-09T11:00:00": [[2.0], [2.0]],
        }
    )

    assert state.prev_align == [
        "2026-07-09T12:00:00",
        "2026-07-09T11:00:00",
        "2026-07-09T10:00:00",
        "original",
    ]


def test_alignment_state_original_and_out_of_range_are_empty() -> None:
    state = AlignmentState()

    assert state.get_alignment_idx(0) == (None, None)
    assert state.get_alignment_idx(5) == (None, None)


def test_alignment_state_filters_legacy_auto_from_persisted_alignments() -> None:
    state = AlignmentState()
    state.set_alignments(
        {
            "auto": [[100.0], [200.0]],
            "2026-07-09T10:00:00": [[0.0], [1.0]],
        }
    )

    assert "auto" not in state.alignments
    assert state.prev_align == ["2026-07-09T10:00:00", "original"]


def test_alignment_state_pending_reference_lines_roundtrip() -> None:
    state = AlignmentState()
    lines = PendingReferenceLines.from_values(
        np.array([100.0, 200.0]),
        np.array([110.0, 210.0]),
    )
    assert lines is not None

    state.set_pending_reference_lines(lines)

    assert state.prev_align == ["original"]
    assert state.pending_reference_lines is lines
    assert not state.has_unsaved_alignment
    np.testing.assert_array_equal(lines.feature_positions_um, [100.0, 200.0])
    np.testing.assert_array_equal(lines.warped_positions_um, [110.0, 210.0])
    np.testing.assert_array_equal(lines.track_positions_um, [110.0, 210.0])

    state.clear_pending_reference_lines()

    assert state.pending_reference_lines is None


def test_alignment_state_select_alignment_rebases_working_history() -> None:
    state = AlignmentState()
    state.active_alignment = ActiveAlignment(
        np.array([9.0, 10.0]),
        np.array([11.0, 12.0]),
    )
    state.set_pending_reference_lines(
        PendingReferenceLines(np.array([1.0]), np.array([2.0]))
    )
    state.set_alignments({"saved": [[0.0, 1.0], [2.0, 3.0]]})

    feature, track = state.select_alignment_idx(0)

    np.testing.assert_array_equal(feature, [0.0, 1.0])
    np.testing.assert_array_equal(track, [2.0, 3.0])
    assert state.active_alignment is not None
    np.testing.assert_array_equal(state.active_alignment.feature, [0.0, 1.0])
    np.testing.assert_array_equal(state.active_alignment.track, [2.0, 3.0])
    assert state.edit_history.current_idx == 0
    assert state.pending_reference_lines is None


def test_alignment_state_can_clear_previous_alignment_selection() -> None:
    state = AlignmentState()
    state.feature_prev = np.array([0.0, 1.0])
    state.track_prev = np.array([2.0, 3.0])
    state.set_pending_reference_lines(
        PendingReferenceLines(np.array([1.0]), np.array([2.0]))
    )
    state.set_alignments({"saved": [[0.0, 1.0], [2.0, 3.0]]})

    state.clear_previous_alignment_selection()

    assert state.feature_prev is None
    assert state.track_prev is None
    assert state.pending_reference_lines is None
    assert state.prev_align == ["saved", "original"]


def test_alignment_state_import_merges_when_active_state_is_dirty() -> None:
    state = AlignmentState()
    state.set_alignments({"saved": [[0.0], [1.0]]})
    state.active_alignment = ActiveAlignment(np.array([2.0]), np.array([3.0]))
    state.mark_alignment_changed()

    state.import_alignments(
        {
            "saved": [[4.0], [5.0]],
            "loaded": [[6.0], [7.0]],
        }
    )

    assert state.alignments == {
        "saved": [[0.0], [1.0]],
        "saved.imported.1": [[4.0], [5.0]],
        "loaded": [[6.0], [7.0]],
    }
    assert state.has_unsaved_alignment
