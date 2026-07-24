"""Tests for the per-shank state container."""

from __future__ import annotations

import datetime as _dt
from types import SimpleNamespace

import numpy as np
import pytest

from ephys_alignment_gui import alignment_state
from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_edit_history import AlignmentEditHistory
from ephys_alignment_gui.alignment_state import AlignmentState
from ephys_alignment_gui.shank_alignment import ShankAlignment
from ephys_alignment_gui.shank_runtime import ShankRuntime


class _FixedDatetime:
    """datetime stand-in whose now() returns a fixed instant."""

    _fixed = _dt.datetime(2026, 7, 9, 12, 0, 0)

    @classmethod
    def now(cls):
        return cls._fixed


def test_get_alignment_idx_original_and_out_of_range():
    sa = ShankAlignment(0)
    # Fresh shank has only "original".
    assert sa.prev_align == ["original"]
    assert sa.get_alignment_idx(0) == (None, None)
    assert sa.get_alignment_idx(5) == (None, None)


def test_edit_history_delegates_legacy_fit_attributes():
    sa = ShankAlignment(0, max_idx=3)

    assert isinstance(sa.alignment_state, AlignmentState)
    assert isinstance(sa.edit_history, AlignmentEditHistory)
    assert sa.max_idx == 3
    assert len(sa.features) == 4

    sa.idx = 2
    sa.current_idx = 5
    sa.total_idx = 6
    sa.last_idx = 4
    sa.diff_idx = 1
    sa.idx_prev = 1
    sa.features[2] = np.array([1.0])
    sa.track[2] = np.array([2.0])
    sa.lin_fit_history[2] = False

    assert sa.edit_history.idx == 2
    assert sa.edit_history.current_idx == 5
    assert sa.edit_history.total_idx == 6
    assert sa.edit_history.last_idx == 4
    assert sa.edit_history.diff_idx == 1
    assert sa.edit_history.idx_prev == 1
    np.testing.assert_array_equal(sa.edit_history.features[2], [1.0])
    np.testing.assert_array_equal(sa.edit_history.track[2], [2.0])
    assert not sa.edit_history.lin_fit_history[2]


def test_active_alignment_delegates_to_edit_history() -> None:
    sa = ShankAlignment(0)
    alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([0.0, 2.0]),
        lin_fit=False,
    )

    sa.active_alignment = alignment

    assert sa.active_alignment is not None
    np.testing.assert_array_equal(sa.features[0], [0.0, 1.0])
    np.testing.assert_array_equal(sa.track[0], [0.0, 2.0])
    assert not sa.lin_fit_history[0]
    np.testing.assert_array_equal(sa.active_alignment.feature, [0.0, 1.0])
    np.testing.assert_array_equal(sa.active_alignment.track, [0.0, 2.0])
    assert not sa.active_alignment.lin_fit

    sa.active_alignment = None
    assert sa.active_alignment is None


def test_add_alignment_and_roundtrip():
    sa = ShankAlignment(0)
    feature = np.array([0.0, 1.0, 2.0])
    track = np.array([0.0, 1.5, 3.0])
    key = sa.add_alignment(feature, track)
    # Newest first, "original" appended.
    assert sa.prev_align[0] == key
    assert sa.prev_align[-1] == "original"
    f, t = sa.get_alignment_idx(0)
    np.testing.assert_array_equal(f, feature)
    np.testing.assert_array_equal(t, track)


def test_add_alignment_same_second_disambiguates(monkeypatch):
    monkeypatch.setattr(alignment_state, "datetime", _FixedDatetime)
    sa = ShankAlignment(0)
    k1 = sa.add_alignment(np.array([0.0]), np.array([0.0]))
    k2 = sa.add_alignment(np.array([1.0]), np.array([1.0]))
    # Same-second saves must not collide.
    assert k1 != k2
    assert k2.startswith(k1)
    assert len(sa.alignments) == 2


def test_ordered_keys_newest_first():
    sa = ShankAlignment(0)
    sa.set_alignments(
        {
            "2026-07-09T10:00:00": [[0.0], [0.0]],
            "2026-07-09T12:00:00": [[1.0], [1.0]],
            "2026-07-09T11:00:00": [[2.0], [2.0]],
        }
    )
    assert sa.prev_align == [
        "2026-07-09T12:00:00",
        "2026-07-09T11:00:00",
        "2026-07-09T10:00:00",
        "original",
    ]


def test_cached_slice_hit_and_miss():
    sa = ShankAlignment(0)
    track = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    assert sa.cached_slice(track) is None  # nothing cached yet
    sa.set_slice({"ccf": "img"}, None, track)
    hit = sa.cached_slice(track)
    assert hit == ({"ccf": "img"}, None)
    # A different track (re-aligned) misses.
    other = track + 1.0
    assert sa.cached_slice(other) is None


def test_runtime_fields_project_to_attached_shank_runtime() -> None:
    collection = SimpleNamespace(
        shank_idx=0,
        local_coordinates=np.array([[5.0, 10.0], [6.0, 20.0]]),
        depths=np.array([10.0, 20.0]),
    )
    runtime = ShankRuntime(collection)
    sa = ShankAlignment(0)

    sa.attach_runtime(runtime)
    sa.ephysalign = "alignment-engine"
    sa.track_annotations_ras = np.array([[1.0, 2.0, 3.0]])
    sa.channel_locations_ras = np.array([[4.0, 5.0, 6.0]])

    assert runtime.ephysalign == "alignment-engine"
    np.testing.assert_array_equal(runtime.track_annotations_ras, [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(runtime.channel_locations_ras, [[4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(sa.chn_coords, [[5.0, 10.0], [6.0, 20.0]])
    np.testing.assert_array_equal(sa.chn_depths, [10.0, 20.0])


def test_attaching_mismatched_runtime_is_rejected() -> None:
    runtime = ShankRuntime(
        SimpleNamespace(
            shank_idx=1,
            local_coordinates=np.array([[0.0, 0.0]]),
            depths=np.array([0.0]),
        )
    )

    with pytest.raises(ValueError, match="Cannot attach runtime"):
        ShankAlignment(0).attach_runtime(runtime)


def test_independent_state_per_instance():
    a = ShankAlignment(0)
    b = ShankAlignment(1)
    a.add_alignment(np.array([0.0]), np.array([0.0]))
    assert len(a.alignments) == 1
    assert len(b.alignments) == 0  # no cross-contamination


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
