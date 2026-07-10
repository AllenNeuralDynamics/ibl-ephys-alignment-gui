"""Tests for the per-shank state container."""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from ephys_alignment_gui import shank_alignment
from ephys_alignment_gui.shank_alignment import ShankAlignment


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
    monkeypatch.setattr(shank_alignment, "datetime", _FixedDatetime)
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


def test_independent_state_per_instance():
    a = ShankAlignment(0)
    b = ShankAlignment(1)
    a.add_alignment(np.array([0.0]), np.array([0.0]))
    assert len(a.alignments) == 1
    assert len(b.alignments) == 0  # no cross-contamination


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
