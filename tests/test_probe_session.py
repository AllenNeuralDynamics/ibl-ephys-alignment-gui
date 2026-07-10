"""Tests for ProbeSession per-shank state delegation."""

from __future__ import annotations

import numpy as np
import pytest

from ephys_alignment_gui.probe_session import ProbeSession


def test_init_shanks_and_bounds():
    s = ProbeSession()
    s.init_shanks(4)
    assert s.n_shanks == 4
    assert s.current_shank_idx == 0
    assert s.has_shank(3)
    assert not s.has_shank(4)
    with pytest.raises(KeyError):
        s.current_shank_idx = 4


def test_delegated_attr_roundtrip():
    s = ProbeSession()
    s.init_shanks(2)
    s.track_annotations_ras = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(
        s.active_shank.track_annotations_ras, [[1.0, 2.0, 3.0]]
    )
    # Read back through the descriptor.
    np.testing.assert_array_equal(s.track_annotations_ras, [[1.0, 2.0, 3.0]])


def test_shanks_keep_independent_state():
    s = ProbeSession()
    s.init_shanks(2)
    s.idx = 3
    s.features[0] = np.array([1.0])
    s.current_shank_idx = 1
    # Shank 1 starts fresh.
    assert s.idx == 0
    s.idx = 9
    # Switching back restores shank 0's state — no cross-contamination.
    s.current_shank_idx = 0
    assert s.idx == 3
    np.testing.assert_array_equal(s.features[0], [1.0])


def test_lines_features_is_four_wide():
    # Column 3 is the perpendicular-slice line handle.
    s = ProbeSession()
    assert s.lines_features.shape == (0, 4)


def test_shank_instances_created_lazily():
    s = ProbeSession()
    s.init_shanks(4)
    assert s.shanks == {}  # none built until accessed
    _ = s.active_shank
    assert set(s.shanks) == {0}
    s.current_shank_idx = 2
    _ = s.ephysalign
    assert set(s.shanks) == {0, 2}
