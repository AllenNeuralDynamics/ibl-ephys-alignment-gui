"""Tests for DesktopViewSession per-shank state delegation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.desktop_view_session import DesktopViewSession
from ephys_alignment_gui.shank_runtime import ShankRuntime


def test_init_shanks_and_bounds():
    s = DesktopViewSession()
    s.init_shanks(4)
    assert s.n_shanks == 4
    assert s.current_shank_idx == 0
    assert s.has_shank(3)
    assert not s.has_shank(4)
    with pytest.raises(KeyError):
        s.current_shank_idx = 4


def test_delegated_attr_roundtrip():
    s = DesktopViewSession()
    s.init_shanks(2)
    s.track_annotations_ras = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(
        s.active_shank.track_annotations_ras, [[1.0, 2.0, 3.0]]
    )
    # Read back through the descriptor.
    np.testing.assert_array_equal(s.track_annotations_ras, [[1.0, 2.0, 3.0]])


def test_shanks_keep_independent_state():
    s = DesktopViewSession()
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


def test_shank_instances_created_lazily():
    s = DesktopViewSession()
    s.init_shanks(4)
    assert s.shanks == {}  # none built until accessed
    _ = s.active_shank
    assert set(s.shanks) == {0}
    s.current_shank_idx = 2
    _ = s.ephysalign
    assert set(s.shanks) == {0, 2}


def test_alignment_history_isolated_per_shank():
    s = DesktopViewSession()
    s.init_shanks(2)
    s.active_shank.add_alignment(np.array([0.0]), np.array([0.0]))
    assert len(s.active_shank.alignments) == 1
    s.current_shank_idx = 1
    # Shank 1 has its own (empty) history — no cross-contamination.
    assert s.active_shank.alignments == {}
    assert s.active_shank.prev_align == ["original"]


def test_active_alignment_delegates_to_active_shank():
    s = DesktopViewSession()
    s.init_shanks(2)
    s.active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([0.0, 2.0]),
        lin_fit=False,
    )

    first = s.active_alignment
    assert first is not None
    np.testing.assert_array_equal(first.feature, [0.0, 1.0])
    assert not first.lin_fit

    s.current_shank_idx = 1
    assert s.active_alignment is None
    s.active_alignment = ActiveAlignment(np.array([3.0, 4.0]), np.array([5.0, 6.0]))

    s.current_shank_idx = 0
    np.testing.assert_array_equal(s.active_alignment.feature, [0.0, 1.0])
    s.current_shank_idx = 1
    np.testing.assert_array_equal(s.active_alignment.feature, [3.0, 4.0])


def test_detach_preserves_runtime_state_but_teardown_clears_active_runtime():
    runtime = ShankRuntime(
        SimpleNamespace(
            shank_idx=0,
            local_coordinates=np.array([[0.0, 0.0]]),
            depths=np.array([0.0]),
        )
    )
    runtime.ephysalign = "alignment-engine"
    runtime.plotdata = "plot-data"
    s = DesktopViewSession()
    s.init_shanks(1)
    s.active_shank.attach_runtime(runtime)

    s.detach({})

    assert runtime.ephysalign == "alignment-engine"
    assert runtime.plotdata == "plot-data"

    s.teardown({})

    assert runtime.ephysalign is None
    assert runtime.plotdata is None
