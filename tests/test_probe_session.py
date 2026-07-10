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


class _FakePlotData:
    """Minimal cached() stub to exercise _LazyPlotAttr."""

    def __init__(self):
        self.calls: dict = {}
        self._memo: dict = {}

    def _compute(self, method):
        return {
            "get_fr_img": "fr",
            "get_fr_p2t_data_scatter": (10, 20, 30),
            "get_rms_data_img_probe": ("img", "probe"),
        }.get(method, "x")

    def cached(self, method, args=()):
        key = (method, args)
        if key not in self._memo:
            self.calls[key] = self.calls.get(key, 0) + 1
            self._memo[key] = self._compute(method)
        return self._memo[key]


def test_lazy_plot_attr_none_before_load():
    s = ProbeSession()
    s.init_shanks(1)
    assert s.plotdata is None
    assert s.img_fr_data is None


def test_lazy_plot_attr_computes_and_indexes():
    s = ProbeSession()
    s.init_shanks(1)
    s.plotdata = _FakePlotData()
    assert s.img_fr_data == "fr"
    assert s.scat_p2t_data == 20  # index 1 of the 3-tuple
    assert s.img_rms_APdata == "img"  # ("AP",) index 0
    assert s.probe_rms_APdata == "probe"  # ("AP",) index 1
    # memoized inside cached(): scat_fr/p2t/amp share one call
    _ = s.scat_fr_data, s.scat_amp_data
    assert s.plotdata.calls[("get_fr_p2t_data_scatter", ())] == 1


def test_lazy_plot_attr_is_read_only():
    s = ProbeSession()
    s.init_shanks(1)
    with pytest.raises(AttributeError):
        s.img_fr_data = "nope"
