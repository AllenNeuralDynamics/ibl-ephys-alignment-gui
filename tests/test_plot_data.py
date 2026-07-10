"""Tests for PlotData memoization and in-brain colour-level masking."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.plot_data import PlotData


def _minimal_plotdata() -> PlotData:
    """A PlotData over a trivial single-shank geometry, no spikes/clusters.

    Enough to exercise ``cached``/``filter_units``/the masking helpers without
    synthesizing full ALF spike + rms + psd payloads.
    """
    local_coordinates = np.column_stack(
        [np.zeros(4), np.array([0.0, 20.0, 40.0, 60.0])]
    )
    data = {
        "channels": {"localCoordinates": local_coordinates},
        "spikes": {
            "exists": False,
            "clusters": np.array([], dtype=int),
            "depths": np.array([]),
            "amps": np.array([]),
        },
        "clusters": {"exists": False},
    }
    return PlotData("dummy_probe_path", data, 0)


def test_cached_memoizes_per_method():
    pd = _minimal_plotdata()
    calls = []

    def _counting():
        calls.append(1)
        return object()

    pd._counting = _counting  # type: ignore[attr-defined]
    first = pd.cached("_counting")
    second = pd.cached("_counting")
    assert first is second
    assert len(calls) == 1


def test_cached_keys_on_args():
    pd = _minimal_plotdata()
    pd._echo = lambda x: (x,)  # type: ignore[attr-defined]
    a = pd.cached("_echo", ("AP",))
    b = pd.cached("_echo", ("LF",))
    assert a == ("AP",)
    assert b == ("LF",)
    assert pd.cached("_echo", ("AP",)) is a


def test_filter_units_idempotent_keeps_cache_warm():
    pd = _minimal_plotdata()
    pd._current_filter = "all"
    pd._img_cache[("marker", ())] = "warm"
    # Same subset -> no-op, cache preserved.
    pd.filter_units("all")
    assert pd._img_cache.get(("marker", ())) == "warm"


def test_filter_units_change_clears_cache():
    pd = _minimal_plotdata()
    pd._current_filter = "all"
    pd._img_cache[("marker", ())] = "warm"
    # Genuine change -> cache cleared before recompute.
    pd.filter_units("KS good")
    assert ("marker", ()) not in pd._img_cache
    assert pd._current_filter == "KS good"


# --- in-brain colour masking (Stage 0b) ---


def test_in_brain_mask_none_when_unset():
    pd = _minimal_plotdata()
    assert pd.in_brain_depths_um is None
    assert pd._in_brain_col_mask(np.array([0.0, 20.0, 40.0])) is None


def test_in_brain_mask_exact_path():
    pd = _minimal_plotdata()
    pd.in_brain_depths_um = np.array([20.0, 40.0])
    mask = pd._in_brain_col_mask(np.array([0.0, 20.0, 40.0, 60.0]))
    np.testing.assert_array_equal(mask, [False, True, True, False])


def test_in_brain_mask_binned_path():
    pd = _minimal_plotdata()
    pd.in_brain_depths_um = np.array([22.0, 41.0])  # near bins 1 and 1/2
    axis = np.array([0.0, 40.0, 80.0])  # bin centers, width 40
    mask = pd._in_brain_col_mask(axis, bin_width=40.0)
    # 22 -> round(22/40)=1; 41 -> round(41/40)=1  => only bin index 1
    np.testing.assert_array_equal(mask, [False, True, False])


def test_in_brain_mask_none_when_no_overlap():
    pd = _minimal_plotdata()
    pd.in_brain_depths_um = np.array([1000.0])  # off the axis entirely
    assert pd._in_brain_col_mask(np.array([0.0, 20.0, 40.0])) is None


def test_probe_levels_narrows_to_in_brain():
    pd = _minimal_plotdata()  # chn_coords depths = [0, 20, 40, 60]
    values = np.array([1.0, 2.0, 3.0, 100.0])  # depth-60 channel is an outlier
    full = pd._probe_levels(values)
    pd.in_brain_depths_um = np.array([0.0, 20.0, 40.0])  # exclude depth 60
    masked = pd._probe_levels(values)
    assert masked[1] < full[1]  # upper level no longer blown out by the outlier
