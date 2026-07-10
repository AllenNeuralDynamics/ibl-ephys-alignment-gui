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
