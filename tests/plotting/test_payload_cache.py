"""Tests for ephys plot payload cache memoization and unit filtering."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.plotting.payload_cache import EphysPlotPayloadCache


def _minimal_payload_cache() -> EphysPlotPayloadCache:
    """A payload cache over a trivial single-shank geometry, no spikes/clusters.

    Enough to exercise payload memoization/filtering without
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
    return EphysPlotPayloadCache("dummy_probe_path", data, 0)


def test_get_or_build_payload_memoizes_per_key():
    pd = _minimal_payload_cache()
    calls = []

    def _counting():
        calls.append(1)
        return object()

    first = pd.get_or_build_payload(("counting",), _counting)
    second = pd.get_or_build_payload(("counting",), _counting)
    assert first is second
    assert len(calls) == 1


def test_get_or_build_payload_keys_on_arguments():
    pd = _minimal_payload_cache()
    a = pd.get_or_build_payload(("echo", "AP"), lambda: ("AP",))
    b = pd.get_or_build_payload(("echo", "LF"), lambda: ("LF",))
    assert a == ("AP",)
    assert b == ("LF",)
    assert pd.get_or_build_payload(("echo", "AP"), lambda: ("AP again",)) is a


def test_filter_units_idempotent_keeps_cache_warm():
    pd = _minimal_payload_cache()
    pd._current_filter = "all"
    pd._img_cache[("marker",)] = "warm"
    # Same subset -> no-op, cache preserved.
    pd.filter_units("all")
    assert pd._img_cache.get(("marker",)) == "warm"


def test_filter_units_change_clears_cache():
    pd = _minimal_payload_cache()
    pd._current_filter = "all"
    pd._img_cache[("marker",)] = "warm"
    # Genuine change -> cache cleared before recompute.
    pd.filter_units("KS good")
    assert ("marker",) not in pd._img_cache
    assert pd._current_filter == "KS good"
