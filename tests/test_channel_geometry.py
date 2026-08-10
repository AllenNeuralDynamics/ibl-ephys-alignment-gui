"""Tests for explicit ephys channel geometry handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.channel_geometry import (
    n_shanks_from_geometry,
    rows_for_shank,
)


def test_rows_for_shank_uses_explicit_shank_ind():
    coords = np.array(
        [
            [0.0, 0.0],
            [250.0, 0.0],
            [0.0, 20.0],
            [250.0, 20.0],
        ]
    )
    shank_ind = np.array([0, 1, 0, 1])

    assert n_shanks_from_geometry(coords, shank_ind) == 2
    np.testing.assert_array_equal(
        rows_for_shank(coords, shank_ind, shank_idx=1, n_shanks=2),
        np.array([1, 3]),
    )


def test_plot_data_uses_channel_table_rows_not_raw_ind():
    pytest.importorskip("PyQt5")
    from ephys_alignment_gui.plotting.payload_cache import EphysPlotPayloadCache

    data = {
        "channels": {
            "localCoordinates": np.array(
                [
                    [0.0, 0.0],
                    [250.0, 0.0],
                    [0.0, 20.0],
                    [250.0, 20.0],
                ]
            ),
            "rawInd": np.array([100, 101, 102, 103]),
            "shankInd": np.array([0, 1, 0, 1]),
        },
        "spikes": {"exists": False},
        "clusters": {"exists": False},
    }

    plot_data = EphysPlotPayloadCache(Path("."), data, shank_idx=1)

    np.testing.assert_array_equal(plot_data.chn_ind, np.array([1, 3]))
    np.testing.assert_array_equal(plot_data.chn_rows, np.array([1, 3]))
