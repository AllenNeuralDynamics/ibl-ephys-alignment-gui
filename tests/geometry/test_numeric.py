"""Tests for small numeric helpers."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.geometry.numeric import bincount2D


def test_bincount2d_bins_points_into_dense_grid() -> None:
    counts, xscale, yscale = bincount2D(
        np.array([0.1, 0.2, 1.2, 1.8]),
        np.array([0.1, 0.9, 0.1, 1.1]),
        xbin=1.0,
        ybin=1.0,
        xlim=[0.0, 2.0],
        ylim=[0.0, 2.0],
    )

    np.testing.assert_array_equal(xscale, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(yscale, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(
        counts,
        np.array(
            [
                [2, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    )
