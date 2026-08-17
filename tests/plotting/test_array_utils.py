"""Tests for small plot array helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ephys_alignment_gui.plotting.array_utils import average_equal_depth_channels


def test_average_equal_depth_channels_groups_by_physical_depth() -> None:
    values = np.array(
        [
            [1.0, 10.0, 3.0, 30.0],
            [2.0, 20.0, 6.0, 60.0],
        ]
    )
    depths = np.array([0.0, 15.0, 0.0, 30.0])

    averaged = average_equal_depth_channels(values, depths)

    np.testing.assert_allclose(
        averaged,
        [
            [2.0, 10.0, 30.0],
            [4.0, 20.0, 60.0],
        ],
    )


def test_average_equal_depth_channels_handles_all_nan_depth_without_warning() -> None:
    values = np.array(
        [
            [np.nan, 1.0, np.nan],
            [np.nan, 2.0, np.nan],
        ]
    )
    depths = np.array([0.0, 15.0, 0.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        averaged = average_equal_depth_channels(values, depths)

    np.testing.assert_allclose(averaged[:, 1], [1.0, 2.0])
    assert np.all(np.isnan(averaged[:, 0]))


def test_average_equal_depth_channels_validates_depth_count() -> None:
    with pytest.raises(ValueError, match="one entry per value column"):
        average_equal_depth_channels(
            np.ones((2, 3)),
            np.array([0.0, 15.0]),
        )
