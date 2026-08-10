"""Tests for plot color-level policies."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.plotting.level_policy import (
    in_brain_depth_mask,
    probe_colour_levels,
)


def test_in_brain_mask_none_when_unset() -> None:
    assert in_brain_depth_mask(np.array([0.0, 20.0, 40.0]), None) is None


def test_in_brain_mask_exact_path() -> None:
    mask = in_brain_depth_mask(
        np.array([0.0, 20.0, 40.0, 60.0]),
        np.array([20.0, 40.0]),
    )

    np.testing.assert_array_equal(mask, [False, True, True, False])


def test_in_brain_mask_binned_path() -> None:
    axis = np.array([0.0, 40.0, 80.0])
    mask = in_brain_depth_mask(
        axis,
        np.array([22.0, 41.0]),
        bin_width=40.0,
    )

    np.testing.assert_array_equal(mask, [False, True, False])


def test_in_brain_mask_none_when_no_overlap() -> None:
    assert (
        in_brain_depth_mask(
            np.array([0.0, 20.0, 40.0]),
            np.array([1000.0]),
        )
        is None
    )


def test_probe_levels_narrows_to_in_brain() -> None:
    values = np.array([1.0, 2.0, 3.0, 100.0])
    channel_depths_um = np.array([0.0, 20.0, 40.0, 60.0])
    full = probe_colour_levels(
        values,
        channel_depths_um=channel_depths_um,
        in_brain_depths_um=None,
    )
    masked = probe_colour_levels(
        values,
        channel_depths_um=channel_depths_um,
        in_brain_depths_um=np.array([0.0, 20.0, 40.0]),
    )

    assert masked[1] < full[1]
