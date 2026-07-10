"""Tests for feature-depth view limit policy."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.view_limits import default_feature_y_limits


def test_default_feature_y_limits_use_full_probe_range_without_in_brain_depths():
    assert default_feature_y_limits(
        probe_tip_um=0,
        probe_top_um=9600,
        probe_extra_um=100,
    ) == (-100.0, 9700.0)


def test_default_feature_y_limits_cap_to_last_in_brain_channel_plus_margin():
    assert default_feature_y_limits(
        probe_tip_um=0,
        probe_top_um=9600,
        probe_extra_um=100,
        in_brain_depths_um=np.array([0, 750, 2200, 6100]),
    ) == (-100.0, 6600.0)


def test_default_feature_y_limits_never_expand_full_probe_range():
    assert default_feature_y_limits(
        probe_tip_um=0,
        probe_top_um=3840,
        probe_extra_um=100,
        in_brain_depths_um=np.array([0, 2000, 3800]),
    ) == (-100.0, 3940.0)


def test_default_feature_y_limits_ignore_empty_or_nan_in_brain_depths():
    assert default_feature_y_limits(
        probe_tip_um=0,
        probe_top_um=3840,
        probe_extra_um=100,
        in_brain_depths_um=np.array([np.nan]),
    ) == (-100.0, 3940.0)
