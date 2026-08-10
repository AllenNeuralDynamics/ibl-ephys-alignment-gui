"""Tests for the active alignment value object."""

from __future__ import annotations

import numpy as np
import pytest

from ephys_alignment_gui.core.active_alignment import ActiveAlignment


def test_active_alignment_copies_and_freezes_arrays() -> None:
    feature = np.array([0.0, 1.0])
    track = np.array([0.0, 2.0])

    alignment = ActiveAlignment(feature, track, lin_fit=False)
    feature[0] = 99.0
    track[0] = 99.0

    np.testing.assert_array_equal(alignment.feature, [0.0, 1.0])
    np.testing.assert_array_equal(alignment.track, [0.0, 2.0])
    assert not alignment.lin_fit
    assert not alignment.feature.flags.writeable
    assert not alignment.track.flags.writeable


def test_active_alignment_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        ActiveAlignment(np.array([0.0, 1.0]), np.array([0.0]))


def test_from_values_returns_none_for_legacy_blank_sentinel() -> None:
    assert ActiveAlignment.from_values(0, 0) is None
    assert ActiveAlignment.from_values(None, np.array([0.0])) is None


def test_active_alignment_returns_mutable_copies() -> None:
    alignment = ActiveAlignment(np.array([0.0, 1.0]), np.array([0.0, 2.0]))

    feature = alignment.feature_copy()
    track = alignment.track_copy()
    feature[0] = 10.0
    track[0] = 20.0

    np.testing.assert_array_equal(alignment.feature, [0.0, 1.0])
    np.testing.assert_array_equal(alignment.track, [0.0, 2.0])
