"""Tests for image display level helpers."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.image_levels import brain_percentile_levels


def test_brain_percentile_levels_uses_nonzero_annotation_ids():
    image = np.arange(100, dtype=float).reshape(10, 10)
    annotation_ids = np.zeros((10, 10), dtype=np.uint16)
    annotation_ids[2:8, 2:8] = 997

    levels = brain_percentile_levels(image, annotation_ids)

    expected = np.percentile(image[2:8, 2:8], [5, 95])
    assert levels == (expected[0], expected[1])


def test_brain_percentile_levels_rejects_rgb_display_labels():
    image = np.arange(9, dtype=float).reshape(3, 3)
    display_labels = np.zeros((3, 3, 3), dtype=np.uint8)
    display_labels[1:, 1:, 2] = 255

    assert brain_percentile_levels(image, display_labels) is None


def test_brain_percentile_levels_ignores_nonfinite_values():
    image = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]], dtype=float)
    annotation_ids = np.ones((3, 3), dtype=np.uint8)

    levels = brain_percentile_levels(image, annotation_ids)

    expected = np.percentile(image[np.isfinite(image)], [5, 95])
    assert levels == (expected[0], expected[1])


def test_brain_percentile_levels_returns_none_for_missing_mask():
    image = np.arange(9, dtype=float).reshape(3, 3)

    assert brain_percentile_levels(image, None) is None


def test_brain_percentile_levels_returns_none_for_constant_masked_values():
    image = np.ones((3, 3), dtype=float)
    annotation_ids = np.ones((3, 3), dtype=np.uint8)

    assert brain_percentile_levels(image, annotation_ids) is None
