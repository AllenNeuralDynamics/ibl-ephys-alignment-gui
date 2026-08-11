"""Tests for anatomical atlas safety checks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ephys_alignment_gui.geometry.anatomical_atlas import _labels_to_region_indices


def test_labels_to_region_indices_rejects_unknown_region_ids() -> None:
    regions = SimpleNamespace(id=np.array([0, 997, 1009], dtype=np.int64))
    label_ids = np.array([[0, 997], [123456, 1009]], dtype=np.int64)

    with pytest.raises(ValueError, match="123456"):
        _labels_to_region_indices(label_ids, regions)


def test_labels_to_region_indices_returns_brain_region_rows() -> None:
    regions = SimpleNamespace(id=np.array([0, 997, 1009], dtype=np.int64))
    label_ids = np.array([[0, 997], [1009, 0]], dtype=np.int64)

    np.testing.assert_array_equal(
        _labels_to_region_indices(label_ids, regions),
        np.array([[0, 1], [2, 0]], dtype=np.int16),
    )
