"""Small array helpers used by plot payload builders."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def safe_take(arr, indices, axis=0):
    """Return ``np.take`` with out-of-bounds positions filled with NaN."""
    max_idx = arr.shape[axis] - 1
    oob = indices > max_idx
    if np.any(oob):
        logger.warning(
            "Channel indices exceed data size "
            "(max_idx=%s, max_chn_ind=%s). Filling %s channels with NaN.",
            max_idx,
            indices.max(),
            np.sum(oob),
        )
        safe_indices = np.clip(indices, 0, max_idx)
        result = np.take(arr, safe_indices, axis=axis).astype(float)
        slices = [slice(None)] * result.ndim
        slices[axis] = oob
        result[tuple(slices)] = np.nan
        return result
    return np.take(arr, indices, axis=axis)
