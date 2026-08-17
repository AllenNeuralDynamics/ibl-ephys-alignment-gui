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


def average_equal_depth_channels(values, channel_depths_um):
    """Average columns that share a physical channel depth.

    Image-style probe payloads collapse left/right contacts at the same depth
    into one row. Channel-table order is not a geometry contract, so grouping
    must use physical depth coordinates instead of adjacent column positions.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape {values.shape}")
    channel_depths_um = np.asarray(channel_depths_um, dtype=float)
    if channel_depths_um.shape != (values.shape[1],):
        raise ValueError(
            "channel_depths_um must have one entry per value column: "
            f"{channel_depths_um.shape} != ({values.shape[1]},)"
        )

    unique_depths = np.unique(channel_depths_um)
    averaged = np.empty((values.shape[0], unique_depths.size), dtype=float)
    for depth_idx, depth in enumerate(unique_depths):
        columns = channel_depths_um == depth
        group = values[:, columns]
        finite = np.isfinite(group)
        counts = np.sum(finite, axis=1)
        sums = np.sum(np.where(finite, group, 0.0), axis=1)
        averaged[:, depth_idx] = np.divide(
            sums,
            counts,
            out=np.full(values.shape[0], np.nan),
            where=counts > 0,
        )
    return averaged
