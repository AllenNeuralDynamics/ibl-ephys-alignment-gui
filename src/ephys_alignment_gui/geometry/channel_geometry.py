"""Helpers for producer-owned ephys channel geometry metadata."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def valid_shank_indices(
    shank_ind: NDArray | None,
    n_channels: int,
) -> NDArray | None:
    """Return validated 0-based shank indices, or None for legacy data."""
    if shank_ind is None:
        return None
    arr = np.asarray(shank_ind, dtype=int)
    if arr.ndim != 1 or arr.shape[0] != n_channels:
        return None
    return arr


def n_shanks_from_geometry(
    local_coordinates: NDArray,
    shank_ind: NDArray | None,
) -> int:
    """Return shank count from explicit metadata, else legacy x-gap geometry."""
    shank_ind = valid_shank_indices(shank_ind, local_coordinates.shape[0])
    if shank_ind is not None and shank_ind.size:
        return int(np.max(shank_ind)) + 1
    chn_x = np.unique(local_coordinates[:, 0])
    if chn_x.size <= 1:
        return 1
    return int(np.sum(np.diff(chn_x) > 100) + 1)


def rows_for_shank(
    local_coordinates: NDArray,
    shank_ind: NDArray | None,
    shank_idx: int,
    n_shanks: int,
) -> NDArray:
    """Return channel-table row positions for a 0-based ephys shank."""
    shank_ind = valid_shank_indices(shank_ind, local_coordinates.shape[0])
    if shank_ind is not None:
        return np.where(shank_ind == shank_idx)[0]
    return _legacy_rows_for_shank(local_coordinates, shank_idx, n_shanks)


def _legacy_rows_for_shank(
    local_coordinates: NDArray,
    shank_idx: int,
    n_shanks: int,
) -> NDArray:
    """Legacy x-coordinate fallback for datasets without channels.shankInd."""
    if n_shanks <= 1:
        return np.arange(local_coordinates.shape[0])

    chn_x = np.unique(local_coordinates[:, 0])
    start = shank_idx * 2
    stop = start + 2
    if stop <= chn_x.size:
        lo, hi = chn_x[start], chn_x[stop - 1]
        return np.where(
            (local_coordinates[:, 0] >= lo) & (local_coordinates[:, 0] <= hi)
        )[0]

    # Last-resort fallback for non-two-column legacy layouts: split x columns
    # at large gaps and use the requested chunk if it exists.
    breaks = np.where(np.diff(chn_x) > 100)[0] + 1
    chunks = np.split(chn_x, breaks)
    if 0 <= shank_idx < len(chunks):
        xs = chunks[shank_idx]
        return np.where(np.isin(local_coordinates[:, 0], xs))[0]
    return np.array([], dtype=int)
