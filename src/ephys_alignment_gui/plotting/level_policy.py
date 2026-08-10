"""Qt-free color-level policies for plot-data payloads."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def in_brain_depth_mask(
    depth_axis_um: Any,
    in_brain_depths_um: Any,
    *,
    bin_width: float | None = None,
) -> NDArray[np.bool_] | None:
    """Return a mask over a depth axis selecting in-brain depths.

    Returns ``None`` when the in-brain depth set is unset/empty or nothing maps
    to the target axis. With ``bin_width`` provided, the axis is treated as
    regularly spaced bin centers and each in-brain channel marks its nearest
    bin. Without ``bin_width``, channel depths must match exactly.
    """
    if depth_axis_um is None or in_brain_depths_um is None:
        return None
    axis = np.asarray(depth_axis_um, dtype=float)
    in_brain = np.asarray(in_brain_depths_um, dtype=float)
    if axis.size == 0 or in_brain.size == 0:
        return None
    if bin_width is None:
        mask = np.isin(axis, in_brain)
    else:
        idx = np.round((in_brain - axis[0]) / bin_width).astype(int)
        idx = idx[(idx >= 0) & (idx < axis.size)]
        mask = np.zeros(axis.size, dtype=bool)
        mask[idx] = True
    if not mask.any():
        return None
    return mask


def probe_colour_levels(
    values: Any,
    *,
    channel_depths_um: Any,
    in_brain_depths_um: Any,
    quantiles: tuple[float, float] = (0.1, 0.9),
) -> NDArray:
    """Return probe color levels from in-brain channels when known."""
    vals = np.asarray(values, dtype=float)
    mask = in_brain_depth_mask(channel_depths_um, in_brain_depths_um)
    if mask is not None and mask.shape[0] == vals.shape[0]:
        vals = vals[mask]
    return np.nanquantile(vals, list(quantiles))
