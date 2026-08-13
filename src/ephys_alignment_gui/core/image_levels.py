"""Image display level helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_BRAIN_LEVEL_PERCENTILES = (0.5, 95.0)


def brain_percentile_levels(
    image: ArrayLike,
    annotation_ids: ArrayLike | None,
    lower: float = DEFAULT_BRAIN_LEVEL_PERCENTILES[0],
    upper: float = DEFAULT_BRAIN_LEVEL_PERCENTILES[1],
) -> tuple[float, float] | None:
    """Return percentile display levels from CCF-annotated brain voxels.

    ``annotation_ids != 0`` defines the brain mask. The annotation input must be
    the raw 2-D CCF annotation-ID slice, not the RGB display label image.
    The low default is intentionally close to the brain minimum so background
    voxels stay dark without clipping much in-brain signal to the same color.
    Returns ``None`` when the mask is unavailable or degenerate so callers can
    fall back to existing behavior.
    """
    if annotation_ids is None:
        return None

    image_arr = np.asarray(image)
    annotation_arr = np.asarray(annotation_ids)
    if image_arr.ndim != 2 or annotation_arr.shape != image_arr.shape:
        return None

    brain_mask = annotation_arr != 0
    values = image_arr[brain_mask & np.isfinite(image_arr)]
    if values.size == 0:
        return None

    lo, hi = np.nanpercentile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        value_min = np.nanmin(values)
        value_max = np.nanmax(values)
        if not np.isfinite(value_min) or not np.isfinite(value_max):
            return None
        if value_min == value_max:
            return None
        lo, hi = value_min, value_max
    return float(lo), float(hi)
