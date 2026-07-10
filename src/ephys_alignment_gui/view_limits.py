"""View-limit policy for feature-depth plots."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

IN_BRAIN_VIEW_MARGIN_UM = 500.0


def default_feature_y_limits(
    *,
    probe_tip_um: float,
    probe_top_um: float,
    probe_extra_um: float,
    in_brain_depths_um: ArrayLike | None = None,
    in_brain_margin_um: float = IN_BRAIN_VIEW_MARGIN_UM,
) -> tuple[float, float]:
    """Return the initial feature-depth y-limits in microns.

    The lower bound keeps the existing probe-tip margin. When CCF labels identify
    in-brain channels, the upper bound is capped to the last in-brain channel
    plus a small margin so long surface-finding tails do not dominate the
    starting view.
    """
    y_min = float(probe_tip_um) - float(probe_extra_um)
    full_y_max = float(probe_top_um) + float(probe_extra_um)

    if in_brain_depths_um is None:
        return y_min, full_y_max

    in_brain = np.asarray(in_brain_depths_um, dtype=float)
    in_brain = in_brain[np.isfinite(in_brain)]
    if in_brain.size == 0:
        return y_min, full_y_max

    capped_y_max = float(np.max(in_brain)) + float(in_brain_margin_um)
    y_max = min(full_y_max, capped_y_max)
    if y_max <= y_min:
        return y_min, full_y_max
    return y_min, y_max
