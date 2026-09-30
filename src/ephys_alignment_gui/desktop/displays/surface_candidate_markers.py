"""Brain-surface candidate markers for the line-amplitude plots.

A candidate is display-only and must never read as a reference line, which is
a full-width, draggable, 2 px line in a random palette colour: candidates are
shapes in one fixed colour outside that palette, with no hover or drag.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg

SURFACE_MARKER_COLOR = "#51606f"
# Below reference lines (z=100), so those stay on top and grabbable.
_MARKER_Z = 60
_WEDGE_SIZE_PX = 14
_DIAMOND_SIZE_PX = 12
_EDGE_INSET = 0.03  # of the x span, so the wedge is not clipped
_DISAGREEING_ALPHA = 90


def _pen(width: float, alpha: int = 255):
    color = pg.mkColor(SURFACE_MARKER_COLOR)
    color.setAlpha(alpha)
    return pg.mkPen(color, width=width)


def _inert(item: Any) -> Any:
    item.setZValue(_MARKER_Z)
    item.setAcceptHoverEvents(False)
    return item


def image_marker_items(markers: Any, xrange: Any) -> list[Any]:
    """A wedge at the right edge at each midpoint, with a bar over the
    candidate's width."""
    if not markers:
        return []
    x = float(xrange[1]) - _EDGE_INSET * float(xrange[1] - xrange[0])
    items = []
    wedges = pg.ScatterPlotItem(
        x=[x] * len(markers),
        y=[m["midpoint_um"] for m in markers],
        symbol="t3",
        size=_WEDGE_SIZE_PX,
        pen=_pen(1),
        brush=pg.mkBrush(SURFACE_MARKER_COLOR),
        hoverable=False,
    )
    items.append(_inert(wedges))
    for marker in markers:
        width = marker.get("width_um")
        if not width:
            continue
        half = float(width) / 2
        bar = pg.PlotCurveItem(
            x=[x, x],
            y=[marker["midpoint_um"] - half, marker["midpoint_um"] + half],
            pen=_pen(3),
        )
        items.append(_inert(bar))
    return items


def line_marker_items(payload: Any) -> list[Any]:
    """The fitted curve, and for each candidate the curve's 10-90 % extent
    highlighted with a hollow diamond at the midpoint; faint where the line
    does not agree with the candidate."""
    fit = payload.get("fit")
    if not fit:
        return []
    fit_x = np.asarray(fit["x"], dtype=float)
    fit_y = np.asarray(fit["y"], dtype=float)
    items = [
        _inert(pg.PlotCurveItem(x=fit_x, y=fit_y, pen=_pen(1.5), connect="finite"))
    ]
    for marker in payload.get("surface_candidates") or ():
        alpha = 255 if marker.get("agrees") else _DISAGREEING_ALPHA
        lo, hi = marker["extent_um"]
        on_extent = (fit_y >= lo) & (fit_y <= hi) & np.isfinite(fit_x)
        if np.count_nonzero(on_extent) >= 2:
            items.append(
                _inert(
                    pg.PlotCurveItem(
                        x=fit_x[on_extent],
                        y=fit_y[on_extent],
                        pen=_pen(4, alpha),
                    )
                )
            )
        if marker.get("curve_db") is None:
            continue
        items.append(
            _inert(
                pg.ScatterPlotItem(
                    x=[marker["curve_db"]],
                    y=[marker["midpoint_um"]],
                    symbol="d",
                    size=_DIAMOND_SIZE_PX,
                    pen=_pen(2, alpha),
                    brush=pg.mkBrush(None),
                    hoverable=False,
                )
            )
        )
    return items
