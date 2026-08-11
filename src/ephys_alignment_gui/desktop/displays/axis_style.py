"""Reusable pyqtgraph axis styling helpers for desktop displays."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PyQt5 import QtGui


def axis_item(fig: Any, orientation: str) -> Any:
    """Return a pyqtgraph axis item from a PlotItem or PlotWidget-like object."""
    if isinstance(fig, pg.PlotItem):
        return fig.getAxis(orientation)
    return fig.plotItem.getAxis(orientation)


def set_axis(
    fig: Any,
    orientation: str,
    show: bool = True,
    label: str | None = None,
    pen: Any = "k",
    ticks: bool = True,
) -> Any:
    """Show/hide and configure one pyqtgraph axis."""
    axis = axis_item(fig, orientation)
    if show:
        axis.show()
        axis.setPen(pen)
        axis.setTextPen(pen)
        axis.setLabel(label or "")
        if not ticks:
            axis.setTicks([[(0, ""), (0.5, ""), (1, "")]])
    else:
        axis.hide()
    return axis


def set_font(
    fig: Any,
    orientation: str,
    ptsize: int = 8,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Apply tick and label font styling to one pyqtgraph axis."""
    axis = axis_item(fig, orientation)
    font = QtGui.QFont()
    font.setPointSize(ptsize)
    axis.setStyle(tickFont=font)
    axis.setLabel(**{"font-size": f"{ptsize}pt"})

    if width:
        axis.setWidth(width)
    if height:
        axis.setHeight(height)
