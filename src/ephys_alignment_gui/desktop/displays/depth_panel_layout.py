"""Shared layout metrics for linked desktop depth panels."""

from __future__ import annotations

from typing import Any

DEPTH_PANEL_HEADER_HEIGHT_PX = 90
DEPTH_PANEL_BOTTOM_AXIS_HEIGHT_PX = 42


def set_depth_panel_header_height(graphics_layout: Any) -> None:
    """Keep linked depth plot ViewBoxes vertically coaxial across panels."""
    layout = getattr(graphics_layout, "layout", None)
    set_row_fixed_height = getattr(layout, "setRowFixedHeight", None)
    if callable(set_row_fixed_height):
        set_row_fixed_height(0, DEPTH_PANEL_HEADER_HEIGHT_PX)


def set_depth_panel_bottom_axis(
    plot: Any,
    set_axis: Any,
    *,
    label: str | None = None,
    pen: Any = "k",
    ticks: bool = True,
) -> Any:
    """Configure a bottom axis without changing linked depth plot height."""
    axis = set_axis(plot, "bottom", label=label, pen=pen, ticks=ticks)
    set_height = getattr(axis, "setHeight", None)
    if callable(set_height):
        set_height(DEPTH_PANEL_BOTTOM_AXIS_HEIGHT_PX)
    return axis
