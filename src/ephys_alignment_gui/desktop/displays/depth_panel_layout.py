"""Shared layout metrics for linked desktop depth panels."""

from __future__ import annotations

from typing import Any

DEPTH_PANEL_HEADER_HEIGHT_PX = 90
DEPTH_PANEL_BOTTOM_AXIS_HEIGHT_PX = 42


def clear_depth_panel_title(plot: Any) -> None:
    """Remove title rows from depth-linked plots.

    Linked depth panels rely on physically coaxial ViewBoxes, not just matching
    y ranges. Pyqtgraph PlotItem titles add a plot-local top row and shift that
    ViewBox, so depth panel labels must live in fixed-height axes or shared
    layout rows instead.
    """
    set_title = getattr(plot, "setTitle", None)
    if callable(set_title):
        try:
            set_title(None)
        except TypeError:
            set_title("")
    title_label = getattr(plot, "titleLabel", None)
    hide = getattr(title_label, "hide", None)
    if callable(hide):
        hide()
    for method_name in (
        "setMinimumHeight",
        "setMaximumHeight",
        "setPreferredHeight",
    ):
        set_height = getattr(title_label, method_name, None)
        if callable(set_height):
            set_height(0)


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


def set_depth_panel_strip_label(
    plot: Any,
    set_axis: Any,
    label: str,
    *,
    pen: Any = "k",
) -> Any:
    """Label a narrow depth strip without changing linked plot geometry."""
    clear_depth_panel_title(plot)
    axis = set_depth_panel_bottom_axis(
        plot,
        set_axis,
        label=label,
        pen=pen,
        ticks=False,
    )
    set_style = getattr(axis, "setStyle", None)
    if callable(set_style):
        set_style(showValues=False)
    return axis
