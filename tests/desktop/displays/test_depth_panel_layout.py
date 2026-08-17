"""Tests for shared depth-panel layout metrics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop.displays.depth_panel_layout import (
    DEPTH_PANEL_BOTTOM_AXIS_HEIGHT_PX,
    DEPTH_PANEL_HEADER_HEIGHT_PX,
    set_depth_panel_bottom_axis,
    set_depth_panel_header_height,
)


class FakeLayout:
    def __init__(self) -> None:
        self.fixed_heights: list[tuple[int, int]] = []

    def setRowFixedHeight(self, row: int, height: int) -> None:
        self.fixed_heights.append((row, height))


class FakeAxis:
    def __init__(self) -> None:
        self.heights: list[int] = []

    def setHeight(self, height: int) -> None:
        self.heights.append(height)


def test_depth_panel_header_height_is_fixed_for_linked_depth_plots() -> None:
    layout = FakeLayout()
    graphics_layout = SimpleNamespace(layout=layout)

    set_depth_panel_header_height(graphics_layout)

    assert layout.fixed_heights == [(0, DEPTH_PANEL_HEADER_HEIGHT_PX)]


def test_depth_panel_header_height_ignores_non_qt_layout_fakes() -> None:
    set_depth_panel_header_height(SimpleNamespace(layout=object()))


def test_depth_panel_bottom_axis_uses_fixed_height() -> None:
    axis = FakeAxis()
    calls: list[tuple[Any, ...]] = []

    result = set_depth_panel_bottom_axis(
        "plot",
        lambda *args, **kwargs: calls.append((args, kwargs)) or axis,
        label="Depth",
        ticks=False,
    )

    assert result is axis
    assert calls == [
        (("plot", "bottom"), {"label": "Depth", "pen": "k", "ticks": False})
    ]
    assert axis.heights == [DEPTH_PANEL_BOTTOM_AXIS_HEIGHT_PX]
