"""Tests for shared depth-panel layout metrics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop.displays.depth_panel_layout import (
    DEPTH_PANEL_BOTTOM_AXIS_HEIGHT_PX,
    DEPTH_PANEL_HEADER_HEIGHT_PX,
    clear_depth_panel_title,
    set_depth_panel_bottom_axis,
    set_depth_panel_header_height,
    set_depth_panel_strip_label,
)


class FakeLayout:
    def __init__(self) -> None:
        self.fixed_heights: list[tuple[int, int]] = []

    def setRowFixedHeight(self, row: int, height: int) -> None:
        self.fixed_heights.append((row, height))


class FakeAxis:
    def __init__(self) -> None:
        self.heights: list[int] = []
        self.styles: list[dict[str, Any]] = []

    def setHeight(self, height: int) -> None:
        self.heights.append(height)

    def setStyle(self, **kwargs: Any) -> None:
        self.styles.append(kwargs)


class FakeTitleLabel:
    def __init__(self) -> None:
        self.hide_count = 0
        self.minimum_heights: list[int] = []
        self.maximum_heights: list[int] = []
        self.preferred_heights: list[int] = []

    def hide(self) -> None:
        self.hide_count += 1

    def setMinimumHeight(self, height: int) -> None:
        self.minimum_heights.append(height)

    def setMaximumHeight(self, height: int) -> None:
        self.maximum_heights.append(height)

    def setPreferredHeight(self, height: int) -> None:
        self.preferred_heights.append(height)


class FakePlotWithTitle:
    def __init__(self) -> None:
        self.title_calls: list[Any] = []
        self.titleLabel = FakeTitleLabel()

    def setTitle(self, value: Any) -> None:
        self.title_calls.append(value)


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


def test_clear_depth_panel_title_removes_plotitem_title_row() -> None:
    plot = FakePlotWithTitle()

    clear_depth_panel_title(plot)

    assert plot.title_calls == [None]
    assert plot.titleLabel.hide_count == 1
    assert plot.titleLabel.minimum_heights == [0]
    assert plot.titleLabel.maximum_heights == [0]
    assert plot.titleLabel.preferred_heights == [0]


def test_depth_panel_strip_label_uses_fixed_bottom_axis_not_title() -> None:
    plot = FakePlotWithTitle()
    axis = FakeAxis()
    calls: list[tuple[Any, ...]] = []

    result = set_depth_panel_strip_label(
        plot,
        lambda *args, **kwargs: calls.append((args, kwargs)) or axis,
        "Warped",
    )

    assert result is axis
    assert plot.title_calls == [None]
    assert plot.titleLabel.hide_count == 1
    assert plot.titleLabel.minimum_heights == [0]
    assert plot.titleLabel.maximum_heights == [0]
    assert plot.titleLabel.preferred_heights == [0]
    assert calls == [
        ((plot, "bottom"), {"label": "Warped", "pen": "k", "ticks": False})
    ]
    assert axis.heights == [DEPTH_PANEL_BOTTOM_AXIS_HEIGHT_PX]
    assert axis.styles == [{"showValues": False}]
