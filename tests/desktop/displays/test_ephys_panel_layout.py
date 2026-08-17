"""Tests for desktop ephys panel layout switching."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop.displays.ephys_panel_layout import (
    DesktopEphysPanelLayout,
    EphysPanelLayoutCallbacks,
    EphysPanelLayoutSizes,
)
from ephys_alignment_gui.desktop.displays.ephys_panel_view import EphysPanelPlots


class FakePlot:
    def __init__(self, name: str) -> None:
        self.name = name
        self.preferred_widths: list[float] = []
        self.fixed_widths: list[float] = []
        self.x_ranges: list[dict[str, Any]] = []
        self.updates = 0

    def setPreferredWidth(self, width: float) -> None:
        self.preferred_widths.append(width)

    def setFixedWidth(self, width: float) -> None:
        self.fixed_widths.append(width)

    def setXRange(self, **kwargs: Any) -> None:
        self.x_ranges.append(kwargs)

    def update(self) -> None:
        self.updates += 1


class FakeInnerLayout:
    def __init__(self) -> None:
        self.column_stretches: list[tuple[int, int]] = []
        self.row_stretches: list[tuple[int, int]] = []

    def setColumnStretchFactor(self, column: int, factor: int) -> None:
        self.column_stretches.append((column, factor))

    def setRowStretchFactor(self, row: int, factor: int) -> None:
        self.row_stretches.append((row, factor))


class FakeGraphicsLayout:
    def __init__(self) -> None:
        self.removed: list[Any] = []
        self.added: list[tuple[Any, tuple[Any, ...]]] = []
        self.layout = FakeInnerLayout()

    def removeItem(self, item: Any) -> None:
        self.removed.append(item)

    def addItem(self, item: Any, *args: Any) -> None:
        self.added.append((item, args))


def _layout(
    *,
    feature_xrange: tuple[float, float] | None = (10.0, 20.0),
) -> tuple[DesktopEphysPanelLayout, dict[str, FakePlot], FakeGraphicsLayout, list[Any]]:
    plots = {
        "image": FakePlot("image"),
        "image_colorbar": FakePlot("image-colorbar"),
        "line": FakePlot("line"),
        "probe": FakePlot("probe"),
        "probe_colorbar": FakePlot("probe-colorbar"),
    }
    graphics_layout = FakeGraphicsLayout()
    axis_calls: list[Any] = []
    reset_calls: list[Any] = []
    layout = DesktopEphysPanelLayout(
        panel=SimpleNamespace(
            plots=EphysPanelPlots(**plots),
            feature_xrange=feature_xrange,
        ),
        graphics_layout=graphics_layout,
        callbacks=EphysPanelLayoutCallbacks(
            set_axis=lambda *args, **kwargs: axis_calls.append((args, kwargs)),
            reset_axis=lambda: reset_calls.append("reset"),
        ),
    )
    return layout, plots, graphics_layout, [axis_calls, reset_calls]


def _sizes() -> EphysPanelLayoutSizes:
    return EphysPanelLayoutSizes(
        axis_width=5,
        image_width=60,
        line_width=30,
        probe_width=10,
    )


def test_ephys_panel_layout_applies_view_1_image_line_probe() -> None:
    layout, plots, graphics_layout, calls = _layout()

    layout.apply_view(1, _sizes())

    assert graphics_layout.added == [
        (plots["image_colorbar"], (0, 0)),
        (plots["probe_colorbar"], (0, 1, 1, 2)),
        (plots["image"], (1, 0)),
        (plots["line"], (1, 1)),
        (plots["probe"], (1, 2)),
    ]
    assert plots["image"].preferred_widths == [65]
    assert plots["line"].preferred_widths == [30]
    assert plots["probe"].fixed_widths == [10]
    assert graphics_layout.layout.column_stretches == [(0, 6), (1, 1), (2, 1)]
    assert calls[1] == ["reset"]


def test_ephys_panel_layout_applies_view_2_image_probe_line() -> None:
    layout, plots, graphics_layout, _calls = _layout()

    layout.apply_view(2, _sizes())

    assert graphics_layout.added == [
        (plots["image_colorbar"], (0, 0)),
        (plots["probe_colorbar"], (0, 1, 1, 2)),
        (plots["image"], (1, 0)),
        (plots["probe"], (1, 1)),
        (plots["line"], (1, 2)),
    ]


def test_ephys_panel_layout_applies_view_3_probe_line_image() -> None:
    layout, plots, graphics_layout, calls = _layout()

    layout.apply_view(3, _sizes())

    assert graphics_layout.added == [
        (plots["probe_colorbar"], (0, 0, 1, 2)),
        (plots["image_colorbar"], (0, 2)),
        (plots["probe"], (1, 0)),
        (plots["line"], (1, 1)),
        (plots["image"], (1, 2)),
    ]
    assert plots["probe"].fixed_widths == [15]
    assert plots["image"].preferred_widths == [60]
    assert plots["line"].preferred_widths == [30]
    assert graphics_layout.layout.column_stretches == [(0, 1), (1, 1), (2, 6)]
    assert (
        (plots["probe"], "left"),
        {"label": "Distance from probe tip (µm)"},
    ) in calls[0]


def test_ephys_panel_layout_refreshes_plots_and_feature_xrange() -> None:
    layout, plots, _graphics_layout, _calls = _layout(feature_xrange=(4.0, 9.0))

    layout.apply_view(1, _sizes())

    assert plots["image"].updates == 1
    assert plots["line"].updates == 1
    assert plots["probe"].updates == 1
    assert plots["image"].x_ranges == [{"min": -6.0, "max": 19.0, "padding": 0}]


def test_ephys_panel_layout_ignores_unknown_view() -> None:
    layout, _plots, graphics_layout, calls = _layout()

    layout.apply_view(99, _sizes())

    assert graphics_layout.added == []
    assert calls[1] == []
