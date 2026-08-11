"""Tests for desktop depth-plot view helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.desktop.views.depth_plot_view import DesktopDepthPlotView


class FakePlot:
    def __init__(self, y_range: tuple[float, float] = (1.0, 2.0)) -> None:
        self.y_range = y_range
        self.set_ranges: list[dict[str, float]] = []

    def viewRange(self) -> list[Any]:
        return [[0.0, 1.0], list(self.y_range)]

    def setYRange(self, **kwargs: float) -> None:
        self.set_ranges.append(kwargs)


class FakeLine:
    def __init__(self) -> None:
        self.y_values: list[float] = []

    def setY(self, value: float) -> None:
        self.y_values.append(value)


def _depth_view(
) -> tuple[DesktopDepthPlotView, dict[str, FakePlot], FakeLine, FakeLine]:
    plots = {
        "image": FakePlot((10.0, 20.0)),
        "histology": FakePlot((30.0, 40.0)),
    }
    tip_line = FakeLine()
    top_line = FakeLine()
    return (
        DesktopDepthPlotView(
            default_range_plots=tuple(plots.values()),
            range_plots=plots,
            probe_tip_lines=[tip_line],
            probe_top_lines=[top_line],
            padding=lambda: 0.05,
        ),
        plots,
        tip_line,
        top_line,
    )


def test_set_probe_limits_updates_guide_lines() -> None:
    view, _plots, tip_line, top_line = _depth_view()

    view.set_probe_limits(-50.0, 500.0)

    assert tip_line.y_values == [-50.0]
    assert top_line.y_values == [500.0]


def test_default_feature_y_range_uses_brain_limited_depth_policy() -> None:
    depth_settings = SimpleNamespace(
        probe_tip_um=0.0,
        probe_top_um=1000.0,
        probe_extra_um=100.0,
    )
    view, plots, _tip_line, _top_line = _depth_view()

    view.set_default_feature_y_range(
        depth_view=depth_settings,
        in_brain_depths_um=np.array([100.0, 300.0]),
    )

    assert plots["image"].set_ranges == [
        {"min": -100.0, "max": 800.0, "padding": 0.05}
    ]
    assert plots["histology"].set_ranges == [
        {"min": -100.0, "max": 800.0, "padding": 0.05}
    ]


def test_capture_and_restore_y_ranges() -> None:
    view, plots, _tip_line, _top_line = _depth_view()

    ranges = view.capture_y_ranges()
    view.restore_y_ranges({"image": (100.0, 200.0), "missing": (1.0, 2.0)})

    assert ranges == {"image": (10.0, 20.0), "histology": (30.0, 40.0)}
    assert plots["image"].set_ranges == [
        {"min": 100.0, "max": 200.0, "padding": 0}
    ]
    assert plots["histology"].set_ranges == []
