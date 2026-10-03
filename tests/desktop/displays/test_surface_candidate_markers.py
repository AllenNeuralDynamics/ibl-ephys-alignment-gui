"""Tests for brain-surface candidate markers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ephys_alignment_gui.desktop.displays import surface_candidate_markers
from ephys_alignment_gui.desktop.displays.surface_candidate_markers import (
    SURFACE_MARKER_COLOR,
    image_marker_items,
    line_marker_items,
)

REFERENCE_LINE_PALETTE = {
    "#cc0000",
    "#6aa84f",
    "#ff8d00",
    "#00fff7",
    "#03fc84",
    "#fc03e7",
    "#1c03fc",
    "#000000",
}


class FakeColor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.alpha = 255

    def setAlpha(self, alpha: int) -> None:
        self.alpha = alpha


class FakeItem:
    def __init__(self, kind: str, **kwargs: Any) -> None:
        self.kind = kind
        self.kwargs = kwargs
        self.z: Any = None
        self.hover: Any = None

    def setZValue(self, z: Any) -> None:
        self.z = z

    def setAcceptHoverEvents(self, accept: bool) -> None:
        self.hover = accept


@pytest.fixture(autouse=True)
def fake_pg(monkeypatch):
    monkeypatch.setattr(
        surface_candidate_markers,
        "pg",
        SimpleNamespace(
            mkColor=FakeColor,
            mkPen=lambda color, width, **kw: {"color": color, "width": width, **kw},
            mkBrush=lambda color: {"brush": color},
            PlotCurveItem=lambda **kw: FakeItem("curve", **kw),
            ScatterPlotItem=lambda **kw: FakeItem("scatter", **kw),
        ),
    )


def _line_payload(agrees: bool = True) -> dict[str, Any]:
    depths = np.arange(0.0, 200.0, 10.0)
    return {
        "x": np.zeros(depths.size),
        "y": depths,
        "fits": {"lines": {"x": np.where(depths > 100, 5.0, 0.0), "y": depths}},
        "surface_candidates": (
            {
                "midpoint_um": 100.0,
                "width_um": 20.0,
                "method": "lines",
                "agrees": agrees,
                "extent_um": (90.0, 130.0),
                "curve_db": 2.5,
            },
        ),
    }


def test_marker_colour_is_outside_the_reference_line_palette():
    assert SURFACE_MARKER_COLOR.lower() not in REFERENCE_LINE_PALETTE


def test_no_markers_without_candidates_or_fit():
    assert image_marker_items((), (0.0, 4.0)) == []
    assert line_marker_items({"x": [1.0], "y": [2.0]}) == []
    assert line_marker_items({"fits": {}, "surface_candidates": ()}) == []


def test_image_markers_are_a_right_edge_wedge_and_width_bar():
    markers = ({"midpoint_um": 100.0, "width_um": 20.0, "method": "lines"},)

    wedges, bar = image_marker_items(markers, (0.0, 4.0))

    assert wedges.kwargs["symbol"] == "t3"
    assert wedges.kwargs["y"] == [100.0]
    (x,) = wedges.kwargs["x"]
    assert 3.5 < x < 4.0
    assert bar.kwargs["x"] == [x, x]
    assert bar.kwargs["y"] == [90.0, 110.0]


def test_line_markers_highlight_extent_and_mark_midpoint():
    fit, extent, diamond = line_marker_items(_line_payload())

    np.testing.assert_array_equal(
        fit.kwargs["y"], _line_payload()["fits"]["lines"]["y"]
    )
    assert "style" not in fit.kwargs["pen"]
    assert extent.kwargs["y"].min() >= 90.0
    assert extent.kwargs["y"].max() <= 130.0
    assert extent.kwargs["pen"]["width"] > fit.kwargs["pen"]["width"]
    assert diamond.kwargs["symbol"] == "d"
    assert (diamond.kwargs["x"], diamond.kwargs["y"]) == ([2.5], [100.0])
    assert diamond.kwargs["brush"] == {"brush": None}


def test_disagreeing_line_markers_are_faint():
    _fit, extent, diamond = line_marker_items(_line_payload(agrees=False))

    assert extent.kwargs["pen"]["color"].alpha < 255
    assert diamond.kwargs["pen"]["color"].alpha < 255


def test_markers_do_not_take_hover_and_sit_below_reference_lines():
    items = line_marker_items(_line_payload()) + image_marker_items(
        _line_payload()["surface_candidates"], (0.0, 4.0)
    )

    assert all(item.hover is False for item in items)
    assert all(item.z < 100 for item in items)


def test_each_method_has_its_own_curve_and_markers_follow_their_method():
    payload = _line_payload()
    depths = payload["y"]
    payload["fits"]["plateau"] = {"x": np.where(depths > 50, 1.0, -1.0), "y": depths}
    payload["surface_candidates"] += (
        {
            "midpoint_um": 50.0,
            "width_um": None,
            "method": "plateau",
            "agrees": True,
            "extent_um": (50.0, 50.0),
            "curve_db": 0.0,
        },
    )

    lines_fit, plateau_fit, extent, lines_diamond, plateau_diamond = line_marker_items(
        payload
    )

    assert "style" in plateau_fit.kwargs["pen"]
    assert "style" not in lines_fit.kwargs["pen"]
    assert extent.kwargs["y"].min() >= 90.0
    assert (plateau_diamond.kwargs["x"], plateau_diamond.kwargs["y"]) == (
        [0.0],
        [50.0],
    )
