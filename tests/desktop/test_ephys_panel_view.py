"""Tests for the desktop ephys panel view."""

from __future__ import annotations

from typing import Any

import numpy as np

from ephys_alignment_gui.desktop.ephys_panel_view import (
    DesktopEphysPanelView,
    EphysPanelPlots,
    EphysPanelStyle,
)


class FakePlot:
    def __init__(self) -> None:
        self.added: list[Any] = []
        self.removed: list[Any] = []
        self.x_ranges: list[dict[str, Any]] = []

    def addItem(self, item: Any) -> None:
        self.added.append(item)

    def removeItem(self, item: Any) -> None:
        self.removed.append(item)

    def setXRange(self, **kwargs: Any) -> None:
        self.x_ranges.append(kwargs)


class FakeSignal:
    def __init__(self) -> None:
        self.connected: list[Any] = []
        self.disconnects = 0

    def connect(self, callback: Any) -> None:
        self.connected.append(callback)

    def disconnect(self) -> None:
        self.disconnects += 1
        self.connected = []


class FakePosition:
    def __init__(self, y: float) -> None:
        self._y = y

    def y(self) -> float:
        return self._y


class FakePlotItem:
    def __init__(self) -> None:
        self.data: dict[str, Any] | None = None
        self.pen: Any = None
        self.sigClicked = FakeSignal()

    def setData(self, **kwargs: Any) -> None:
        self.data = kwargs

    def setPen(self, pen: Any) -> None:
        self.pen = pen

    def mapFromScene(self, _scene_pos: Any) -> FakePosition:
        return FakePosition(7.0)


class FakeImageItem:
    def __init__(self) -> None:
        self.image: Any = None
        self.auto_levels: Any = None
        self.transform: Any = None
        self.lookup_table: Any = None
        self.levels: Any = None

    def setImage(self, image: Any, autoLevels: Any = None) -> None:
        self.image = image
        self.auto_levels = autoLevels

    def setTransform(self, transform: Any) -> None:
        self.transform = transform

    def setLookupTable(self, lookup_table: Any) -> None:
        self.lookup_table = lookup_table

    def setLevels(self, levels: Any) -> None:
        self.levels = levels

    def mapFromScene(self, _scene_pos: Any) -> FakePosition:
        return FakePosition(9.0)


class FakeInfiniteLine:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeColorBar:
    def __init__(self, name: str) -> None:
        self.name = name

    def getColourMap(self) -> str:
        return "lookup"

    def makeColourBar(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"args": args, "kwargs": kwargs, "cmap": self.name}


class FakeTransform:
    def __init__(self, *values: Any) -> None:
        self.values = values


def _view(monkeypatch) -> tuple[DesktopEphysPanelView, dict[str, FakePlot], list[Any]]:
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.ephys_panel_view.pg.ImageItem",
        FakeImageItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.ephys_panel_view.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.ephys_panel_view.pg.PlotCurveItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.ephys_panel_view.pg.InfiniteLine",
        FakeInfiniteLine,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.ephys_panel_view.ColorBar",
        FakeColorBar,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.ephys_panel_view.QtGui.QTransform",
        FakeTransform,
    )
    plots = {
        "image": FakePlot(),
        "image_colorbar": FakePlot(),
        "line": FakePlot(),
        "probe": FakePlot(),
        "probe_colorbar": FakePlot(),
    }
    axis_calls: list[Any] = []
    return (
        DesktopEphysPanelView(
            plots=EphysPanelPlots(**plots),
            style=EphysPanelStyle(line_pen="line-pen"),
            set_axis=lambda *args, **kwargs: axis_calls.append((args, kwargs)),
            cluster_clicked=lambda *_args: None,
        ),
        plots,
        axis_calls,
    )


def test_render_image_owns_image_items_and_feature_coordinate_mapping(
    monkeypatch,
) -> None:
    view, plots, axis_calls = _view(monkeypatch)

    view.render_image(
        {
            "img": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "scale": [2.0, 3.0],
            "offset": [10.0, 20.0],
            "cmap": "viridis",
            "levels": [1.0, 4.0],
            "title": "feature",
            "xrange": (0.0, 100.0),
            "xaxis": "depth",
        }
    )

    image = view.items.image_plots[0]
    assert image in plots["image"].added
    assert image.lookup_table == "lookup"
    assert image.levels == (1.0, 4.0)
    assert image.transform.values == (2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 10.0, 20.0, 1.0)
    assert plots["image"].x_ranges == [{"min": 0.0, "max": 100.0, "padding": 0}]
    assert view.feature_xrange == (0.0, 100.0)
    assert view.feature_y_from_scene("scene") == 27.0
    assert axis_calls[-1][0] == (plots["image"], "bottom")


def test_render_scatter_connects_cluster_clicks_and_maps_cluster_index(
    monkeypatch,
) -> None:
    view, plots, _axis_calls = _view(monkeypatch)

    view.render_scatter(
        {
            "x": np.array([10.0, 20.0]),
            "y": np.array([1.0, 2.0]),
            "symbol": np.array(["o", "t"]),
            "size": np.array([4, 5]),
            "colours": np.array(["r", "b"]),
            "pen": "pen",
            "levels": [[0.0], [1.0]],
            "cmap": "magma",
            "title": "clusters",
            "xrange": (5.0, 25.0),
            "xaxis": "x",
            "cluster": True,
        }
    )

    plot = view.items.image_plots[0]
    assert plot in plots["image"].added
    assert plot.data is not None
    assert plot.data["x"].tolist() == [10.0, 20.0]
    assert plot.sigClicked.connected
    assert view.cluster_index_for_plot_x(20.0) == 1
    assert view.cluster_index_for_plot_x(30.0) is None


def test_render_line_and_probe_clear_previous_items(monkeypatch) -> None:
    view, plots, _axis_calls = _view(monkeypatch)

    view.render_line({"x": [1], "y": [2], "xrange": (0, 3), "xaxis": "depth"})
    first_line = view.items.line_plots[0]
    view.render_line({"x": [3], "y": [4], "xrange": (0, 5), "xaxis": "depth"})

    assert first_line in plots["line"].removed
    assert view.items.line_plots[0].pen == "line-pen"
    assert plots["line"].x_ranges[-1] == {"min": 0, "max": 5, "padding": 0}

    view.render_probe(
        {
            "img": [np.array([[1.0]])],
            "scale": [[1.0, 2.0]],
            "offset": [[3.0, 4.0]],
            "cmap": "cividis",
            "levels": [0.0, 1.0],
            "title": "probe",
            "xrange": (-1.0, 1.0),
        },
        bounds=[12.0],
    )

    assert len(view.items.probe_plots) == 1
    assert len(view.probe_colorbars) == 1
    assert len(view.items.probe_bounds) == 1
    assert view.items.probe_bounds[0].kwargs == {"pos": 12.0, "angle": 0, "pen": "w"}
    assert plots["probe"].x_ranges == [{"min": -1.0, "max": 1.0, "padding": 0}]


def test_clear_detaches_all_owned_items_and_resets_feature_state(monkeypatch) -> None:
    view, plots, _axis_calls = _view(monkeypatch)
    view.render_image(
        {
            "img": np.array([[1.0]]),
            "scale": [1.0, 2.0],
            "offset": [3.0, 4.0],
            "cmap": "viridis",
            "levels": [0.0, 1.0],
            "title": "feature",
            "xrange": (0.0, 1.0),
            "xaxis": "depth",
        }
    )
    image = view.items.image_plots[0]
    cbar = view.items.image_colorbars[0]

    view.clear()

    assert image in plots["image"].removed
    assert cbar in plots["image_colorbar"].removed
    assert view.items.image_plots == []
    assert view.items.image_colorbars == []
    assert view.feature_xrange is None
