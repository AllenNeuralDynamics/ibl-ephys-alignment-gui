"""Tests for the desktop ephys panel view."""

from __future__ import annotations

from typing import Any

import numpy as np

from ephys_alignment_gui.desktop.displays.ephys_panel_view import (
    DesktopEphysPanelView,
    EphysPanelPlots,
    EphysPanelStyle,
    EphysPanelWidgets,
)


class FakePlot:
    def __init__(self, *, width: float = 900.0, height: float = 600.0) -> None:
        self.added: list[Any] = []
        self.removed: list[Any] = []
        self.x_ranges: list[dict[str, Any]] = []
        self.y_ranges: list[dict[str, Any]] = []
        self._width = width
        self._height = height
        self.axes = {
            "top": FakeConfiguredAxis(),
            "bottom": FakeConfiguredAxis(),
            "left": FakeConfiguredAxis(),
            "right": FakeConfiguredAxis(),
        }

    def addItem(self, item: Any) -> None:
        self.added.append(item)

    def removeItem(self, item: Any) -> None:
        self.removed.append(item)

    def setXRange(self, **kwargs: Any) -> None:
        self.x_ranges.append(kwargs)

    def setYRange(self, **kwargs: Any) -> None:
        self.y_ranges.append(kwargs)

    def getAxis(self, orientation: str) -> Any:
        return self.axes[orientation]

    def width(self) -> float:
        return self._width

    def height(self) -> float:
        return self._height


class FakeConfiguredAxis:
    def __init__(self) -> None:
        self.height: Any = None
        self.ticks: Any = None
        self.label: Any = None
        self.pen: Any = None
        self.text_pen: Any = None
        self.visible = False

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def setPen(self, pen: Any) -> None:
        self.pen = pen

    def setTextPen(self, pen: Any) -> None:
        self.text_pen = pen

    def setLabel(self, label: str = "", **_kwargs: Any) -> None:
        self.label = label

    def setHeight(self, height: Any) -> None:
        self.height = height

    def setTicks(self, ticks: Any) -> None:
        self.ticks = ticks


class FakeAxis:
    def __init__(self, width: float = 100.0) -> None:
        self._width = width

    def width(self) -> float:
        return self._width


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

    def getBrush(self, data: Any, levels: Any = None) -> list[Any]:
        return [("brush", float(value), tuple(levels)) for value in data]

    def makeColourBar(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"args": args, "kwargs": kwargs, "cmap": self.name}


class FakeTransform:
    def __init__(self, *values: Any) -> None:
        self.values = values


def _view(monkeypatch) -> tuple[DesktopEphysPanelView, dict[str, FakePlot], list[Any]]:
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.ephys_panel_view.pg.ImageItem",
        FakeImageItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.ephys_panel_view.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.ephys_panel_view.pg.PlotCurveItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.ephys_panel_view.pg.InfiniteLine",
        FakeInfiniteLine,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.ephys_panel_view.ColorBar",
        FakeColorBar,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.ephys_panel_view.QtGui.QTransform",
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

    def set_axis(fig: Any, orientation: str, **kwargs: Any) -> Any:
        axis_calls.append(((fig, orientation), kwargs))
        axis = fig.getAxis(orientation)
        if kwargs.get("show", True):
            axis.show()
            axis.setPen(kwargs.get("pen", "k"))
            axis.setTextPen(kwargs.get("pen", "k"))
            axis.setLabel(kwargs.get("label") or "")
        else:
            axis.hide()
        return axis

    return (
        DesktopEphysPanelView(
            plots=EphysPanelPlots(**plots),
            widgets=EphysPanelWidgets(
                area=object(),
                graphics_layout=object(),
                image_axis=FakeAxis(),
            ),
            style=EphysPanelStyle(
                line_pen="line-pen",
                depth_guide_pen="depth-guide-pen",
            ),
            set_axis=set_axis,
            cluster_clicked=lambda *_args: None,
        ),
        plots,
        axis_calls,
    )


def test_image_raster_request_uses_visible_plot_area(monkeypatch) -> None:
    view, _plots, _axis_calls = _view(monkeypatch)

    request = view.image_raster_request()

    assert request.max_time_bins == 800
    assert request.max_depth_bins == 600


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
    cbar = view.items.image_colorbars[0]
    assert cbar["kwargs"]["label"] == "feature"
    assert cbar["kwargs"]["axis_height"] == 42


def test_render_image_overlays_no_data_mask(monkeypatch) -> None:
    view, plots, _axis_calls = _view(monkeypatch)

    view.render_image(
        {
            "img": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "scale": [2.0, 3.0],
            "offset": [10.0, 20.0],
            "cmap": "viridis",
            "levels": [1.0, 6.0],
            "title": "feature",
            "xrange": (0.0, 100.0),
            "xaxis": "depth",
            "no_data_mask": np.array(
                [
                    [False, True, False],
                    [False, True, True],
                ]
            ),
            "no_data_color": (145, 158, 170, 210),
        }
    )

    assert len(view.items.image_plots) == 2
    image, overlay = view.items.image_plots
    assert image in plots["image"].added
    assert overlay in plots["image"].added
    assert overlay.auto_levels is False
    assert overlay.transform.values == image.transform.values
    assert overlay.image.shape == (2, 3, 4)
    np.testing.assert_array_equal(overlay.image[0, 1], [145, 158, 170, 210])
    np.testing.assert_array_equal(overlay.image[0, 0], [0, 0, 0, 0])


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


def test_render_scatter_maps_numeric_colours_through_color_bar(monkeypatch) -> None:
    view, _plots, _axis_calls = _view(monkeypatch)

    view.render_scatter(
        {
            "x": np.array([10.0, 20.0]),
            "y": np.array([1.0, 2.0]),
            "symbol": np.array(["o", "o"]),
            "size": np.array([4, 4]),
            "colours": np.array([5.0, 25.0]),
            "pen": "pen",
            "levels": np.array([0.0, 30.0]),
            "cmap": "magma",
            "title": "clusters",
            "xrange": (5.0, 25.0),
            "xaxis": "x",
            "cluster": True,
        }
    )

    plot = view.items.image_plots[0]
    assert plot.data is not None
    assert plot.data["brush"] == [
        ("brush", 5.0, (0.0, 30.0)),
        ("brush", 25.0, (0.0, 30.0)),
    ]


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
    assert view.probe_colorbars[0]["kwargs"]["label"] == "probe"
    assert view.probe_colorbars[0]["kwargs"]["axis_height"] == 42
    assert view.probe_colorbars[0]["kwargs"]["edge_tick_padding"] == 1.0
    assert len(view.items.probe_bounds) == 1
    assert view.items.probe_bounds[0].kwargs == {"pos": 12.0, "angle": 0, "pen": "w"}
    assert plots["probe"].x_ranges == [{"min": -1.0, "max": 1.0, "padding": 0}]


def test_render_phase_image_configures_full_colorbar(monkeypatch) -> None:
    view, plots, axis_calls = _view(monkeypatch)

    view.render_image(
        {
            "img": np.zeros((4, 4, 4), dtype=np.uint8),
            "scale": [2.0, 3.0],
            "offset": [10.0, 20.0],
            "cmap": None,
            "levels": None,
            "title": "LFP coherency phase (theta)",
            "xrange": (0.0, 100.0),
            "xaxis": "Distance from probe tip (um)",
        }
    )

    legend = view.items.image_colorbars[0]
    assert legend in plots["image_colorbar"].added
    assert legend.image.shape == (256, 20, 3)
    assert legend.transform.values == (
        1.0 / 256,
        0.0,
        0.0,
        0.0,
        0.1,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert plots["image_colorbar"].x_ranges[-1] == {
        "min": 0.0,
        "max": 1.0,
        "padding": 0,
    }
    assert plots["image_colorbar"].y_ranges[-1] == {
        "min": 0.0,
        "max": 2.0,
        "padding": 0,
    }
    assert (
        (plots["image_colorbar"], "top"),
        {"pen": "k", "label": "phase (rad) / coherence"},
    ) in axis_calls
    top_axis = plots["image_colorbar"].axes["top"]
    assert top_axis.height == 52
    assert top_axis.ticks == [
        [(0.0, "0"), (0.5, "pi"), (1.0, "2pi")],
        [(0.0, "coh 0"), (1.0, "coh 1")],
    ]


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
