"""Tests for the desktop slice panel view."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.core.alignment_read_models import (
    ActiveSliceRenderState,
    PerpendicularSliceRenderState,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.slice_display_policy import (
    SliceImageKind,
    SliceRenderDecision,
    SliceSelection,
)
from ephys_alignment_gui.desktop.displays.slice_panel_view import (
    SlicePanelPlots,
    SlicePanelStyle,
    SlicePanelView,
    SlicePanelViewState,
)


class FakePlot:
    def __init__(self) -> None:
        self.added: list[Any] = []
        self.removed: list[Any] = []
        self.x_ranges: list[dict[str, Any]] = []
        self.clear_count = 0

    def addItem(self, item: Any) -> None:
        self.added.append(item)

    def removeItem(self, item: Any) -> None:
        self.removed.append(item)

    def setXRange(self, **kwargs: Any) -> None:
        self.x_ranges.append(kwargs)

    def clear(self) -> None:
        self.clear_count += 1


class FakeLayout:
    def __init__(self) -> None:
        self.added: list[Any] = []
        self.removed: list[Any] = []

    def addItem(self, item: Any, *_args: Any) -> None:
        self.added.append(item)

    def removeItem(self, item: Any) -> None:
        self.removed.append(item)


class FakePlotItem:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.data: dict[str, Any] | None = None

    def setData(self, **kwargs) -> None:
        self.data = kwargs


class FakeImageItem:
    def __init__(self) -> None:
        self.image: Any = None
        self.transform: Any = None
        self.lookup_table: Any = None
        self.levels: Any = None

    def setImage(self, image: Any) -> None:
        self.image = image

    def setTransform(self, transform: Any) -> None:
        self.transform = transform

    def setLookupTable(self, lookup_table: Any) -> None:
        self.lookup_table = lookup_table

    def setLevels(self, levels: Any) -> None:
        self.levels = levels

    def getHistogram(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.0, 1.0]), np.array([0, 20])


class FakeSignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in list(self._callbacks):
            callback()


class FakeGradient:
    def __init__(self) -> None:
        self.color_map = None

    def setColorMap(self, color_map: Any) -> None:
        self.color_map = color_map


class FakeAxis:
    def __init__(self) -> None:
        self.hidden = False
        self.shown = False
        self.pen: Any = None
        self.text_pen: Any = None
        self.label: Any = None

    def hide(self) -> None:
        self.hidden = True

    def show(self) -> None:
        self.shown = True

    def setPen(self, pen: Any) -> None:
        self.pen = pen

    def setTextPen(self, pen: Any) -> None:
        self.text_pen = pen

    def setLabel(self, label: str) -> None:
        self.label = label


class FakeHistogramCurve:
    def __init__(self) -> None:
        self.xData: np.ndarray | None = None
        self.yData: np.ndarray | None = None

    def setData(self, x: Any, y: Any) -> None:
        self.xData = np.asarray(x)
        self.yData = np.asarray(y)


class FakeHistogramLUTItem:
    def __init__(self) -> None:
        self.axis = FakeAxis()
        self.gradient = FakeGradient()
        self.sigLevelsChanged = FakeSignal()
        self.plot = FakeHistogramCurve()
        self.plots = [self.plot]
        self.image_item = None
        self.levels = (0.0, 1.0)
        self.auto_range_count = 0

    def setImageItem(self, image_item: Any) -> None:
        self.image_item = image_item
        self.plot.setData(*image_item.getHistogram())

    def autoHistogramRange(self) -> None:
        self.auto_range_count += 1

    def setLevels(self, *, min: float, max: float) -> None:
        self.levels = (min, max)

    def getLevels(self) -> tuple[float, float]:
        return self.levels


class FakeColorBar:
    map = "fake-color-map"

    def __init__(self, name: str) -> None:
        self.name = name

    def getColourMap(self) -> str:
        return "fake-lookup-table"


def _projection() -> SimpleNamespace:
    return SimpleNamespace(
        channel_locations_ras=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
        tip_location_ras=np.array([7.0, 8.0, 9.0]),
        perpendicular_vectors=[
            np.array(
                [
                    [1.0, 0.0, 2.0],
                    [3.0, 0.0, 4.0],
                ]
            )
        ],
    )


def _slice_render_state(
    selection: SliceSelection,
    *,
    initial_levels: tuple[float, float] | None,
) -> ActiveSliceRenderState:
    return ActiveSliceRenderState(
        key=AlignmentKey("rec", "stream", 0),
        selection=selection,
        image=np.array([[1.0, 2.0], [3.0, 4.0]]),
        scale=np.array([1.0, 1.0]),
        offset=np.array([0.0, 0.0]),
        decision=SliceRenderDecision(
            kind=SliceImageKind.SCALAR,
            scalar_channel=selection.key,
            initial_levels=initial_levels,
        ),
        track_annos_and_ends_ras=np.array(
            [
                [1.0, 0.0, 2.0],
                [3.0, 0.0, 4.0],
            ]
        ),
        projection=_projection(),
    )


def _view_with_plots() -> tuple[
    SlicePanelView,
    FakePlot,
    FakePlot,
    FakeLayout,
]:
    coronal = FakePlot()
    perpendicular = FakePlot()
    layout = FakeLayout()
    view = SlicePanelView(
        plots=SlicePanelPlots(
            coronal=coronal,
            coronal_layout=layout,
            histogram_alt=None,
            perpendicular=perpendicular,
        ),
        style=SlicePanelStyle(
            dotted_pen="dot",
            solid_pen="solid",
            reference_line_pen="ref",
        ),
        histology_exists=lambda: True,
        view_state=SlicePanelViewState(),
    )
    return view, coronal, perpendicular, layout


def test_slice_panel_owns_channel_overlay_handles(monkeypatch) -> None:
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.PlotCurveItem",
        FakePlotItem,
    )
    view, coronal, perpendicular, _layout = _view_with_plots()
    projection = _projection()

    view.plot_channels(projection)

    state = view.view_state
    assert state.channel_projection is projection
    assert state.channel_status
    assert isinstance(state.slice_chns, FakePlotItem)
    assert isinstance(state.slice_tip, FakePlotItem)
    assert len(state.slice_lines) == 1
    assert state.slice_chns in coronal.added
    assert state.slice_tip in coronal.added
    assert state.slice_lines[0] in coronal.added

    view.toggle_channel_visibility()

    assert not state.channel_status
    assert state.slice_chns in coronal.removed
    assert state.slice_tip in coronal.removed
    assert state.slice_lines[0] in coronal.removed
    assert perpendicular.removed == []

    view.toggle_channel_visibility()

    assert state.channel_status
    assert state.slice_chns in coronal.added
    assert state.slice_tip in coronal.added
    assert state.slice_lines[0] in coronal.added


def test_slice_panel_owns_export_trajectory_handle(monkeypatch) -> None:
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.PlotCurveItem",
        FakePlotItem,
    )
    view, coronal, _perpendicular, _layout = _view_with_plots()
    view.view_state.channel_projection = _projection()

    view.render_export_trajectory_overlay("export-pen")

    state = view.view_state
    assert isinstance(state.traj_line, FakePlotItem)
    assert state.traj_line in coronal.added
    assert state.traj_line.data is not None
    np.testing.assert_array_equal(state.traj_line.data["x"], [1.0, 4.0])
    np.testing.assert_array_equal(state.traj_line.data["y"], [3.0, 6.0])
    assert state.traj_line.data["pen"] == "export-pen"


def test_slice_panel_owns_perpendicular_overlay_handles(monkeypatch) -> None:
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.ImageItem",
        FakeImageItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.InfiniteLine",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.ColorBar",
        FakeColorBar,
    )
    view, _coronal, perpendicular, _layout = _view_with_plots()
    view.view_state.slice_hist_levels = (5.0, 95.0)

    view.render_perpendicular_histology(
        PerpendicularSliceRenderState(
            key=AlignmentKey("rec", "stream", 0),
            channel_name="histology_registration",
            image=np.array([[1.0, 2.0], [3.0, 4.0]]),
            extent_um=100.0,
            feature_min_um=10.0,
            feature_max_um=30.0,
            n_perp_samples=2,
            n_depths=2,
            channel_depths_um=np.array([10.0, 30.0]),
        )
    )

    state = view.view_state
    assert isinstance(state.perp_image_item, FakeImageItem)
    assert state.perp_image_item in perpendicular.added
    assert state.perp_image_item.lookup_table == "fake-lookup-table"
    assert state.perp_image_item.levels == (5.0, 95.0)
    assert isinstance(state.perp_probe_line, FakePlotItem)
    assert isinstance(state.perp_channel_dots, FakePlotItem)
    assert isinstance(state.perp_tip_marker, FakePlotItem)
    assert state.perp_probe_line in perpendicular.added
    assert state.perp_channel_dots in perpendicular.added
    assert state.perp_tip_marker in perpendicular.added
    assert perpendicular.x_ranges == [{"min": -100.0, "max": 100.0, "padding": 0}]


def test_slice_panel_clear_perpendicular_only_removes_owned_handles() -> None:
    view, _coronal, perpendicular, _layout = _view_with_plots()
    state = view.view_state
    external_reference_line = object()
    perp_image_item = object()
    perp_probe_line = object()
    perp_channel_dots = object()
    perp_tip_marker = object()
    state.perp_image_item = perp_image_item
    state.perp_probe_line = perp_probe_line
    state.perp_channel_dots = perp_channel_dots
    state.perp_tip_marker = perp_tip_marker

    view.clear_perpendicular()

    assert external_reference_line not in perpendicular.removed
    assert perpendicular.clear_count == 0
    assert perpendicular.removed == [
        perp_image_item,
        perp_probe_line,
        perp_channel_dots,
        perp_tip_marker,
    ]
    assert state.perp_image_item is None
    assert state.perp_probe_line is None
    assert state.perp_channel_dots is None
    assert state.perp_tip_marker is None


def test_slice_panel_preserves_scalar_levels_per_slice_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.ImageItem",
        FakeImageItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.HistogramLUTItem",
        FakeHistogramLUTItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.PlotCurveItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.ColorBar",
        FakeColorBar,
    )
    view, _coronal, _perpendicular, _layout = _view_with_plots()
    histology = SliceSelection("slice_data", "histology_registration")
    fluorescence = SliceSelection("slice_data", "Ex_561_Em_600")

    view.render_slice(_slice_render_state(histology, initial_levels=(5.0, 95.0)))
    histology_histogram = view.view_state.histogram_item
    assert histology_histogram.getLevels() == (5.0, 95.0)

    view.view_state.perp_image_item = FakeImageItem()
    histology_histogram.setLevels(min=20.0, max=80.0)
    histology_histogram.sigLevelsChanged.emit()

    view.render_slice(_slice_render_state(fluorescence, initial_levels=(1.0, 9.0)))
    assert view.view_state.histogram_item.getLevels() == (1.0, 9.0)

    view.render_slice(_slice_render_state(histology, initial_levels=(5.0, 95.0)))

    assert view.view_state.histogram_item.getLevels() == (20.0, 80.0)
    assert view.view_state.slice_hist_levels == (20.0, 80.0)
    assert view.view_state.slice_levels_by_selection[histology] == (20.0, 80.0)


def test_slice_panel_displays_lut_histogram_counts_on_log_scale(monkeypatch) -> None:
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.ImageItem",
        FakeImageItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.HistogramLUTItem",
        FakeHistogramLUTItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.PlotCurveItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.displays.slice_panel_view.ColorBar",
        FakeColorBar,
    )
    view, _coronal, _perpendicular, _layout = _view_with_plots()

    view.render_slice(
        _slice_render_state(
            SliceSelection("slice_data", "histology_registration"),
            initial_levels=(5.0, 95.0),
        )
    )

    histogram = view.view_state.histogram_item
    assert histogram.axis.shown
    assert not histogram.axis.hidden
    assert histogram.axis.pen == "k"
    assert histogram.axis.text_pen == "k"
    assert histogram.axis.label == "intensity (a.u.)"
    np.testing.assert_array_equal(histogram.plot.xData, [0.0, 1.0])
    np.testing.assert_allclose(histogram.plot.yData, np.log1p([0, 20]))
    assert histogram.getLevels() == (5.0, 95.0)


def test_slice_panel_clear_resets_owned_plots_and_handles() -> None:
    view, coronal, perpendicular, layout = _view_with_plots()
    state = view.view_state
    slice_item = object()
    state.channel_projection = object()
    state.slice_lines = [object()]
    state.slice_chns = object()
    state.slice_tip = object()
    state.traj_line = object()
    perp_image_item = object()
    perp_probe_line = object()
    perp_channel_dots = object()
    perp_tip_marker = object()
    state.perp_image_item = perp_image_item
    state.perp_probe_line = perp_probe_line
    state.perp_channel_dots = perp_channel_dots
    state.perp_tip_marker = perp_tip_marker
    state.slice_color_bar = object()
    state.slice_hist_levels = (1.0, 2.0)
    state.active_slice_selection = object()
    state.slice_levels_by_selection[object()] = (1.0, 2.0)
    state.slice_item = slice_item
    state.histogram_item = object()

    view.clear()

    assert coronal.clear_count == 1
    assert perpendicular.clear_count == 0
    assert perp_image_item in perpendicular.removed
    assert perp_probe_line in perpendicular.removed
    assert perp_channel_dots in perpendicular.removed
    assert perp_tip_marker in perpendicular.removed
    assert layout.removed == [slice_item]
    assert state.channel_projection is None
    assert state.slice_lines == []
    assert state.slice_chns is None
    assert state.slice_tip is None
    assert state.traj_line is None
    assert state.perp_image_item is None
    assert state.perp_probe_line is None
    assert state.perp_channel_dots is None
    assert state.perp_tip_marker is None
    assert state.slice_color_bar is None
    assert state.slice_hist_levels is None
    assert state.active_slice_selection is None
    assert state.slice_levels_by_selection == {}
    assert state.slice_item is None
    assert state.histogram_item is None
