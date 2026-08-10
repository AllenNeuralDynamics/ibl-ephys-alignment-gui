"""Tests for the desktop slice panel presenter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.alignment_read_models import (
    PerpendicularSliceRenderState,
)
from ephys_alignment_gui.desktop.slice_panel_presenter import (
    SlicePanelPlots,
    SlicePanelPresenter,
    SlicePanelStyle,
    SlicePanelView,
    SlicePanelViewState,
)
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.slice_display_policy import SliceSelection


class FakeAction:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def data(self) -> Any:
        return self._payload


class FakeActionGroup:
    def __init__(self, actions: list[FakeAction], checked: FakeAction | None) -> None:
        self._actions = actions
        self._checked = checked

    def checkedAction(self) -> FakeAction | None:
        return self._checked

    def actions(self) -> list[FakeAction]:
        return self._actions


class FakeQueries:
    def __init__(self) -> None:
        self.rendered_selections: list[SliceSelection] = []
        self.slices = SimpleNamespace(
            active_slice_render_state=self.active_slice_render_state,
        )

    def active_slice_render_state(self, selection: SliceSelection) -> Any:
        self.rendered_selections.append(selection)
        return SimpleNamespace(scalar_channel=selection.key)


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
        self.removed: list[Any] = []

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


def _presenter(
    queries: FakeQueries,
    action_group: FakeActionGroup | None = None,
) -> SlicePanelPresenter:
    view = SlicePanelView(
        plots=SlicePanelPlots(
            coronal=None,
            coronal_layout=None,
            histogram_alt=None,
            perpendicular=None,
        ),
        style=SlicePanelStyle(
            dotted_pen=None,
            solid_pen=None,
            reference_line_pen=None,
        ),
        histology_exists=lambda: True,
    )
    return SlicePanelPresenter(
        app=SimpleNamespace(queries=queries),
        view=view,
        action_group_provider=lambda: action_group,
    )


def _presenter_with_plots() -> tuple[
    SlicePanelPresenter,
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
    return (
        SlicePanelPresenter(
            app=SimpleNamespace(queries=FakeQueries()),
            view=view,
            action_group_provider=lambda: None,
        ),
        view,
        coronal,
        perpendicular,
        layout,
    )


def test_slice_panel_reads_current_selection_from_action_group() -> None:
    selection = SliceSelection("slice_data", "histology_registration")
    checked = FakeAction(selection.to_payload())
    other = FakeAction(SliceSelection("slice_data", "ccf").to_payload())
    action_group = FakeActionGroup([other, checked], checked)
    queries = FakeQueries()
    presenter = _presenter(queries, action_group)

    assert presenter.current_slice_selection() == selection
    assert presenter.action_for_selection(selection) is checked
    assert presenter.current_scalar_slice_channel() == "histology_registration"
    assert queries.rendered_selections == [selection]


def test_slice_panel_plot_selection_queries_render_state() -> None:
    selection = SliceSelection("slice_data", "ccf")
    queries = FakeQueries()
    presenter = _presenter(queries)
    calls: list[Any] = []
    presenter.render_slice = calls.append

    presenter.plot_slice_selection(selection)

    assert queries.rendered_selections == [selection]
    assert calls == [SimpleNamespace(scalar_channel="ccf")]


def test_slice_panel_owns_channel_overlay_handles(monkeypatch) -> None:
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.slice_panel_presenter.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.slice_panel_presenter.pg.PlotCurveItem",
        FakePlotItem,
    )
    _presenter, view, coronal, perpendicular, _layout = _presenter_with_plots()
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
        "ephys_alignment_gui.desktop.slice_panel_presenter.pg.PlotCurveItem",
        FakePlotItem,
    )
    _presenter, view, coronal, _perpendicular, _layout = _presenter_with_plots()
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
        "ephys_alignment_gui.desktop.slice_panel_presenter.pg.ImageItem",
        FakeImageItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.slice_panel_presenter.pg.InfiniteLine",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.slice_panel_presenter.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.desktop.slice_panel_presenter.ColorBar",
        FakeColorBar,
    )
    _presenter, view, _coronal, perpendicular, _layout = _presenter_with_plots()
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


def test_slice_panel_clear_resets_owned_plots_and_handles() -> None:
    _presenter, view, coronal, perpendicular, layout = _presenter_with_plots()
    state = view.view_state
    slice_item = object()
    state.channel_projection = object()
    state.slice_lines = [object()]
    state.slice_chns = object()
    state.slice_tip = object()
    state.traj_line = object()
    state.perp_image_item = object()
    state.perp_probe_line = object()
    state.perp_channel_dots = object()
    state.perp_tip_marker = object()
    state.slice_color_bar = object()
    state.slice_hist_levels = (1.0, 2.0)
    state.slice_item = slice_item
    state.histogram_item = object()

    view.clear()

    assert coronal.clear_count == 1
    assert perpendicular.clear_count == 1
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
    assert state.slice_item is None
    assert state.histogram_item is None
