"""Tests for the desktop slice panel presenter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.alignment_read_models import (
    ActiveSliceDataState,
    PerpendicularSliceRenderState,
)
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.slice_display_policy import SliceSelection
from ephys_alignment_gui.slice_panel_presenter import (
    SlicePanelPlots,
    SlicePanelPresenter,
    SlicePanelStyle,
    SlicePanelViewState,
)


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
    def __init__(self, slice_state: ActiveSliceDataState | None = None) -> None:
        self.slice_state = slice_state
        self.rendered_selections: list[SliceSelection] = []

    def active_slice_render_state(self, selection: SliceSelection) -> Any:
        self.rendered_selections.append(selection)
        return SimpleNamespace(scalar_channel=selection.key)

    def active_slice_data_state(self) -> ActiveSliceDataState | None:
        return self.slice_state


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
    return SlicePanelPresenter(
        app=SimpleNamespace(queries=queries),
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
        session_provider=lambda: SimpleNamespace(),
        histology_exists=lambda: True,
        action_group_provider=lambda: action_group,
    )


def _presenter_with_plots(
    session: Any,
) -> tuple[SlicePanelPresenter, FakePlot, FakePlot]:
    coronal = FakePlot()
    perpendicular = FakePlot()
    return (
        SlicePanelPresenter(
            app=SimpleNamespace(queries=FakeQueries()),
            plots=SlicePanelPlots(
                coronal=coronal,
                coronal_layout=SimpleNamespace(removeItem=lambda item: None),
                histogram_alt=None,
                perpendicular=perpendicular,
            ),
            style=SlicePanelStyle(
                dotted_pen="dot",
                solid_pen="solid",
                reference_line_pen="ref",
            ),
            session_provider=lambda: session,
            histology_exists=lambda: True,
            action_group_provider=lambda: None,
            view_state=SlicePanelViewState(),
        ),
        coronal,
        perpendicular,
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


def test_slice_panel_maps_legacy_slice_payload_by_identity() -> None:
    slice_data = {"ccf": np.array([[1.0]])}
    fp_slice_data = {"label": np.zeros((1, 1, 3))}
    queries = FakeQueries(
        ActiveSliceDataState(
            key=AlignmentKey("rec", "stream", 0),
            slice_data=slice_data,
            fp_slice_data=fp_slice_data,
        )
    )
    presenter = _presenter(queries)
    calls: list[SliceSelection] = []
    presenter.plot_slice_selection = calls.append

    presenter.plot_slice(slice_data, "ccf")
    presenter.plot_slice(fp_slice_data, "label")
    presenter.plot_slice({"ccf": np.array([[1.0]])}, "ccf")

    assert calls == [
        SliceSelection("slice_data", "ccf"),
        SliceSelection("fp_slice_data", "label"),
    ]


def test_slice_panel_owns_channel_overlay_handles(monkeypatch) -> None:
    monkeypatch.setattr(
        "ephys_alignment_gui.slice_panel_presenter.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.slice_panel_presenter.pg.PlotCurveItem",
        FakePlotItem,
    )
    session = SimpleNamespace()
    presenter, coronal, perpendicular = _presenter_with_plots(session)
    projection = _projection()

    presenter.plot_channels(projection)

    state = presenter.view_state
    assert state.channel_status
    assert isinstance(state.slice_chns, FakePlotItem)
    assert isinstance(state.slice_tip, FakePlotItem)
    assert len(state.slice_lines) == 1
    assert state.slice_chns in coronal.added
    assert state.slice_tip in coronal.added
    assert state.slice_lines[0] in coronal.added
    assert not hasattr(session, "slice_chns")
    assert not hasattr(session, "slice_lines")
    assert not hasattr(session, "channel_status")

    presenter.toggle_channel_visibility()

    assert not state.channel_status
    assert state.slice_chns in coronal.removed
    assert state.slice_tip in coronal.removed
    assert state.slice_lines[0] in coronal.removed
    assert perpendicular.removed == []

    presenter.toggle_channel_visibility()

    assert state.channel_status
    assert state.slice_chns in coronal.added
    assert state.slice_tip in coronal.added
    assert state.slice_lines[0] in coronal.added


def test_slice_panel_owns_export_trajectory_handle(monkeypatch) -> None:
    monkeypatch.setattr(
        "ephys_alignment_gui.slice_panel_presenter.pg.PlotCurveItem",
        FakePlotItem,
    )
    session = SimpleNamespace(
        channel_locations_ras=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    )
    presenter, coronal, _perpendicular = _presenter_with_plots(session)

    presenter.render_export_trajectory_overlay("export-pen")

    state = presenter.view_state
    assert isinstance(state.traj_line, FakePlotItem)
    assert state.traj_line in coronal.added
    assert state.traj_line.data is not None
    np.testing.assert_array_equal(state.traj_line.data["x"], [1.0, 4.0])
    np.testing.assert_array_equal(state.traj_line.data["y"], [3.0, 6.0])
    assert state.traj_line.data["pen"] == "export-pen"
    assert not hasattr(session, "traj_line")


def test_slice_panel_owns_perpendicular_overlay_handles(monkeypatch) -> None:
    monkeypatch.setattr(
        "ephys_alignment_gui.slice_panel_presenter.pg.ImageItem",
        FakeImageItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.slice_panel_presenter.pg.InfiniteLine",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.slice_panel_presenter.pg.ScatterPlotItem",
        FakePlotItem,
    )
    monkeypatch.setattr(
        "ephys_alignment_gui.slice_panel_presenter.ColorBar",
        FakeColorBar,
    )
    session = SimpleNamespace()
    presenter, _coronal, perpendicular = _presenter_with_plots(session)
    presenter.view_state.slice_hist_levels = (5.0, 95.0)

    presenter.render_perpendicular_histology(
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

    state = presenter.view_state
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
    assert not hasattr(session, "perp_image_item")
    assert not hasattr(session, "perp_probe_line")
    assert not hasattr(session, "perp_channel_dots")
    assert not hasattr(session, "perp_tip_marker")
