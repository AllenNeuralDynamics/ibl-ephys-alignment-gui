"""Tests for slice-panel presenter query/render choreography."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.core.slice_display_policy import SliceSelection
from ephys_alignment_gui.desktop.presenters.slice_panel_presenter import (
    SlicePanelPresenter,
)


class FakeSliceQueries:
    def __init__(self) -> None:
        self.slice_state: Any = None
        self.perpendicular_state: Any = None
        self.slice_selections: list[SliceSelection] = []
        self.perpendicular_channels: list[str] = []

    def active_slice_render_state(self, selection: SliceSelection) -> Any:
        self.slice_selections.append(selection)
        return self.slice_state

    def active_perpendicular_slice_state(self, channel_name: str) -> Any:
        self.perpendicular_channels.append(channel_name)
        return self.perpendicular_state


class FakeView:
    def __init__(self) -> None:
        self.histology_loaded = True
        self.cleared = False
        self.perpendicular_clear_count = 0
        self.rendered_slices: list[Any] = []
        self.rendered_perpendicular: list[Any] = []
        self.updated_levels = False
        self.plotted_projections: list[Any] = []
        self.toggled_channels = False
        self.export_calls: list[tuple[Any, Any]] = []
        self.current_locations: Any | None = None
        self.stored_projection: Any | None = None

    def histology_exists(self) -> bool:
        return self.histology_loaded

    def clear(self) -> None:
        self.cleared = True

    def clear_perpendicular(self) -> None:
        self.perpendicular_clear_count += 1

    def render_slice(self, render_state: Any) -> None:
        self.rendered_slices.append(render_state)

    def render_perpendicular_histology(self, render_state: Any) -> None:
        self.rendered_perpendicular.append(render_state)

    def update_perpendicular_levels(self) -> None:
        self.updated_levels = True

    def plot_channels(self, projection: Any) -> None:
        self.plotted_projections.append(projection)

    def toggle_channel_visibility(self) -> None:
        self.toggled_channels = True

    def render_export_trajectory_overlay(
        self,
        pen: Any,
        *,
        channel_locations_ras: Any | None = None,
    ) -> None:
        self.export_calls.append((pen, channel_locations_ras))

    def current_channel_locations_ras(self) -> Any | None:
        return self.current_locations

    def set_channel_projection(self, projection: Any) -> None:
        self.stored_projection = projection


def _selection(key: str = "histology_registration") -> SliceSelection:
    return SliceSelection("slice_data", key)


def _presenter(
    queries: FakeSliceQueries,
    view: FakeView | None = None,
) -> tuple[SlicePanelPresenter, FakeView]:
    view = view or FakeView()
    app = SimpleNamespace(queries=SimpleNamespace(slices=queries))
    return SlicePanelPresenter(app=app, view=view), view


def test_slice_panel_presenter_renders_selection_from_query() -> None:
    selection = _selection("ccf")
    queries = FakeSliceQueries()
    render_state = SimpleNamespace(scalar_channel=None, projection="projection")
    queries.slice_state = render_state
    presenter, view = _presenter(queries)

    presenter.render_slice_selection(selection)

    assert queries.slice_selections == [selection]
    assert view.rendered_slices == [render_state]
    assert queries.perpendicular_channels == []
    assert view.rendered_perpendicular == []


def test_scalar_slice_render_refreshes_perpendicular_after_coronal_render() -> None:
    selection = _selection()
    queries = FakeSliceQueries()
    render_state = SimpleNamespace(
        scalar_channel="histology_registration",
        projection="projection",
    )
    queries.slice_state = render_state
    queries.perpendicular_state = "perpendicular-state"
    presenter, view = _presenter(queries)

    presenter.render_slice_selection(selection)

    assert view.rendered_slices == [render_state]
    assert view.perpendicular_clear_count == 1
    assert queries.perpendicular_channels == ["histology_registration"]
    assert view.rendered_perpendicular == ["perpendicular-state"]


def test_refresh_perpendicular_uses_current_selection_scalar_channel() -> None:
    selection = _selection("Ex_561_Em_600")
    queries = FakeSliceQueries()
    queries.slice_state = SimpleNamespace(
        scalar_channel="Ex_561_Em_600",
        projection="projection",
    )
    queries.perpendicular_state = "perpendicular-state"
    presenter, view = _presenter(queries)

    presenter.refresh_perpendicular_histology(selection)

    assert queries.slice_selections == [selection]
    assert queries.perpendicular_channels == ["Ex_561_Em_600"]
    assert view.perpendicular_clear_count == 1
    assert view.rendered_perpendicular == ["perpendicular-state"]


def test_refresh_perpendicular_ignores_non_scalar_selection() -> None:
    queries = FakeSliceQueries()
    queries.slice_state = SimpleNamespace(scalar_channel=None, projection="projection")
    presenter, view = _presenter(queries)

    presenter.refresh_perpendicular_histology(_selection("ccf"))

    assert queries.perpendicular_channels == []
    assert view.perpendicular_clear_count == 0
    assert view.rendered_perpendicular == []


def test_plot_channels_queries_projection_when_not_provided() -> None:
    selection = _selection()
    projection = SimpleNamespace(channel_locations_ras=np.array([[1.0, 2.0, 3.0]]))
    queries = FakeSliceQueries()
    queries.slice_state = SimpleNamespace(
        scalar_channel="histology_registration",
        projection=projection,
    )
    presenter, view = _presenter(queries)

    presenter.plot_channels(selection=selection)

    assert queries.slice_selections == [selection]
    assert view.plotted_projections == [projection]


def test_export_trajectory_uses_cached_or_queried_channel_locations() -> None:
    selection = _selection()
    projection = SimpleNamespace(channel_locations_ras=np.array([[1.0, 2.0, 3.0]]))
    queries = FakeSliceQueries()
    queries.slice_state = SimpleNamespace(
        scalar_channel="histology_registration",
        projection=projection,
    )
    presenter, view = _presenter(queries)

    presenter.render_export_trajectory_overlay("pen", selection=selection)

    assert view.stored_projection is projection
    assert view.export_calls == [("pen", projection.channel_locations_ras)]

    view.current_locations = np.array([[4.0, 5.0, 6.0]])
    presenter.render_export_trajectory_overlay("pen2", selection=selection)

    assert len(queries.slice_selections) == 1
    np.testing.assert_array_equal(view.export_calls[-1][1], view.current_locations)
