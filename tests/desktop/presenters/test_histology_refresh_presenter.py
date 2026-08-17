"""Tests for loaded-shank histology refresh presentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop.presenters.histology_refresh_presenter import (
    DesktopHistologyRefreshPresenter,
)


class FakeQueries:
    def __init__(
        self,
        line_state: Any = None,
        *,
        reference_lines_visible: bool = True,
    ) -> None:
        self.line_state = line_state
        self._reference_lines_visible = reference_lines_visible
        self.line_calls: list[int] = []
        self.workspace = SimpleNamespace(
            active_shank_selection=self.active_shank_selection,
            active_reference_line_state=self.active_reference_line_state,
            reference_lines_visible=self.reference_lines_visible,
        )

    def active_shank_selection(self) -> Any:
        return SimpleNamespace(shank_idx=2)

    def active_reference_line_state(self, shank_idx: int) -> Any:
        self.line_calls.append(shank_idx)
        return self.line_state

    def reference_lines_visible(self) -> bool:
        return self._reference_lines_visible


class FakeHistologyPresenter:
    def __init__(self, render_result: bool = True) -> None:
        self.render_result = render_result
        self.render_count = 0

    def render_active_panels(self) -> bool:
        self.render_count += 1
        return self.render_result


class FakeSlicePanelPresenter:
    def __init__(self) -> None:
        self.refreshed_selections: list[Any] = []

    def refresh_perpendicular_histology(self, selection: Any) -> None:
        self.refreshed_selections.append(selection)


class FakeSliceMenuCoordinator:
    def __init__(self) -> None:
        self.selection = "slice-selection"

    def current_selection(self) -> Any:
        return self.selection


class FakeReferenceLineDisplay:
    def __init__(self) -> None:
        self.created: list[tuple[Any, Any]] = []
        self.remove_count = 0

    def create_lines(self, positions: Any, track_positions: Any = None) -> None:
        self.created.append((positions, track_positions))

    def remove_from_plots(self) -> None:
        self.remove_count += 1


def _presenter(
    *,
    line_state: Any = None,
    render_result: bool = True,
    reference_lines_visible: bool = True,
) -> tuple[
    DesktopHistologyRefreshPresenter,
    FakeQueries,
    FakeHistologyPresenter,
    FakeSlicePanelPresenter,
    FakeReferenceLineDisplay,
]:
    queries = FakeQueries(
        line_state,
        reference_lines_visible=reference_lines_visible,
    )
    histology = FakeHistologyPresenter(render_result)
    slice_panel = FakeSlicePanelPresenter()
    slice_menu = FakeSliceMenuCoordinator()
    reference_lines = FakeReferenceLineDisplay()
    return (
        DesktopHistologyRefreshPresenter(
            app=SimpleNamespace(queries=queries),
            histology_presenter=histology,
            slice_panel_presenter=slice_panel,
            slice_menu_coordinator=slice_menu,
            reference_line_display=reference_lines,
        ),
        queries,
        histology,
        slice_panel,
        reference_lines,
    )


def test_render_loaded_shank_histology_restores_reference_lines() -> None:
    line_state = SimpleNamespace(
        feature_positions_um=[1.0],
        track_positions_um=[2.0],
    )
    presenter, queries, histology, slice_panel, reference_lines = _presenter(
        line_state=line_state
    )

    assert presenter.render_loaded_shank_histology(1)

    assert histology.render_count == 1
    assert slice_panel.refreshed_selections == ["slice-selection"]
    assert queries.line_calls == [1]
    assert reference_lines.created == [([1.0], [2.0])]
    assert reference_lines.remove_count == 0


def test_render_loaded_shank_histology_preserves_hidden_reference_lines() -> None:
    line_state = SimpleNamespace(
        feature_positions_um=[1.0],
        track_positions_um=[2.0],
    )
    presenter, _queries, _histology, _slice_panel, reference_lines = _presenter(
        line_state=line_state,
        reference_lines_visible=False,
    )

    assert presenter.render_loaded_shank_histology(1)

    assert reference_lines.created == [([1.0], [2.0])]
    assert reference_lines.remove_count == 1


def test_render_loaded_shank_histology_uses_active_shank_by_default() -> None:
    presenter, queries, _histology, _slice_display, _reference_lines = _presenter()

    assert presenter.render_loaded_shank_histology()

    assert queries.line_calls == [2]


def test_render_loaded_shank_histology_stops_when_panel_render_fails() -> None:
    presenter, queries, histology, slice_panel, reference_lines = _presenter(
        render_result=False
    )

    assert not presenter.render_loaded_shank_histology(1)

    assert histology.render_count == 1
    assert slice_panel.refreshed_selections == []
    assert queries.line_calls == []
    assert reference_lines.created == []
