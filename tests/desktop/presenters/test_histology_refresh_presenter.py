"""Tests for loaded-shank histology refresh presentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop.presenters.histology_refresh_presenter import (
    DesktopHistologyRefreshPresenter,
)


class FakeQueries:
    def __init__(self, line_state: Any = None) -> None:
        self.line_state = line_state
        self.line_calls: list[int] = []
        self.workspace = SimpleNamespace(
            active_shank_selection=self.active_shank_selection,
            active_reference_line_state=self.active_reference_line_state,
        )

    def active_shank_selection(self) -> Any:
        return SimpleNamespace(shank_idx=2)

    def active_reference_line_state(self, shank_idx: int) -> Any:
        self.line_calls.append(shank_idx)
        return self.line_state


class FakeHistologyDisplay:
    def __init__(self, render_result: bool = True) -> None:
        self.render_result = render_result
        self.render_count = 0

    def render_active_panels(self) -> bool:
        self.render_count += 1
        return self.render_result


class FakeSliceDisplay:
    def __init__(self) -> None:
        self.refresh_count = 0

    def refresh_perpendicular_histology(self) -> None:
        self.refresh_count += 1


class FakeReferenceLineDisplay:
    def __init__(self) -> None:
        self.created: list[tuple[Any, Any]] = []

    def create_lines(self, positions: Any, track_positions: Any = None) -> None:
        self.created.append((positions, track_positions))


def _presenter(
    *,
    line_state: Any = None,
    render_result: bool = True,
) -> tuple[
    DesktopHistologyRefreshPresenter,
    FakeQueries,
    FakeHistologyDisplay,
    FakeSliceDisplay,
    FakeReferenceLineDisplay,
]:
    queries = FakeQueries(line_state)
    histology = FakeHistologyDisplay(render_result)
    slice_display = FakeSliceDisplay()
    reference_lines = FakeReferenceLineDisplay()
    return (
        DesktopHistologyRefreshPresenter(
            app=SimpleNamespace(queries=queries),
            histology_display=histology,
            slice_display=slice_display,
            reference_line_display=reference_lines,
        ),
        queries,
        histology,
        slice_display,
        reference_lines,
    )


def test_render_loaded_shank_histology_restores_reference_lines() -> None:
    line_state = SimpleNamespace(
        feature_positions_um=[1.0],
        track_positions_um=[2.0],
    )
    presenter, queries, histology, slice_display, reference_lines = _presenter(
        line_state=line_state
    )

    assert presenter.render_loaded_shank_histology(1)

    assert histology.render_count == 1
    assert slice_display.refresh_count == 1
    assert queries.line_calls == [1]
    assert reference_lines.created == [([1.0], [2.0])]


def test_render_loaded_shank_histology_uses_active_shank_by_default() -> None:
    presenter, queries, _histology, _slice_display, _reference_lines = _presenter()

    assert presenter.render_loaded_shank_histology()

    assert queries.line_calls == [2]


def test_render_loaded_shank_histology_stops_when_panel_render_fails() -> None:
    presenter, queries, histology, slice_display, reference_lines = _presenter(
        render_result=False
    )

    assert not presenter.render_loaded_shank_histology(1)

    assert histology.render_count == 1
    assert slice_display.refresh_count == 0
    assert queries.line_calls == []
    assert reference_lines.created == []
