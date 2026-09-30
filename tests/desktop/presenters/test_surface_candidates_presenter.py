"""Tests for the brain-surface candidate dock presenter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.core.alignment_read_models import (
    SurfaceCandidateLineState,
    SurfaceCandidateState,
)
from ephys_alignment_gui.desktop.presenters.surface_candidates_presenter import (
    CANDIDATE_HEADERS,
    LINE_HEADERS,
    SurfaceCandidatesPresenter,
)


class FakeView:
    def __init__(self) -> None:
        self.candidates: list[list[str]] = []
        self.lines: list[list[str]] = []
        self.headers: dict[str, Any] = {}
        self.selected: int | None = None

    def set_candidates(self, headers, rows) -> None:
        self.headers["candidates"] = headers
        self.candidates = rows

    def set_lines(self, headers, rows) -> None:
        self.headers["lines"] = headers
        self.lines = rows

    def select_candidate(self, row: int) -> None:
        self.selected = row


def _line(freq_hz: float, plot_key: str | None, agrees: bool = True):
    return SurfaceCandidateLineState(
        freq_hz=freq_hz,
        step_db=-12.34,
        extent_um=(2090.0, 2123.0),
        support=812.4,
        agrees=agrees,
        line_plot_key=plot_key,
    )


CANDIDATES = (
    SurfaceCandidateState(
        method="lines",
        midpoint_um=2122.5,
        width_um=None,
        delta_bic=3269.4,
        n_agreeing_lines=1,
        n_lines=2,
        lines=(_line(3870.25, "line.a"), _line(7503.1, None, agrees=False)),
    ),
    SurfaceCandidateState(
        method="driven",
        midpoint_um=1500.0,
        width_um=40.0,
        delta_bic=800.0,
        n_agreeing_lines=0,
        n_lines=0,
        lines=(),
    ),
)


def _presenter(candidates=CANDIDATES):
    view = FakeView()
    selected: list[str] = []
    app = SimpleNamespace(
        queries=SimpleNamespace(
            ephys=SimpleNamespace(active_surface_candidates=lambda: candidates)
        )
    )
    presenter = SurfaceCandidatesPresenter(
        app=app,
        view=view,
        select_line_plot=lambda key: selected.append(key) or True,
    )
    return presenter, view, selected


def test_refresh_lists_candidates_by_position_and_selects_the_strongest() -> None:
    presenter, view, _selected = _presenter()

    presenter.refresh()

    assert view.headers == {"candidates": CANDIDATE_HEADERS, "lines": LINE_HEADERS}
    assert view.candidates == [
        ["1", "2122", "–", "lines", "1/2", "3269"],
        ["2", "1500", "40", "driven", "0/0", "800"],
    ]
    assert view.selected == 0
    assert view.lines == [
        ["3870.2", "-12.3", "2090–2123", "812", "yes"],
        ["7503.1", "-12.3", "2090–2123", "812", "no"],
    ]


def test_selecting_a_candidate_shows_its_lines() -> None:
    presenter, view, _selected = _presenter()
    presenter.refresh()

    presenter.candidate_selected(1)

    assert view.lines == []


def test_clicking_a_line_shows_its_plot_when_it_has_one() -> None:
    presenter, _view, selected = _presenter()
    presenter.refresh()

    presenter.line_selected(0)
    presenter.line_selected(1)
    presenter.line_selected(5)

    assert selected == ["line.a"]


def test_refresh_without_candidates_clears_both_tables() -> None:
    presenter, view, selected = _presenter(candidates=())

    presenter.refresh()
    presenter.line_selected(0)

    assert view.candidates == []
    assert view.lines == []
    assert view.selected is None
    assert selected == []
