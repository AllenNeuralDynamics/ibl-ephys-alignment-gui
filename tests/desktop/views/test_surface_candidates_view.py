"""Tests for the brain-surface candidate tables."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.desktop.views.surface_candidates_view import (
    DesktopSurfaceCandidatesView,
)


class FakeSignal:
    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def connect(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def emit(self, *args: Any) -> None:
        for callback in self.callbacks:
            callback(*args)


class FakeTable:
    def __init__(self) -> None:
        self.items: dict[tuple[int, int], str] = {}
        self.headers: list[str] = []
        self.rows = 0
        self.blocked: list[bool] = []
        self.selected_row: int | None = None
        self.currentCellChanged = FakeSignal()
        self.cellClicked = FakeSignal()

    def blockSignals(self, block: bool) -> None:
        self.blocked.append(block)

    def clearContents(self) -> None:
        self.items.clear()

    def setColumnCount(self, _count: int) -> None:
        pass

    def setHorizontalHeaderLabels(self, labels: list[str]) -> None:
        self.headers = labels

    def setRowCount(self, rows: int) -> None:
        self.rows = rows

    def setItem(self, row: int, col: int, item: str) -> None:
        self.items[(row, col)] = item

    def selectRow(self, row: int) -> None:
        self.selected_row = row


def _view() -> DesktopSurfaceCandidatesView:
    return DesktopSurfaceCandidatesView(
        candidate_table=FakeTable(),
        line_table=FakeTable(),
        item_factory=str,
    )


def test_fill_replaces_rows_with_signals_blocked() -> None:
    view = _view()

    view.set_candidates(("#", "Depth"), [["1", "2122"], ["2", "1500"]])

    table = view.candidate_table
    assert table.headers == ["#", "Depth"]
    assert table.rows == 2
    assert table.items[(1, 1)] == "1500"
    assert table.blocked == [True, False]


def test_selection_signals_report_rows() -> None:
    view = _view()
    candidates: list[int] = []
    lines: list[int] = []
    view.connect_candidate_selected(candidates.append)
    view.connect_line_selected(lines.append)

    view.candidate_table.currentCellChanged.emit(1, 0, 0, 0)
    view.line_table.cellClicked.emit(3, 2)

    assert candidates == [1]
    assert lines == [3]
