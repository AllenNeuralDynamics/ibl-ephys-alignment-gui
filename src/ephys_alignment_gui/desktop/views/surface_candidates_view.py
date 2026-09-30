"""Brain-surface candidate tables in their dock."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from PyQt6 import QtWidgets


@dataclass
class DesktopSurfaceCandidatesView:
    """Candidate and per-line tables; the line table follows the selected
    candidate."""

    candidate_table: Any
    line_table: Any
    item_factory: Callable[[str], Any] = QtWidgets.QTableWidgetItem

    def set_candidates(
        self, headers: Sequence[str], rows: Sequence[Sequence[str]]
    ) -> None:
        """Replace the candidate rows."""
        self._fill(self.candidate_table, headers, rows)

    def set_lines(self, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        """Replace the selected candidate's line rows."""
        self._fill(self.line_table, headers, rows)

    def select_candidate(self, row: int) -> None:
        """Highlight one candidate row."""
        self.candidate_table.selectRow(row)

    def connect_candidate_selected(self, callback: Callable[[int], None]) -> None:
        """Call back with the row of each newly current candidate."""
        self.candidate_table.currentCellChanged.connect(
            lambda row, _col, _prev_row, _prev_col: callback(row)
        )

    def connect_line_selected(self, callback: Callable[[int], None]) -> None:
        """Call back with the row of each clicked line, including a re-click."""
        self.line_table.cellClicked.connect(lambda row, _col: callback(row))

    def _fill(
        self, table: Any, headers: Sequence[str], rows: Sequence[Sequence[str]]
    ) -> None:
        # Refilling moves the current cell; the stale rows must not report it.
        table.blockSignals(True)
        try:
            table.clearContents()
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(list(headers))
            table.setRowCount(len(rows))
            for row, values in enumerate(rows):
                for col, value in enumerate(values):
                    table.setItem(row, col, self.item_factory(value))
        finally:
            table.blockSignals(False)
