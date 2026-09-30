"""Present the active shank's brain-surface candidates in their dock."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.core.alignment_read_models import (
    SurfaceCandidateLineState,
    SurfaceCandidateState,
)

CANDIDATE_HEADERS = (
    "#",
    "Depth (µm)",
    "Width (µm)",
    "Method",
    "Agreeing lines",
    "ΔBIC",
)
LINE_HEADERS = ("Line (Hz)", "Step (dB)", "Extent (µm)", "Support", "Agrees")


@dataclass
class SurfaceCandidatesPresenter:
    """Fill the candidate tables and show a clicked line's amplitude plot."""

    app: Any
    view: Any
    select_line_plot: Callable[[str], bool]
    _candidates: tuple[SurfaceCandidateState, ...] = ()
    _selected: int | None = None

    def connect(self) -> None:
        """Subscribe to the view's selections."""
        self.view.connect_candidate_selected(self.candidate_selected)
        self.view.connect_line_selected(self.line_selected)

    def refresh(self) -> None:
        """Show the active shank's candidates, the strongest selected."""
        self._candidates = self.app.queries.ephys.active_surface_candidates()
        self.view.set_candidates(
            CANDIDATE_HEADERS,
            [
                _candidate_row(position, candidate)
                for position, candidate in enumerate(self._candidates, start=1)
            ],
        )
        self.candidate_selected(0)
        if self._candidates:
            self.view.select_candidate(0)

    def candidate_selected(self, row: int) -> None:
        """Show one candidate's lines; an out-of-range row clears them."""
        self._selected = row if 0 <= row < len(self._candidates) else None
        lines = () if self._selected is None else self._candidates[row].lines
        self.view.set_lines(LINE_HEADERS, [_line_row(line) for line in lines])

    def line_selected(self, row: int) -> None:
        """Show the clicked line's amplitude plot, if it has one here."""
        if self._selected is None:
            return
        lines = self._candidates[self._selected].lines
        if 0 <= row < len(lines) and lines[row].line_plot_key is not None:
            self.select_line_plot(lines[row].line_plot_key)


def _candidate_row(position: int, candidate: SurfaceCandidateState) -> list[str]:
    return [
        str(position),
        f"{candidate.midpoint_um:.0f}",
        "–" if candidate.width_um is None else f"{candidate.width_um:.0f}",
        candidate.method,
        f"{candidate.n_agreeing_lines}/{candidate.n_lines}",
        f"{candidate.delta_bic:.0f}",
    ]


def _line_row(line: SurfaceCandidateLineState) -> list[str]:
    low, high = line.extent_um
    return [
        f"{line.freq_hz:.1f}",
        f"{line.step_db:+.1f}",
        f"{low:.0f}–{high:.0f}",
        f"{line.support:.0f}",
        "yes" if line.agrees else "no",
    ]
