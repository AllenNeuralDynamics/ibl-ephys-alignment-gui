"""Qt-free operations for transient alignment edit history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_edit_history import AlignmentEditHistory


@dataclass(frozen=True)
class AlignmentEditResult:
    """Outcome of an edit-history command."""

    changed: bool
    alignment: ActiveAlignment | None = None
    lin_fit: bool | None = None


class AlignmentEditService:
    """Mutates alignment edit history without depending on Qt or plots."""

    def go_next(self, history: AlignmentEditHistory) -> AlignmentEditResult:
        """Move the edit cursor forward one saved alignment, if possible."""
        if not (
            history.current_idx < history.total_idx
            and history.current_idx > history.total_idx - history.max_idx
        ):
            return AlignmentEditResult(changed=False)

        history.current_idx += 1
        history.idx = self._slot(history)
        return self._result_from_history(history)

    def go_previous(self, history: AlignmentEditHistory) -> AlignmentEditResult:
        """Move the edit cursor backward one saved alignment, if possible."""
        if history.total_idx > history.last_idx:
            history.last_idx = history.total_idx

        earliest_available_idx = max(0, history.total_idx - history.diff_idx)
        if history.current_idx <= earliest_available_idx:
            return AlignmentEditResult(changed=False)

        history.current_idx -= 1
        history.idx = self._slot(history)
        return self._result_from_history(history)

    def reset_to_initial(
        self,
        history: AlignmentEditHistory,
        *,
        feature_init: NDArray[Any],
        track_init: NDArray[Any],
        lin_fit: bool,
    ) -> AlignmentEditResult:
        """Append the probe's initial alignment as the newest edit state."""
        if history.current_idx < history.last_idx:
            history.total_idx = history.current_idx
            history.diff_idx = self._future_edit_span(history)
        else:
            history.diff_idx = history.max_idx - 1

        history.total_idx += 1
        history.current_idx += 1
        history.idx = self._slot(history)
        history.set_current_alignment(
            ActiveAlignment(feature_init, track_init, lin_fit=lin_fit)
        )
        return self._result_from_history(history)

    @staticmethod
    def _slot(history: AlignmentEditHistory) -> int:
        return int(history.current_idx % history.max_idx)

    @staticmethod
    def _future_edit_span(history: AlignmentEditHistory) -> int:
        diff_idx = int(
            np.mod(history.last_idx, history.max_idx)
            - np.mod(history.total_idx, history.max_idx)
        )
        if diff_idx >= 0:
            return history.max_idx - diff_idx
        return abs(diff_idx)

    @staticmethod
    def _result_from_history(history: AlignmentEditHistory) -> AlignmentEditResult:
        return AlignmentEditResult(
            changed=True,
            alignment=history.current_alignment,
            lin_fit=history.lin_fit_history[history.idx],
        )
