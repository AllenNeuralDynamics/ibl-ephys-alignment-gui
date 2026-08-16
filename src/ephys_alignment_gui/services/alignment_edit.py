"""Qt-free operations for transient alignment edit history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_edit_history import AlignmentEditHistory


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
        self._append_edit_slot(history, lin_fit=lin_fit, remember_previous=False)
        history.set_current_alignment(
            ActiveAlignment(feature_init, track_init, lin_fit=lin_fit)
        )
        return self._result_from_history(history)

    def fit_to_reference_lines(
        self,
        history: AlignmentEditHistory,
        *,
        ephysalign: Any,
        line_features_um: NDArray[Any],
        line_tracks_um: NDArray[Any],
        lin_fit: bool,
        extend_feature: int,
    ) -> AlignmentEditResult:
        """Append a fit edit from user-positioned feature and track lines."""
        line_feature = np.asarray(line_features_um, dtype=float) / 1e6
        line_track = np.asarray(line_tracks_um, dtype=float) / 1e6
        if line_feature.size == 0 or line_track.size == 0:
            return self.reset_to_initial(
                history,
                feature_init=ephysalign.feature_init,
                track_init=ephysalign.track_init,
                lin_fit=lin_fit,
            )

        self._append_edit_slot(history, lin_fit=lin_fit, remember_previous=True)

        previous_feature = np.asarray(history.features[history.idx_prev])
        previous_track = np.asarray(history.track[history.idx_prev])

        depths_track = np.sort(np.r_[previous_track[[0, -1]], line_track])
        track = ephysalign.feature2track(
            depths_track,
            previous_feature,
            previous_track,
        )
        feature = np.sort(np.r_[previous_feature[[0, -1]], line_feature])

        if feature.size >= 5 and lin_fit:
            feature, track = ephysalign.adjust_extremes_linear(
                feature,
                track,
                extend_feature,
            )
        else:
            track = ephysalign.adjust_extremes_uniform(feature, track)

        history.set_current_alignment(ActiveAlignment(feature, track, lin_fit=lin_fit))
        return self._result_from_history(history)

    def offset_from_tip(
        self,
        history: AlignmentEditHistory,
        *,
        tip_position_um: float,
        probe_tip_um: float,
        lin_fit: bool,
        track_shift_m: float = 0.0,
    ) -> AlignmentEditResult:
        """Append an offset edit from the current tip line position."""
        self._append_edit_slot(history, lin_fit=lin_fit, remember_previous=True)

        feature = np.array(history.features[history.idx_prev], copy=True)
        track = np.array(history.track[history.idx_prev], copy=True)
        if track_shift_m:
            track = track + track_shift_m

        offset_delta = (tip_position_um - probe_tip_um) / 1e6
        track[0] += offset_delta
        track[-1] += offset_delta

        history.set_current_alignment(ActiveAlignment(feature, track, lin_fit=lin_fit))
        return self._result_from_history(history)

    def _append_edit_slot(
        self,
        history: AlignmentEditHistory,
        *,
        lin_fit: bool,
        remember_previous: bool,
    ) -> None:
        if history.current_idx < history.last_idx:
            history.total_idx = history.current_idx
            history.diff_idx = self._future_edit_span(history)
        else:
            history.diff_idx = history.max_idx - 1

        history.total_idx += 1
        history.current_idx += 1
        if remember_previous:
            history.idx_prev = history.idx
        history.idx = self._slot(history)
        history.lin_fit_history[history.idx] = lin_fit

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
