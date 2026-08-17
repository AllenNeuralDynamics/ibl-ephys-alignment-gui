"""Pure editable state for one shank alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_edit_history import AlignmentEditHistory

LEGACY_AUTO_ALIGNMENT_LABEL = "auto"
AlignmentSignature = tuple[tuple[float, ...], tuple[float, ...], bool]


@dataclass
class AlignmentSaveState:
    """Per-alignment save revision metadata."""

    revision: int = 0
    saved_revision: int = 0
    saved_signature: AlignmentSignature | None = None

    def is_dirty(self, active_signature: AlignmentSignature | None) -> bool:
        """Whether the current active alignment differs from the saved state."""
        if active_signature is None or self.revision == self.saved_revision:
            return False
        return active_signature != self.saved_signature

    def mark_changed(self) -> None:
        """Record that a user edit changed the saveable alignment output."""
        self.revision += 1

    def mark_saved(self, active_signature: AlignmentSignature | None) -> None:
        """Record that the current revision was successfully persisted."""
        self.saved_revision = self.revision
        self.saved_signature = active_signature


@dataclass(frozen=True)
class PendingReferenceLines:
    """Document-owned coordinates for user reference lines.

    The Qt/pyqtgraph line objects remain view state. This value object stores
    only the paired feature-space and warped-space display y positions that
    should be recreated when a shank/probe view is rebuilt. The warped-space
    positions are not raw track depths; fitting converts them through the
    current warp to select raw track positions.
    """

    feature_positions_um: NDArray[np.floating[Any]]
    warped_positions_um: NDArray[np.floating[Any]]

    def __post_init__(self) -> None:
        feature_positions_um = np.array(
            self.feature_positions_um,
            dtype=float,
            copy=True,
        )
        warped_positions_um = np.array(
            self.warped_positions_um,
            dtype=float,
            copy=True,
        )
        if feature_positions_um.ndim != 1 or warped_positions_um.ndim != 1:
            raise ValueError("reference-line positions must be 1D arrays")
        if feature_positions_um.shape != warped_positions_um.shape:
            raise ValueError(
                "feature and warped reference-line positions must have matching shapes"
            )
        feature_positions_um.setflags(write=False)
        warped_positions_um.setflags(write=False)
        object.__setattr__(self, "feature_positions_um", feature_positions_um)
        object.__setattr__(self, "warped_positions_um", warped_positions_um)

    @property
    def track_positions_um(self) -> NDArray[np.floating[Any]]:
        """Backward-compatible alias for warped-space display positions."""
        return self.warped_positions_um

    @classmethod
    def from_values(
        cls,
        feature_positions_um: Any,
        warped_positions_um: Any,
    ) -> PendingReferenceLines | None:
        """Create pending reference lines, returning None for empty input."""
        feature = np.asarray(feature_positions_um, dtype=float)
        warped = np.asarray(warped_positions_um, dtype=float)
        if feature.size == 0 and warped.size == 0:
            return None
        return cls(feature, warped)


@dataclass
class AlignmentState:
    """Document-owned editable alignment state for one alignment key.

    ``AlignmentDocument`` is the top-level workspace/document. This object is
    the per-recording/per-stream/per-shank edit state that the document will
    eventually store by ``AlignmentKey``. It intentionally owns no loaded ephys
    arrays, atlas images, plot data, slice images, ``EphysAlignment`` engine, or
    Qt/pyqtgraph objects.
    """

    max_idx: int = 10
    alignments: dict[str, list[list[float]]] = field(default_factory=dict)
    prev_align: list[str] = field(default_factory=lambda: ["original"])
    feature_prev: Any = None
    track_prev: Any = None
    pending_reference_lines: PendingReferenceLines | None = None
    save_state: AlignmentSaveState = field(default_factory=AlignmentSaveState)
    edit_history: AlignmentEditHistory = field(init=False)

    def __post_init__(self) -> None:
        self.edit_history = AlignmentEditHistory(max_idx=self.max_idx)
        self._refresh_prev_align()

    @property
    def active_alignment(self) -> ActiveAlignment | None:
        """Current feature/track control points for this alignment."""
        return self.edit_history.current_alignment

    @active_alignment.setter
    def active_alignment(self, alignment: ActiveAlignment | None) -> None:
        if alignment is None:
            self.edit_history.clear_current_alignment()
            return
        self.edit_history.set_current_alignment(alignment)

    @property
    def has_unsaved_alignment(self) -> bool:
        """Whether this state has saveable edits not yet persisted."""
        return self.save_state.is_dirty(self._active_alignment_signature())

    @property
    def has_saveable_alignment(self) -> bool:
        """Whether this state has active alignment output to materialize."""
        return self.active_alignment is not None

    def mark_alignment_changed(self) -> None:
        """Record that the current working alignment changed via user edit."""
        self.save_state.mark_changed()

    def mark_saved(self) -> None:
        """Record that the current working alignment output was saved."""
        self.save_state.mark_saved(self._active_alignment_signature())

    def set_alignments(self, alignments: dict[str, list[list[float]]]) -> None:
        """Replace persisted alignment history and rebuild dropdown order."""
        self.alignments = self._filtered_alignments(alignments)
        self._refresh_prev_align()

    def merge_alignments(self, alignments: dict[str, list[list[float]]]) -> None:
        """Merge persisted history without dropping existing local entries."""
        merged = dict(self.alignments)
        for key, value in self._filtered_alignments(alignments).items():
            if key not in merged or merged[key] == value:
                merged[key] = value
                continue
            merged[self._disambiguated_import_key(key, merged)] = value
        self.alignments = merged
        self._refresh_prev_align()

    def import_alignments(
        self,
        alignments: dict[str, list[list[float]]],
    ) -> None:
        """Import previous alignments without replacing active user edits."""
        if self.has_unsaved_alignment or self.pending_reference_lines is not None:
            self.merge_alignments(alignments)
        else:
            self.set_alignments(alignments)

    @staticmethod
    def _filtered_alignments(
        alignments: dict[str, list[list[float]]],
    ) -> dict[str, list[list[float]]]:
        return {
            key: value
            for key, value in alignments.items()
            if key != LEGACY_AUTO_ALIGNMENT_LABEL
        }

    @staticmethod
    def _disambiguated_import_key(
        key: str,
        alignments: dict[str, list[list[float]]],
    ) -> str:
        idx = 1
        candidate = f"{key}.imported.{idx}"
        while candidate in alignments:
            idx += 1
            candidate = f"{key}.imported.{idx}"
        return candidate

    def add_alignment(self, feature: NDArray, track: NDArray) -> str:
        """Record a new saved alignment and return its unique timestamp key."""
        key, alignments = self.with_alignment_added(feature, track)
        self.alignments = alignments
        self._refresh_prev_align()
        return key

    def with_alignment_added(
        self,
        feature: NDArray,
        track: NDArray,
    ) -> tuple[str, dict[str, list[list[float]]]]:
        """Return a history copy with one alignment appended."""
        alignments = dict(self.alignments)
        base = datetime.now().replace(microsecond=0).isoformat()
        date = base
        n = 1
        while date in alignments:
            date = f"{base}.{n}"
            n += 1
        alignments[date] = [feature.tolist(), track.tolist()]
        return date, alignments

    def alignment_history_for_save(self) -> dict[str, list[list[float]]]:
        """Return persisted history with the active alignment represented once."""
        alignment = self.active_alignment
        if alignment is None or self._active_alignment_is_in_history(alignment):
            return dict(self.alignments)
        _key, alignments = self.with_alignment_added(alignment.feature, alignment.track)
        return alignments

    def select_alignment_idx(
        self,
        idx: int,
    ) -> tuple[NDArray | None, NDArray | None]:
        """Select a saved/original alignment and rebase working history."""
        feature, track = self.get_alignment_idx(idx)
        self.feature_prev = feature
        self.track_prev = track
        alignment = None
        if feature is not None and track is not None:
            alignment = ActiveAlignment(feature, track)
        self.rebase_working_alignment(alignment)
        self.clear_pending_reference_lines()
        return feature, track

    def clear_previous_alignment_selection(self) -> None:
        """Clear the previous-alignment seed used to initialize runtime state."""
        self.feature_prev = None
        self.track_prev = None
        self.clear_pending_reference_lines()

    def rebase_working_alignment(self, alignment: ActiveAlignment | None) -> None:
        """Reset undo/redo history to a new starting alignment."""
        self.edit_history = AlignmentEditHistory(max_idx=self.max_idx)
        self.active_alignment = alignment

    def set_pending_reference_lines(
        self,
        lines: PendingReferenceLines | None,
    ) -> None:
        """Store pending reference-line coordinates for this alignment."""
        self.pending_reference_lines = lines

    def clear_pending_reference_lines(self) -> None:
        """Clear pending reference-line coordinates for this alignment."""
        self.pending_reference_lines = None

    def get_alignment_idx(self, idx: int) -> tuple[NDArray | None, NDArray | None]:
        """Return ``(feature, track)`` for a dropdown index."""
        if len(self.prev_align) <= idx:
            return None, None
        alignment = self.prev_align[idx]
        if alignment == "original":
            return None, None
        feature = np.array(self.alignments[alignment][0])
        track = np.array(self.alignments[alignment][1])
        return feature, track

    def _refresh_prev_align(self) -> None:
        self.prev_align = self._ordered_keys(self.alignments)

    def _active_alignment_signature(self) -> AlignmentSignature | None:
        alignment = self.active_alignment
        if alignment is None:
            return None
        return (
            tuple(float(value) for value in alignment.feature),
            tuple(float(value) for value in alignment.track),
            alignment.lin_fit,
        )

    def _active_alignment_is_in_history(self, alignment: ActiveAlignment) -> bool:
        return any(
            self._history_entry_matches_alignment(value, alignment)
            for value in self.alignments.values()
        )

    @staticmethod
    def _history_entry_matches_alignment(
        value: Any,
        alignment: ActiveAlignment,
    ) -> bool:
        try:
            feature, track = value
            feature_arr = np.asarray(feature, dtype=float)
            track_arr = np.asarray(track, dtype=float)
        except (TypeError, ValueError):
            return False
        return (
            feature_arr.shape == alignment.feature.shape
            and track_arr.shape == alignment.track.shape
            and np.allclose(feature_arr, alignment.feature)
            and np.allclose(track_arr, alignment.track)
        )

    @staticmethod
    def _ordered_keys(alignments: dict[str, Any]) -> list[str]:
        prev_align = sorted(
            key for key in alignments.keys() if key != LEGACY_AUTO_ALIGNMENT_LABEL
        )
        prev_align.reverse()
        prev_align.append("original")
        return prev_align
