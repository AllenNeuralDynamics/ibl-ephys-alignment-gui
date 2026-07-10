"""Pure editable state for one shank alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_edit_history import AlignmentEditHistory


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
    edit_history: AlignmentEditHistory = field(init=False)

    def __post_init__(self) -> None:
        self.edit_history = AlignmentEditHistory(max_idx=self.max_idx)
        self.prev_align = self._ordered_keys(self.alignments)

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

    def set_alignments(self, alignments: dict[str, list[list[float]]]) -> None:
        """Replace persisted alignment history and rebuild dropdown order."""
        self.alignments = alignments
        self.prev_align = self._ordered_keys(alignments)

    def add_alignment(self, feature: NDArray, track: NDArray) -> str:
        """Record a new saved alignment and return its unique timestamp key."""
        base = datetime.now().replace(microsecond=0).isoformat()
        date = base
        n = 1
        while date in self.alignments:
            date = f"{base}.{n}"
            n += 1
        self.alignments[date] = [feature.tolist(), track.tolist()]
        self.prev_align = self._ordered_keys(self.alignments)
        return date

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

    @staticmethod
    def _ordered_keys(alignments: dict[str, Any]) -> list[str]:
        prev_align = sorted(alignments.keys(), reverse=True)
        prev_align.append("original")
        return prev_align
