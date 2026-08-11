"""Transient edit history for one shank alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ephys_alignment_gui.core.active_alignment import ActiveAlignment


@dataclass
class AlignmentEditHistory:
    """Cyclic fit/undo buffer for an active shank edit session.

    This is editor runtime state, not persisted document state. The active
    alignment can be autosaved separately; this object preserves transient undo
    history while the stream/shank is loaded.
    """

    max_idx: int = 10
    idx: int = 0
    current_idx: int = 0
    total_idx: int = 0
    last_idx: int = 0
    diff_idx: int = 0
    idx_prev: int = 0
    track: list[Any] = field(init=False)
    features: list[Any] = field(init=False)
    lin_fit_history: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.track = [0] * (self.max_idx + 1)
        self.features = [0] * (self.max_idx + 1)
        self.lin_fit_history = [False] * (self.max_idx + 1)

    @property
    def current_alignment(self) -> ActiveAlignment | None:
        """Return the active alignment at the current cursor, if initialized."""
        return ActiveAlignment.from_values(
            self.features[self.idx],
            self.track[self.idx],
            lin_fit=self.lin_fit_history[self.idx],
        )

    def set_current_alignment(self, alignment: ActiveAlignment) -> None:
        """Replace the active cursor slot from an alignment value."""
        self.features[self.idx] = alignment.feature_copy()
        self.track[self.idx] = alignment.track_copy()
        self.lin_fit_history[self.idx] = alignment.lin_fit

    def clear_current_alignment(self) -> None:
        """Clear the active cursor slot back to the legacy blank sentinel."""
        self.features[self.idx] = 0
        self.track[self.idx] = 0
        self.lin_fit_history[self.idx] = False
