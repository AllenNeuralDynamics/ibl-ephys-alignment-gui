"""Transient edit history for one shank alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        self.lin_fit_history = [True] * (self.max_idx + 1)
