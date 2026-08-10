"""Typed GUI events for active alignment changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.document import AlignmentKey

AlignmentEditKind = Literal[
    "fit",
    "offset",
    "next",
    "previous",
    "reset",
]


@dataclass(frozen=True)
class AlignmentEdited:
    """Application event emitted after editable alignment state changes.

    This is the durable command-level event. It intentionally carries no
    plot-specific payloads or desktop refresh instructions.
    """

    edit_kind: AlignmentEditKind
    active_key: AlignmentKey
    active_alignment: ActiveAlignment
    lin_fit: bool | None = None


@dataclass(frozen=True)
class ShankChanged:
    """Payload emitted after the active shank selection has changed."""

    source: str
    previous_shank_idx: int
    shank_idx: int
    previous_key: AlignmentKey | None
    active_key: AlignmentKey | None
    data_loaded: bool
    preserve_plot_selection: bool | None = None
