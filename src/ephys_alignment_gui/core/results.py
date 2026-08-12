"""Core document/controller result DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_state import PendingReferenceLines
from ephys_alignment_gui.core.document import AlignmentKey


@dataclass(frozen=True)
class LoadDataPrepared:
    """State needed by a UI before heavy data loading starts."""

    preserve_plot_selection: bool


@dataclass(frozen=True)
class ShankSelected:
    """The active shank selection changed in the document."""

    previous_key: AlignmentKey | None
    selected_key: AlignmentKey | None
    previous_shank_idx: int
    shank_idx: int
    data_loaded: bool


@dataclass(frozen=True)
class AlignmentChoicesUpdated:
    """Alignment dropdown choices were updated for the active state."""

    choices: list[str]


@dataclass(frozen=True)
class PreviousAlignmentSelected:
    """A previous/original alignment choice was selected."""

    feature_prev: Any
    track_prev: Any
    choice: str | None
    choices: list[str]


@dataclass(frozen=True)
class PendingReferenceLinesUpdated:
    """Pending reference-line coordinates were updated for the active state."""

    lines: PendingReferenceLines | None


@dataclass(frozen=True)
class AlignmentEditApplied:
    """An editable alignment command changed the active alignment."""

    alignment: ActiveAlignment
    lin_fit: bool | None = None


@dataclass(frozen=True)
class AlignmentEditNoop:
    """An editable alignment command completed without changing state."""


@dataclass(frozen=True)
class ShankRuntimeInitialized:
    """Runtime alignment engine was initialized for one shank."""

    feature_init: Any
    track_init: Any
    track_annos_and_ends_ras: Any
    seeded_document_alignment: bool
