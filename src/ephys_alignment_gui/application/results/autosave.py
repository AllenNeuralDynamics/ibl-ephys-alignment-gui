"""Autosave checkpoint command result DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.document_snapshot import AlignmentDocumentSnapshot


@dataclass(frozen=True)
class AutosaveCheckpointWritten:
    """A document checkpoint was written to disk."""

    path: Path
    alignment_state_count: int


@dataclass(frozen=True)
class AutosaveCheckpointRead:
    """A document checkpoint was read from disk."""

    path: Path
    snapshot: AlignmentDocumentSnapshot


@dataclass(frozen=True)
class AutosaveCheckpointRestored:
    """A document checkpoint was restored into the live document."""

    path: Path
    alignment_state_count: int
    selected_alignment_key: AlignmentKey | None


@dataclass(frozen=True)
class AutosaveCheckpointCleared:
    """A document checkpoint file was removed or was already absent."""

    path: Path
    existed: bool
