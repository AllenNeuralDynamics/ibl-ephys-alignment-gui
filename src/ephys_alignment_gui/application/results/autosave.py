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


@dataclass(frozen=True)
class AutosaveCheckpointSkippedKey:
    """One checkpoint alignment key was not valid for the loaded input dataset."""

    key: AlignmentKey
    reason: str


@dataclass(frozen=True)
class AutosaveCheckpointInspected:
    """Summary of a checkpoint before mutating the live document."""

    path: Path
    mouse_id: str | None
    mouse_root: Path | None
    output_package_directory: Path | None
    selected_alignment_key: AlignmentKey | None
    alignment_state_count: int
    saveable_alignment_count: int
    dirty_alignment_count: int
    recoverable_alignment_count: int
    skipped_keys: tuple[AutosaveCheckpointSkippedKey, ...] = ()


@dataclass(frozen=True)
class AutosaveCheckpointRecovered:
    """A checkpoint was validated, restored, and normalized for GUI recovery."""

    path: Path
    backup_path: Path | None
    selected_alignment_key: AlignmentKey | None
    restored_alignment_count: int
    skipped_keys: tuple[AutosaveCheckpointSkippedKey, ...] = ()
    warnings: tuple[str, ...] = ()
