"""Persistence command DTOs for alignment output writes."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.services.alignment_repository import (
    AlignmentHistory,
    SavedAlignmentOutputs,
)


@dataclass(frozen=True)
class NoPreviousAlignments:
    """No previous alignments were available."""


@dataclass(frozen=True)
class PreviousAlignmentPackageLoaded:
    """Previous alignments were loaded for multiple stream/shank keys."""

    loaded_count: int
    loaded_keys: tuple[AlignmentKey, ...]
    active_choices: list[str] | None = None


@dataclass(frozen=True)
class AlignmentOutputBuilt:
    """Output dictionaries computed from alignment channel locations."""

    channel_results: dict
    ccf_channel_results: dict
    multi_shank: bool


@dataclass(frozen=True)
class AlignmentOutputsSaved:
    """Alignment output files were persisted."""

    saved: SavedAlignmentOutputs
    previous_alignments: AlignmentHistory
