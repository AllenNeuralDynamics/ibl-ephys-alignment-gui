"""Prepared alignment-output save jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ephys_alignment_gui.application.results.alignment_persistence import (
    AlignmentOutputsSaved,
)
from ephys_alignment_gui.core.alignment_output import (
    AlignmentOutputInput,
    AlignmentOutputMetadata,
)
from ephys_alignment_gui.core.alignment_state import AlignmentState
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.services.alignment_repository import AlignmentHistory


@dataclass(frozen=True)
class PreparedAlignmentSaveTarget:
    """Prepared save inputs for one alignment output.

    The worker phase uses the copied arrays, alignment history, and output path.
    The document ``state`` reference is only consumed later on the application
    thread when a successful save is published.
    """

    key: AlignmentKey
    state: AlignmentState
    output_input: AlignmentOutputInput
    output_metadata: AlignmentOutputMetadata
    output_directory: Path
    multi_shank: bool
    alignments_to_save: AlignmentHistory


@dataclass(frozen=True)
class PreparedAlignmentSave:
    """Prepared save job that can build/write outputs off the GUI thread."""

    targets: tuple[PreparedAlignmentSaveTarget, ...]
    use_docdb: bool

    @property
    def target_keys(self) -> tuple[AlignmentKey, ...]:
        """Return alignment keys in save order."""
        return tuple(target.key for target in self.targets)


@dataclass
class AlignmentSaveCancelToken:
    """Cooperative cancellation flag for prepared alignment save jobs."""

    reason: str | None = None

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self.reason is not None

    def cancel(self, reason: str = "cancelled") -> None:
        """Request cancellation at the next save-job checkpoint."""
        self.reason = reason


@dataclass(frozen=True)
class AlignmentSaveJobCompleted:
    """Terminal result from the thread-safe save job phase."""

    saved_outputs: dict[AlignmentKey, AlignmentOutputsSaved]


@dataclass(frozen=True)
class AlignmentSaveJobCancelled:
    """Prepared alignment save was cancelled at a cooperative checkpoint."""

    reason: str
