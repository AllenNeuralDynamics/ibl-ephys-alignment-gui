"""Narrow selected-stream context for alignment-key construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.document import AlignmentKey


@dataclass
class AlignmentKeyContext:
    """Resolved facts needed to build per-shank alignment keys.

    This is intentionally smaller than ``AlignmentDataContext``. The controller
    needs to know which recording/ephys collection is selected and how many
    shanks are valid; it should not depend on the full datapackage/cache object.
    """

    recording_id: str | None = None
    ephys_collection: str | None = None
    n_shanks: int = 0

    @property
    def is_ready(self) -> bool:
        """Whether alignment keys can be built for the selected stream."""
        return self.recording_id is not None and self.ephys_collection is not None

    def clear(self) -> None:
        """Clear selected stream facts."""
        self.recording_id = None
        self.ephys_collection = None
        self.n_shanks = 0

    def set_from_probe(self, probe: Any, *, n_shanks: int) -> None:
        """Store alignment-key facts resolved from a selected probe/stream."""
        if n_shanks < 0:
            raise ValueError("n_shanks must be non-negative")
        self.recording_id = probe.recording_id
        self.ephys_collection = probe.ephys_collection
        self.n_shanks = n_shanks

    def key_for_shank(self, shank_idx: int) -> AlignmentKey:
        """Return an alignment key for one shank of the selected stream."""
        if not self.is_ready:
            raise RuntimeError("No probe selected. Please select a probe first.")
        self.validate_shank(shank_idx)
        assert self.recording_id is not None
        assert self.ephys_collection is not None
        return AlignmentKey(
            recording_id=self.recording_id,
            ephys_collection=self.ephys_collection,
            shank_idx=shank_idx,
        )

    def validate_shank(self, shank_idx: int) -> None:
        """Raise if the shank is outside known channel metadata bounds."""
        if self.n_shanks > 0 and not 0 <= shank_idx < self.n_shanks:
            raise ValueError(
                f"Shank index {shank_idx} is outside valid range 0..{self.n_shanks - 1}"
            )
