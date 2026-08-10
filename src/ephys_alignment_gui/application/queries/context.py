"""Shared context helpers for app query builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey


@dataclass(frozen=True)
class ActiveAlignmentContext:
    """Active alignment, key, and shank runtime available for read models."""

    key: AlignmentKey
    active_alignment: ActiveAlignment
    shank_runtime: Any


@dataclass(frozen=True)
class AlignmentQueryContext:
    """Common active document/runtime lookups for app query builders."""

    document: AlignmentDocument
    runtime: Any

    def active_shank_idx(self) -> int:
        """Return the active shank index from the selected alignment key."""
        key = self.document.selected_alignment_key
        if key is not None:
            return key.shank_idx
        return self.document.selected_shank

    def active_shank_runtime(self) -> Any | None:
        """Return active shank runtime data, if loaded."""
        stream_runtime = getattr(self.runtime, "active_stream_runtime", None)
        if stream_runtime is None:
            return None
        runtimes = getattr(stream_runtime, "shank_runtime_by_idx", None)
        if runtimes is None:
            return None
        return runtimes.get(self.active_shank_idx())

    def active_alignment_context(self) -> ActiveAlignmentContext | None:
        """Return active alignment plus loaded shank runtime, if available."""
        key = self.document.selected_alignment_key
        state = self.document.active_alignment_state
        if key is None or state is None:
            return None
        active_alignment = state.active_alignment
        if active_alignment is None:
            return None
        shank_runtime = self.active_shank_runtime()
        if shank_runtime is None or getattr(shank_runtime, "ephysalign", None) is None:
            return None
        return ActiveAlignmentContext(
            key=key,
            active_alignment=active_alignment,
            shank_runtime=shank_runtime,
        )
