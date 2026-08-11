"""Immutable targets for fresh ephys/histology load jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ephys_alignment_gui.io.datapackage_loader import MouseRoot, ProbeInfo
from ephys_alignment_gui.runtime.ephys_stream import StreamKey
from ephys_alignment_gui.services.ephys_data import ChannelTable


@dataclass(frozen=True, eq=False)
class LoadDataJobTarget:
    """Resolved load target snapshot independent of mutable UI selection."""

    recording_id: str
    probe_name: str
    stream_key: StreamKey
    shank_idx: int
    mouse_root: MouseRoot
    probe_info: ProbeInfo
    channel_table: ChannelTable

    @property
    def mouse_root_path(self) -> Path:
        """Filesystem root represented by this load target."""
        return self.mouse_root.root

    @property
    def identity(self) -> tuple[str, str, StreamKey, int, Path]:
        """Small equality key that avoids comparing array-bearing metadata."""
        return (
            self.recording_id,
            self.probe_name,
            self.stream_key,
            self.shank_idx,
            self.mouse_root_path,
        )

    def same_identity(self, other: object) -> bool:
        """Return whether another target addresses the same stream/shank."""
        if not isinstance(other, LoadDataJobTarget):
            return False
        return self.identity == other.identity
