"""Selected datapackage/probe metadata context."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ephys_alignment_gui.datapackage_loader import (
    MouseRoot,
    ProbeInfo,
    load_mouse_root,
)
from ephys_alignment_gui.services.ephys_data import ChannelTable, EphysStreamData

logger = logging.getLogger(__name__)


@dataclass
class AlignmentDataContext:
    """Resolved metadata for the selected mouse root and ephys stream.

    This object intentionally owns only lightweight metadata: the resolved
    mouse-root datapackage, selected probe, and stream-level channel table.
    Heavy ephys arrays, active shank state, histology images, slices, and plot
    data live in services/runtime models outside this context.
    """

    mouse_root: MouseRoot | None = None
    probe_info: ProbeInfo | None = None
    channel_table: ChannelTable | None = None

    def set_mouse_root(self, mouse_root: Path) -> MouseRoot:
        """Resolve a mouse-root directory and clear selected-probe metadata."""
        loaded = load_mouse_root(Path(mouse_root))
        self.mouse_root = loaded
        self.probe_info = None
        self.channel_table = None
        return loaded

    def list_sessions(self) -> list[str]:
        """Recording IDs available in the current mouse root."""
        if self.mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        return self.mouse_root.sessions

    def list_probes(self, recording_id: str) -> list[str]:
        """Probe names for a recording in the current mouse root."""
        if self.mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        return self.mouse_root.probes_for_session(recording_id)

    def select_probe(self, recording_id: str, probe_name: str) -> ProbeInfo:
        """Resolve and select a probe, clearing channel-table metadata."""
        if self.mouse_root is None:
            raise RuntimeError("No mouse root loaded — call set_mouse_root() first")
        probe = self.mouse_root.get_probe(recording_id, probe_name)
        self.probe_info = probe
        self.channel_table = None
        return probe

    def stream_key_for_selection(
        self,
        recording_id: str,
        probe_name: str,
    ) -> tuple[str, str] | None:
        """Return the ephys stream key for a recording/probe selection."""
        if self.mouse_root is None or not recording_id or not probe_name:
            return None
        probe = self.mouse_root.get_probe(recording_id, probe_name)
        return probe.recording_id, probe.ephys_collection

    def attach_channel_table(self, channel_table: ChannelTable) -> None:
        """Attach stream-level channel metadata for the selected probe."""
        if self.probe_info is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        if channel_table.n_shanks != self.probe_info.num_shanks:
            logger.warning(
                "Channel table implies %d shanks but datapackage says %d; "
                "trusting channel table.",
                channel_table.n_shanks,
                self.probe_info.num_shanks,
            )
        self.channel_table = channel_table

    def validate_cached_stream(self, stream: EphysStreamData) -> None:
        """Validate a cached stream against the selected probe metadata."""
        if self.probe_info is None:
            raise RuntimeError("No probe selected — call select_probe() first")
        if stream.recording_id != self.probe_info.recording_id:
            raise ValueError(
                "Cached stream recording does not match selected recording: "
                f"{stream.recording_id!r} != {self.probe_info.recording_id!r}"
            )
        if stream.ephys_collection != self.probe_info.ephys_collection:
            raise ValueError(
                "Cached stream collection does not match selected collection: "
                f"{stream.ephys_collection!r} != {self.probe_info.ephys_collection!r}"
            )

    @property
    def probe_id(self) -> str | None:
        """Resolved selected-probe ID, if a probe is selected."""
        return self.probe_info.probe_id if self.probe_info is not None else None

    @property
    def n_shanks(self) -> int:
        """Number of shanks implied by attached channel metadata."""
        if self.channel_table is None:
            return 0
        return self.channel_table.n_shanks

    def shank_labels(self) -> list[str]:
        """Build user-facing shank labels for the selected stream."""
        n_shanks = self.n_shanks
        if n_shanks == 1:
            return ["1/1"]
        if n_shanks > 1:
            return [f"{idx + 1}/{n_shanks}" for idx in range(n_shanks)]
        return []
