"""Qt-free loader for selected ephys stream runtime data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.ephys_data_service import (
    ChannelCollectionView,
    EphysDataService,
    EphysStreamData,
)


@dataclass(frozen=True)
class LoadedEphysSelection:
    """Runtime ephys data for one loaded stream and active shank."""

    stream: EphysStreamData
    channel_collection: ChannelCollectionView

    @property
    def ephys_dir(self) -> Path:
        """Directory containing the loaded ephys ALF data."""
        return self.stream.ephys_dir

    @property
    def depths(self) -> NDArray:
        """Channel depths for the active shank/channel collection."""
        return self.channel_collection.depths

    @property
    def session_notes(self) -> str:
        """Loaded session notes."""
        return self.stream.session_notes

    @property
    def alf_data(self) -> dict[str, Any]:
        """Loaded ALF object data."""
        return self.stream.alf_data


class EphysStreamLoader:
    """Load runtime ephys data for the currently selected probe.

    This loader owns the Qt-free IO part of "Load Data": resolving the selected
    probe, loading stream-level ALF data, and selecting the active shank view.
    UI teardown, progress messages, histology rendering, and plot updates stay
    in the Qt layer for now.
    """

    def __init__(
        self,
        data_context: AlignmentDataContext,
        ephys_data_service: EphysDataService,
    ) -> None:
        self.data_context = data_context
        self.ephys_data_service = ephys_data_service

    def load(self, shank_idx: int) -> LoadedEphysSelection:
        """Load stream data for the selected probe and return an active shank view."""
        probe = self.data_context.probe_info
        if probe is None:
            raise RuntimeError("No probe selected. Please select a probe first.")
        channel_table = self.data_context.channel_table
        if channel_table is None:
            raise RuntimeError("Channel info not loaded. Please select a probe first.")

        stream = self.ephys_data_service.load_stream_data(
            probe,
            channel_table=channel_table,
        )
        return self.from_stream(stream, shank_idx)

    def from_stream(
        self,
        stream: EphysStreamData,
        shank_idx: int,
    ) -> LoadedEphysSelection:
        """Build an active shank view from an already-loaded stream."""
        self.data_context.validate_cached_stream(stream)
        self.data_context.attach_channel_table(stream.channel_table)
        return LoadedEphysSelection(
            stream=stream,
            channel_collection=stream.channel_collection(shank_idx),
        )
