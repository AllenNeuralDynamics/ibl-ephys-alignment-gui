"""Factory for plot computation objects built from runtime stream views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.ephys_data_service import (
    ChannelCollectionView,
    EphysStreamData,
)
from ephys_alignment_gui.plot_data import PlotData


@dataclass(frozen=True)
class PlotDataFactory:
    """Build :class:`PlotData` from stream-owned runtime data."""

    def build(self, collection: ChannelCollectionView) -> PlotData:
        """Build plot data for a shank/channel-collection view."""
        stream = collection.stream
        return PlotData(
            stream.ephys_dir,
            stream.alf_data,
            collection.shank_idx,
            channel_collection=collection,
        )

    def build_for_stream(
        self,
        stream: EphysStreamData,
        shank_idx: int,
    ) -> PlotData:
        """Build plot data for one shank in a stream."""
        return self.build(stream.channel_collection(shank_idx))

    def build_legacy(
        self, probe_path: Path, data: dict[str, Any], shank_idx: int
    ) -> PlotData:
        """Build plot data from the legacy constructor inputs."""
        return PlotData(probe_path, data, shank_idx)
