"""Factory for plot computation objects built from runtime stream views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.plotting.payload_cache import EphysPlotPayloadCache
from ephys_alignment_gui.services.ephys_data import (
    ChannelCollectionView,
    EphysStreamData,
)


@dataclass(frozen=True)
class EphysPlotPayloadCacheFactory:
    """Build ephys plot payload caches from stream-owned runtime data."""

    def build(self, collection: ChannelCollectionView) -> EphysPlotPayloadCache:
        """Build plot payload cache for a shank/channel-collection view."""
        stream = collection.stream
        return EphysPlotPayloadCache(
            stream.ephys_dir,
            stream.alf_data,
            collection.shank_idx,
            channel_collection=collection,
        )

    def build_for_stream(
        self,
        stream: EphysStreamData,
        shank_idx: int,
    ) -> EphysPlotPayloadCache:
        """Build plot payload cache for one shank in a stream."""
        return self.build(stream.channel_collection(shank_idx))

    def build_legacy(
        self, probe_path: Path, data: dict[str, Any], shank_idx: int
    ) -> EphysPlotPayloadCache:
        """Build plot payload cache from the legacy constructor inputs."""
        return EphysPlotPayloadCache(probe_path, data, shank_idx)
