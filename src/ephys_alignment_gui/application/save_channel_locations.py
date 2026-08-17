"""Build save-time channel locations without full ephys stream runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.application.save_geometry_catalog import SaveGeometry
from ephys_alignment_gui.geometry.ephys_alignment import EphysAlignment
from ephys_alignment_gui.services.probe_track import ProbeTrackService


class SaveChannelLocationError(RuntimeError):
    """Raised when save channel locations cannot be derived."""


@dataclass(frozen=True)
class AlignmentSaveChannelLocationBuilder:
    """Compute saved channel locations from cached metadata and track picks."""

    probe_track_service: ProbeTrackService
    alignment_cls: Any = EphysAlignment

    def compute(
        self,
        *,
        geometry: SaveGeometry,
        feature: Any,
        track: Any,
        brain_atlas: Any,
    ) -> NDArray[Any]:
        """Return RAS channel locations for one saveable alignment state."""
        if brain_atlas is None:
            raise SaveChannelLocationError(
                "Brain atlas is not loaded, cannot prepare alignment save input."
            )
        try:
            track_annotations_ras = self.probe_track_service.load_track_annotations(
                probe=geometry.probe,
                shank_idx=geometry.key.shank_idx,
                brain_atlas=brain_atlas,
            )
            ephysalign = self.alignment_cls(
                track_annotations_ras=track_annotations_ras,
                chn_depths=geometry.channel_depths_um,
                brain_atlas=brain_atlas,
            )
            channel_locations_ras = np.asarray(
                ephysalign.get_channel_locations(feature, track),
                dtype=float,
            )
        except Exception as exc:
            raise SaveChannelLocationError(
                "Failed to compute save channel locations for "
                f"{geometry.key.recording_id}/{geometry.key.ephys_collection} "
                f"shank {geometry.key.shank_idx + 1}: {exc}"
            ) from exc

        expected_shape = (len(geometry.channel_coordinates), 3)
        if channel_locations_ras.shape != expected_shape:
            raise SaveChannelLocationError(
                "Save channel locations for "
                f"{geometry.key.recording_id}/{geometry.key.ephys_collection} "
                f"shank {geometry.key.shank_idx + 1} have shape "
                f"{channel_locations_ras.shape}, expected {expected_shape}."
            )
        return channel_locations_ras
