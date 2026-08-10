"""Probe-track loading and coordinate normalization."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.io.datapackage_loader import ProbeInfo


class ProbeTrackService:
    """Load shank track picks and rotate them into canonical atlas space."""

    def load_track_annotations(
        self,
        *,
        probe: ProbeInfo,
        shank_idx: int,
        brain_atlas: Any,
    ) -> NDArray[np.floating]:
        """Read xyz-picks for one shank and return canonical RAS coordinates."""
        picks = probe.picks_for_shank(shank_idx)
        path = picks.image_space
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing probe trajectory file: {path}. "
                "This file must contain probe insertion coordinates in image space."
            )
        with open(path) as f:
            user_picks = json.load(f)

        # xyz_picks on disk are SPIM-native image-space RAS, in microns. The
        # GUI operates in the rotated canonical frame.
        track_annotations_ras_spim = np.array(user_picks["xyz_picks"]) / 1e6
        return brain_atlas.rotate_to_canonical(track_annotations_ras_spim)
