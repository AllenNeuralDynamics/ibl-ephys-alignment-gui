"""Raw AP activity line and probe plot payload builder.

The producer measures one event rate per contact, so the probe plot shows
each contact and the line plot averages contacts sharing a depth. Rejected
contacts are NaN and stay blank rather than reading as silent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ephys_alignment_gui.plotting.array_utils import (
    average_equal_depth_channels,
    safe_take,
)
from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry
from ephys_alignment_gui.plotting.level_policy import probe_colour_levels
from ephys_alignment_gui.plotting.probe_bank_layout import (
    BNK_SIZE,
    arrange_channels_to_banks,
)

TITLE = "Raw Activity (Hz)"


class RawActivityPlotDataBuilder:
    """Build raw-activity line and probe plot payloads."""

    def __init__(self, data, geometry: PlotChannelGeometry) -> None:
        self.data = data
        self.geometry = geometry

    def rates_hz(self) -> np.ndarray | None:
        """Event rate of each channel on this shank, or None without any."""
        entry = self.data.get("raw_activity")
        if not entry or not entry.get("exists", False):
            return None
        rates = np.asarray(
            safe_take(np.asarray(entry["rate_hz"]), self.geometry.chn_ind),
            dtype=float,
        )
        return rates if np.isfinite(rates).any() else None

    def build(self, in_brain_depths_um=None) -> tuple[Any, Any]:
        """Return the line and probe payloads, or ``(None, None)``."""
        rates = self.rates_hz()
        if rates is None:
            return None, None
        depths = self.geometry.chn_coords[:, 1]

        profile = average_equal_depth_channels(rates[np.newaxis, :], depths)[0]
        data_line = {
            "x": profile,
            "y": np.unique(depths),
            "xrange": np.array([0.0, max(float(np.nanmax(profile)), 1.0)]),
            "xaxis": TITLE,
        }

        probe_img, probe_scale, probe_offset = arrange_channels_to_banks(
            rates,
            self.geometry,
        )
        data_probe = {
            "img": probe_img,
            "scale": probe_scale,
            "offset": probe_offset,
            "levels": probe_colour_levels(
                rates,
                channel_depths_um=depths,
                in_brain_depths_um=in_brain_depths_um,
            ),
            "cmap": "viridis",
            "xrange": np.array([0 * BNK_SIZE, self.geometry.n_banks * BNK_SIZE]),
            "title": TITLE,
        }
        return data_line, data_probe
