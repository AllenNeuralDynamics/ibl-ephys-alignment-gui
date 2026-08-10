"""RMS image/probe plot payload builder."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.plotting.array_utils import safe_take
from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry
from ephys_alignment_gui.plotting.level_policy import (
    in_brain_depth_mask,
    probe_colour_levels,
)
from ephys_alignment_gui.plotting.probe_bank_layout import (
    BNK_SIZE,
    arrange_channels_to_banks,
)


class RmsPlotDataBuilder:
    """Build RMS image and probe plot payloads."""

    def __init__(self, data, geometry: PlotChannelGeometry) -> None:
        self.data = data
        self.geometry = geometry

    def build(self, format: str, in_brain_depths_um=None):
        """Return image and probe payloads for one RMS format."""
        entry = self.data[f"rms_{format}"]
        if not entry["exists"]:
            return None, None

        rms = safe_take(entry["rms"], self.geometry.chn_ind, axis=1)
        img = self._average_equal_depth_channels(rms * 1e6)
        median = np.nanmean(np.nanmedian(img, axis=1))
        img = (img - np.nanmedian(img, axis=1, keepdims=True)) + median

        img_full = np.full((img.shape[0], self.geometry.chn_full.shape[0]), np.nan)
        img_full[:, self.geometry.idx_full] = img

        unique_depths = np.unique(self.geometry.chn_coords[:, 1])
        col = in_brain_depth_mask(unique_depths, in_brain_depths_um)
        levels = np.nanquantile(img if col is None else img[:, col], [0.1, 0.9])
        xscale = (entry["timestamps"][-1] - entry["timestamps"][0]) / img_full.shape[0]
        yscale = (self.geometry.chn_max - self.geometry.chn_min) / img_full.shape[1]

        cmap = "plasma" if format == "AP" else "inferno"

        data_img = {
            "img": img_full,
            "scale": np.array([xscale, yscale]),
            "levels": levels,
            "offset": np.array([0, self.geometry.chn_min]),
            "cmap": cmap,
            "xrange": np.array([entry["timestamps"][0], entry["timestamps"][-1]]),
            "xaxis": entry["xaxis"],
            "title": format + " RMS (uV)",
        }

        rms_avg = safe_take(
            np.mean(entry["rms"], axis=0),
            indices=self.geometry.chn_ind,
        ) * 1e6
        probe_levels = probe_colour_levels(
            rms_avg,
            channel_depths_um=self.geometry.chn_coords[:, 1],
            in_brain_depths_um=in_brain_depths_um,
        )
        probe_img, probe_scale, probe_offset = arrange_channels_to_banks(
            rms_avg,
            self.geometry,
        )

        data_probe = {
            "img": probe_img,
            "scale": probe_scale,
            "offset": probe_offset,
            "levels": probe_levels,
            "cmap": cmap,
            "xrange": np.array([0 * BNK_SIZE, (self.geometry.n_banks) * BNK_SIZE]),
            "title": format + " RMS (uV)",
        }

        return data_img, data_probe

    def _average_equal_depth_channels(self, values):
        _, chn_depth, chn_count = np.unique(
            self.geometry.chn_coords[:, 1],
            return_index=True,
            return_counts=True,
        )
        chn_depth_eq = np.copy(chn_depth)
        chn_depth_eq[np.where(chn_count == 2)] += 1

        def avg_chn_depth(row):
            return np.mean([row[chn_depth], row[chn_depth_eq]], axis=0)

        return np.apply_along_axis(avg_chn_depth, 1, values)
