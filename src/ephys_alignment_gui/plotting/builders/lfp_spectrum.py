"""LFP spectrum image/probe plot payload builder."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

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

FREQ_BANDS = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))


class LfpSpectrumPlotDataBuilder:
    """Build LFP spectrum image and probe plot payloads."""

    def __init__(self, data, geometry: PlotChannelGeometry) -> None:
        self.data = data
        self.geometry = geometry

    def build(self, format: str, in_brain_depths_um=None):
        """Return image and probe payloads for one PSD format."""
        data_probe = {}
        if not self.data[f"psd_{format}"]["exists"]:
            for freq in FREQ_BANDS:
                data_probe.update({f"{freq[0]} - {freq[1]} Hz": None})
            return None, data_probe

        data_img = self._build_spectrum_image(in_brain_depths_um)
        for freq in FREQ_BANDS:
            data_probe.update(
                self._build_probe_band(freq, in_brain_depths_um)
            )
        return data_img, data_probe

    def _build_spectrum_image(self, in_brain_depths_um=None):
        freq_range = [0.5, 300]
        freq_idx = np.where(
            (self.data["psd_lf"]["freqs"] >= freq_range[0])
            & (self.data["psd_lf"]["freqs"] < freq_range[1])
        )[0]
        lfp = safe_take(
            self.data["psd_lf"]["power"][freq_idx],
            self.geometry.chn_ind,
            axis=1,
        )
        lfp_db = 10 * np.log10(np.maximum(lfp, 1e-20))
        lfp_db -= np.median(lfp_db, axis=1, keepdims=True)
        img = self._average_equal_depth_channels(lfp_db)

        freqs_linear = self.data["psd_lf"]["freqs"][freq_idx]
        freqs_log = np.geomspace(freq_range[0], freq_range[1], num=img.shape[0])
        interp_fn = interp1d(
            freqs_linear,
            img,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        img_log = interp_fn(freqs_log)

        img_full = np.full((img_log.shape[0], self.geometry.chn_full.shape[0]), np.nan)
        img_full[:, self.geometry.idx_full] = img_log

        unique_depths = np.unique(self.geometry.chn_coords[:, 1])
        col = in_brain_depth_mask(unique_depths, in_brain_depths_um)
        level_src = img_log if col is None else img_log[:, col]
        finite_vals = level_src[np.isfinite(level_src)]
        max_abs = np.quantile(np.abs(finite_vals), 0.95) if len(finite_vals) > 0 else 1.0
        levels = np.array([-max_abs, max_abs])

        log_min = np.log10(freq_range[0])
        log_max = np.log10(freq_range[1])
        xscale = (log_max - log_min) / img_full.shape[0]
        yscale = (self.geometry.chn_max - self.geometry.chn_min) / img_full.shape[1]

        return {
            "img": img_full,
            "scale": np.array([xscale, yscale]),
            "levels": levels,
            "offset": np.array([log_min, self.geometry.chn_min]),
            "cmap": "RdBu_r",
            "xrange": np.array([log_min, log_max]),
            "xaxis": "Frequency (log10 Hz)",
            "title": "PSD deviation (dB)",
        }

    def _build_probe_band(self, freq, in_brain_depths_um=None):
        freq_idx = np.where(
            (self.data["psd_lf"]["freqs"] >= freq[0])
            & (self.data["psd_lf"]["freqs"] < freq[1])
        )[0]
        lfp_avg = safe_take(
            np.mean(self.data["psd_lf"]["power"][freq_idx], axis=0),
            self.geometry.chn_ind,
        )
        lfp_avg_db = 10 * np.log10(np.maximum(lfp_avg, 1e-20))
        probe_img, probe_scale, probe_offset = arrange_channels_to_banks(
            lfp_avg_db,
            self.geometry,
        )
        probe_levels = probe_colour_levels(
            lfp_avg_db,
            channel_depths_um=self.geometry.chn_coords[:, 1],
            in_brain_depths_um=in_brain_depths_um,
        )

        return {
            f"{freq[0]} - {freq[1]} Hz": {
                "img": probe_img,
                "scale": probe_scale,
                "offset": probe_offset,
                "levels": probe_levels,
                "cmap": "viridis",
                "xaxis": "Time (s)",
                "xrange": np.array([0 * BNK_SIZE, (self.geometry.n_banks) * BNK_SIZE]),
                "title": f"{freq[0]} - {freq[1]} Hz (dB)",
            }
        }

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
