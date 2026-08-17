"""Spike-derived ephys plot payload builders."""

from __future__ import annotations

import logging

import numpy as np
import scipy
from matplotlib import cm

from ephys_alignment_gui.geometry.numeric import bincount2D
from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry
from ephys_alignment_gui.plotting.level_policy import in_brain_depth_mask
from ephys_alignment_gui.plotting.raster_request import (
    DEFAULT_IMAGE_RASTER_REQUEST,
    ImageRasterRequest,
)

logger = logging.getLogger(__name__)

AUTOCORR_BIN_SIZE = 0.25 / 1000
AUTOCORR_WIN_SIZE = 10 / 1000
FS = 30000


class SpikePlotDataBuilder:
    """Build spike and cluster plot payloads for one channel collection."""

    def __init__(
        self,
        data,
        geometry: PlotChannelGeometry,
        shank_idx: int,
    ) -> None:
        self.data = data
        self.geometry = geometry
        self.shank_idx = shank_idx
        self.spike_idx = np.array([], dtype=int)
        self.kp_idx = np.array([], dtype=bool)
        self.clust_id = np.array([], dtype=int)
        self.t_autocorr = np.array([])
        self.t_template = np.array([])
        self.max_spike_time = None
        self._shank_spike_indices = None
        self._fr_img_base_by_request = {}

        if self.data["spikes"]["exists"]:
            self.max_spike_time = np.max(self.data["spikes"]["times"])

        if self.data["clusters"]["exists"]:
            self._shank_spike_indices = np.where(
                self.data["spike_shanks"] == shank_idx
            )[0]
            self.filter_units("all")
            self.compute_timescales()

    @property
    def chn_min(self):
        return self.geometry.chn_min

    @property
    def chn_max(self):
        return self.geometry.chn_max

    def filter_units(self, subset: str) -> None:
        """Select spike indices for one unit-quality subset."""
        try:
            spikes = self.data["spikes"]
            clusters = self.data["clusters"]
            metrics = clusters["metrics"]
            spike_clusters = spikes["clusters"]

            if subset == "all":
                self.spike_idx = np.arange(spike_clusters.size)
            else:
                conditions = {}
                if "ks2_label" in metrics:
                    conditions["KS good"] = metrics.ks2_label == "good"
                    conditions["KS mua"] = metrics.ks2_label == "mua"
                if "label" in metrics:
                    conditions["IBL good"] = metrics.label == 1
                if "default_qc" in metrics:
                    conditions["aind_qc"] = metrics["default_qc"]
                if "unitrefine_label" in metrics:
                    conditions["unitrefine_sua"] = metrics["unitrefine_label"] == "sua"
                    conditions["unitrefine_neural"] = (
                        metrics["unitrefine_label"] != "noise"
                    )

                if subset in conditions:
                    mask = conditions[subset]
                    cluster_indices = np.where(mask)[0]
                    self.spike_idx = np.where(np.isin(spike_clusters, cluster_indices))[
                        0
                    ]
                else:
                    logger.warning(
                        "Unknown unit filter %r, returning all units",
                        subset,
                    )
                    self.spike_idx = np.arange(spike_clusters.size)
        except Exception:
            logger.warning(
                "%s metrics not found or invalid, returning all units instead",
                subset,
            )
            self.spike_idx = np.arange(self.data["spikes"]["clusters"].size)

        if self._shank_spike_indices is not None:
            self.spike_idx = np.intersect1d(self.spike_idx, self._shank_spike_indices)

        self.kp_idx = np.where(
            ~np.isnan(self.data["spikes"]["depths"][self.spike_idx])
            & ~np.isnan(self.data["spikes"]["amps"][self.spike_idx])
        )[0]
        self._fr_img_base_by_request.clear()

    def _valid_spike_indices(self):
        """Return selected spike row indices with finite depth/amplitude values."""
        return self.spike_idx[self.kp_idx]

    def get_depth_data_scatter(self):
        """Return time/depth spike-amplitude scatter payload."""
        if not self.data["spikes"]["exists"]:
            return None

        valid_idx = self._valid_spike_indices()
        spike_amps = self.data["spikes"]["amps"][valid_idx]
        A_BIN = 10
        amp_range = np.quantile(spike_amps, [0, 0.9])
        amp_bins = np.linspace(amp_range[0], amp_range[1], A_BIN)
        colour_bin = np.linspace(0.0, 1.0, A_BIN + 1)
        colours = (
            (cm.get_cmap("BuPu")(colour_bin)[np.newaxis, :, :3][0]) * 255
        ).astype(np.int32)
        spikes_colours = np.empty(spike_amps.size, dtype=object)
        spikes_size = np.empty(spike_amps.size)
        for i_a in range(amp_bins.size):
            if i_a == (amp_bins.size - 1):
                idx = np.where(spike_amps > amp_bins[i_a])[0]
                spikes_colours[idx] = (64, 0, 128)
            else:
                idx = np.where(
                    (spike_amps > amp_bins[i_a]) & (spike_amps <= amp_bins[i_a + 1])
                )[0]
                spikes_colours[idx] = tuple(int(channel) for channel in colours[i_a])

            spikes_size[idx] = i_a / (A_BIN / 4)

        display_idx = valid_idx[0:-1:100]
        x = self.data["spikes"]["times"][display_idx]
        y = self.data["spikes"]["depths"][display_idx]
        return {
            "x": x,
            "y": y,
            "levels": amp_range * 1e6,
            "colours": spikes_colours[0:-1:100],
            "pen": None,
            "size": spikes_size[0:-1:100],
            "symbol": np.array("o"),
            "xrange": np.array([np.min(x), np.max(x)]),
            "xaxis": "Time (s)",
            "title": "Amplitude (uV)",
            "cmap": "BuPu",
            "cluster": False,
        }

    def get_fr_p2t_data_scatter(self):
        """Return cluster firing-rate, duration, and amplitude scatter payloads."""
        if not self.data["spikes"]["exists"]:
            return None, None, None

        valid_idx = self._valid_spike_indices()
        clu, spike_depths, spike_amps, n_spikes = self.compute_spike_average(
            self.data["spikes"]["clusters"][valid_idx],
            self.data["spikes"]["depths"][valid_idx],
            self.data["spikes"]["amps"][valid_idx],
        )
        spike_amps = spike_amps * 1e6
        fr = n_spikes / np.max(self.data["spikes"]["times"])

        data_fr_scatter = {
            "x": spike_amps,
            "y": spike_depths,
            "colours": fr,
            "pen": "k",
            "size": np.array(8),
            "symbol": np.array("o"),
            "levels": np.quantile(fr, [0, 1]),
            "xrange": np.array([0.9 * np.min(spike_amps), 1.1 * np.max(spike_amps)]),
            "xaxis": "Amplitude (uV)",
            "title": "Firing Rate (Sp/s)",
            "cmap": "hot",
            "cluster": True,
        }

        p2t = self.data["clusters"]["peakToTrough"][clu]
        data_p2t_scatter = {
            "x": spike_amps,
            "y": spike_depths,
            "colours": p2t,
            "pen": "k",
            "size": np.array(8),
            "symbol": np.array("o"),
            "levels": [-1.5, 1.5],
            "xrange": np.array([0.9 * np.min(spike_amps), 1.1 * np.max(spike_amps)]),
            "xaxis": "Amplitude (uV)",
            "title": "Peak to Trough duration (ms)",
            "cmap": "RdYlGn",
            "cluster": True,
        }

        data_amp_scatter = {
            "x": fr,
            "y": spike_depths,
            "colours": spike_amps,
            "pen": "k",
            "size": np.array(8),
            "symbol": np.array("o"),
            "levels": np.quantile(spike_amps, [0, 1]),
            "xrange": np.array([0.9 * np.min(fr), 1.1 * np.max(fr)]),
            "xaxis": "Firing Rate (Sp/s)",
            "title": "Amplitude (uV)",
            "cmap": "magma",
            "cluster": True,
        }

        return data_fr_scatter, data_p2t_scatter, data_amp_scatter

    def get_fr_img(
        self,
        in_brain_depths_um=None,
        *,
        raster_request: ImageRasterRequest | None = None,
    ):
        """Return time/depth firing-rate image payload."""
        base = self._fr_image_base(raster_request)
        if base is None:
            return None

        img = base["img"]
        depths = base["depths"]
        support = base["depth_support_mask"]
        col = in_brain_depth_mask(depths, in_brain_depths_um, bin_width=base["d_bin"])
        level_col = support if col is None else (col & support)
        if not level_col.any():
            level_col = support
        fr_by_depth = np.mean(img[:, level_col], axis=0)

        return {
            "img": img,
            "scale": base["scale"],
            "levels": np.quantile(fr_by_depth, [0, 1]),
            "offset": base["offset"],
            "xrange": base["xrange"],
            "xaxis": "Time (s)",
            "cmap": "binary",
            "title": "Firing Rate",
            "no_data_mask": np.broadcast_to(~support, img.shape),
            "no_data_color": (145, 158, 170, 210),
        }

    def _fr_image_base(
        self,
        raster_request: ImageRasterRequest | None = None,
    ):
        """Return the expensive firing-rate image arrays cached by unit filter."""
        raster_request = raster_request or DEFAULT_IMAGE_RASTER_REQUEST
        if raster_request in self._fr_img_base_by_request:
            return self._fr_img_base_by_request[raster_request]
        if not self.data["spikes"]["exists"]:
            return None

        valid_idx = self._valid_spike_indices()
        spike_times = self.data["spikes"]["times"][valid_idx]
        spike_depths = self.data["spikes"]["depths"][valid_idx]
        if spike_times.size == 0 or spike_depths.size == 0:
            return None

        chn_min, chn_max = self._spike_depth_extent()
        t_bin = raster_request.time_bin_s(float(np.ptp(spike_times)))
        d_bin = raster_request.depth_bin_um(float(chn_max - chn_min))
        n, times, depths = bincount2D(
            spike_times,
            spike_depths,
            t_bin,
            d_bin,
            ylim=[chn_min, chn_max],
        )
        img = n.T / t_bin
        xscale = (times[-1] - times[0]) / img.shape[0]
        yscale = (depths[-1] - depths[0]) / img.shape[1]
        depth_support_mask = self._depth_support_mask(depths, d_bin)

        self._fr_img_base_by_request[raster_request] = {
            "img": img,
            "scale": np.array([xscale, yscale]),
            "offset": np.array([0, np.min(depths)]),
            "xrange": np.array([times[0], times[-1]]),
            "depths": depths,
            "d_bin": d_bin,
            "depth_support_mask": depth_support_mask,
        }
        return self._fr_img_base_by_request[raster_request]

    def _depth_support_mask(self, depths, bin_width_um: float):
        """Return depth bins supported by the selected channel geometry."""
        depths = np.asarray(depths)
        chn_coords = np.asarray(getattr(self.geometry, "chn_coords", []))
        if chn_coords.ndim != 2 or chn_coords.shape[1] < 2:
            return np.ones(depths.size, dtype=bool)

        channel_depths = chn_coords[:, 1]
        channel_depths = np.unique(channel_depths[np.isfinite(channel_depths)])
        if channel_depths.size == 0:
            return np.ones(depths.size, dtype=bool)

        if channel_depths.size > 1:
            spacing = np.diff(channel_depths)
            positive_spacing = spacing[spacing > 0]
            pitch = (
                float(np.min(positive_spacing))
                if positive_spacing.size
                else float(bin_width_um)
            )
        else:
            pitch = float(getattr(self.geometry, "chn_diff", bin_width_um))

        support_radius_um = max(float(bin_width_um) / 2.0, pitch / 2.0)
        distance_to_channel = np.min(
            np.abs(depths[:, np.newaxis] - channel_depths[np.newaxis, :]),
            axis=1,
        )
        return distance_to_channel <= support_radius_um

    def get_fr_amp_data_line(self):
        """Return firing-rate and amplitude depth-profile line payloads."""
        if not self.data["spikes"]["exists"]:
            return None, None

        t_bin = np.max(self.data["spikes"]["times"])
        d_bin = 10
        chn_min, chn_max = self._spike_depth_extent()
        valid_idx = self._valid_spike_indices()
        spike_times = self.data["spikes"]["times"][valid_idx]
        spike_depths = self.data["spikes"]["depths"][valid_idx]
        spike_amps = self.data["spikes"]["amps"][valid_idx]
        nspikes, _times, depths = bincount2D(
            spike_times,
            spike_depths,
            t_bin,
            d_bin,
            ylim=[chn_min, chn_max],
        )

        amp, _times, depths = bincount2D(
            spike_amps,
            spike_depths,
            t_bin,
            d_bin,
            ylim=[chn_min, chn_max],
            weights=spike_amps,
        )
        mean_fr = nspikes[:, 0] / t_bin
        mean_amp = np.zeros_like(amp[:, 0], dtype=float)
        np.divide(
            amp[:, 0],
            nspikes[:, 0],
            out=mean_amp,
            where=nspikes[:, 0] > 0,
        )
        mean_amp *= 1e6
        mean_amp[np.where(nspikes[:, 0] < 50)[0]] = 0

        return (
            {
                "x": mean_fr,
                "y": depths,
                "xrange": np.array([0, np.max(mean_fr)]),
                "xaxis": "Firing Rate (Sp/s)",
            },
            {
                "x": mean_amp,
                "y": depths,
                "xrange": np.array([0, np.max(mean_amp)]),
                "xaxis": "Amplitude (uV)",
            },
        )

    def get_spike_correlation_data_img(self, in_brain_depths_um=None):
        """Return depth-bin spike count correlation image payload."""
        if not self.data["spikes"]["exists"]:
            return None

        t_bin = 0.05
        d_bin = 40
        chn_min, chn_max = self._spike_depth_extent()
        valid_idx = self._valid_spike_indices()
        counts, _times, depths = bincount2D(
            self.data["spikes"]["times"][valid_idx],
            self.data["spikes"]["depths"][valid_idx],
            t_bin,
            d_bin,
            ylim=[chn_min, chn_max],
        )
        corr = np.corrcoef(counts)
        corr[np.isnan(corr)] = 0
        np.fill_diagonal(corr, 0)
        scale = (np.max(depths) - np.min(depths)) / corr.shape[0]
        col = in_brain_depth_mask(depths, in_brain_depths_um, bin_width=d_bin)
        corr_lvl = corr if col is None else corr[np.ix_(col, col)]
        return {
            "img": corr,
            "scale": np.array([scale, scale]),
            "levels": np.array([np.min(corr_lvl), np.max(corr_lvl)]),
            "offset": np.array([self.chn_min, self.chn_min]),
            "xrange": np.array([self.chn_min, self.chn_max]),
            "cmap": "viridis",
            "title": "Correlation",
            "xaxis": "Distance from probe tip (µm)",
        }

    def get_autocorr(self, clust_idx):
        """Return autocorrelogram and cluster id for a clicked cluster."""
        from brainbox.population.decode import xcorr

        idx = np.where(self.data["spikes"]["clusters"] == self.clust_id[clust_idx])[0]
        autocorr = xcorr(
            self.data["spikes"]["times"][idx],
            self.data["spikes"]["clusters"][idx],
            AUTOCORR_BIN_SIZE,
            AUTOCORR_WIN_SIZE,
        )

        return autocorr[0, 0, :], self.data["clusters"].metrics.cluster_id[
            self.clust_id[clust_idx]
        ]

    def get_template_wf(self, clust_idx):
        """Return the primary template waveform for a clicked cluster."""
        template_wf = self.data["clusters"]["waveforms"][self.clust_id[clust_idx], :, 0]
        return template_wf * 1e6

    def compute_spike_average(self, spike_clusters, spike_depth, spike_amp):
        """Return per-cluster average depth, amplitude, and spike counts."""
        clust, inverse, counts = np.unique(
            spike_clusters,
            return_inverse=True,
            return_counts=True,
        )
        spike_depth_sum = scipy.sparse.csr_matrix(
            (spike_depth, (inverse, np.zeros(inverse.size, dtype=int)))
        )
        spike_amp_sum = scipy.sparse.csr_matrix(
            (spike_amp, (inverse, np.zeros(inverse.size, dtype=int)))
        )
        spike_depth_avg = np.ravel(spike_depth_sum.toarray()) / counts
        spike_amp_avg = np.ravel(spike_amp_sum.toarray()) / counts
        self.clust_id = clust
        return clust, spike_depth_avg, spike_amp_avg, counts

    def compute_timescales(self) -> None:
        """Compute cluster-detail time axes."""
        self.t_autocorr = 1e3 * np.arange(
            (AUTOCORR_WIN_SIZE / 2) - AUTOCORR_WIN_SIZE,
            (AUTOCORR_WIN_SIZE / 2) + AUTOCORR_BIN_SIZE,
            AUTOCORR_BIN_SIZE,
        )
        n_template = self.data["clusters"]["waveforms"][0, :, 0].size
        self.t_template = 1e3 * (np.arange(n_template)) / FS

    def _spike_depth_extent(self):
        valid_idx = self._valid_spike_indices()
        spike_depths = self.data["spikes"]["depths"][valid_idx]
        return (
            np.min(
                np.r_[
                    self.chn_min,
                    spike_depths,
                ]
            ),
            np.max(
                np.r_[
                    self.chn_max,
                    spike_depths,
                ]
            ),
        )
