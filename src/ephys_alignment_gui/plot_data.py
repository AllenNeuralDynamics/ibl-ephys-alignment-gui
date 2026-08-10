import logging

import numpy as np

# from brainbox.io.spikeglx import Streamer
# from brainbox.population.decode import xcorr
# from brainbox.task import passive
# from neurodsp import voltage
# import neuropixel
import scipy
from matplotlib import cm
from numpy.typing import NDArray
from one.alf.io import AlfBunch
from pandas import DataFrame

from ephys_alignment_gui.ephys_data_service import ChannelCollectionView
from ephys_alignment_gui.lfp_correlation_plot_data import (
    LfpCorrelationPlotDataBuilder,
)
from ephys_alignment_gui.plot_channel_geometry import (
    PlotChannelGeometry,
    build_plot_channel_geometry,
)
from ephys_alignment_gui.plot_level_policy import (
    in_brain_depth_mask,
    probe_colour_levels,
)
from ephys_alignment_gui.utils import bincount2D

logger = logging.getLogger(__name__)

BNK_SIZE = 10
AUTOCORR_BIN_SIZE = 0.25 / 1000
AUTOCORR_WIN_SIZE = 10 / 1000

FS = 30000
np.seterr(divide="ignore", invalid="ignore")


def _safe_take(arr, indices, axis=0):
    """np.take that fills out-of-bounds positions with NaN.

    When channel indices exceed the data array (e.g. main-block RMS
    has fewer channels than the combined channel set), valid indices
    are taken normally and out-of-bounds positions are filled with NaN.
    """
    max_idx = arr.shape[axis] - 1
    oob = indices > max_idx
    if np.any(oob):
        logger.warning(
            f"Channel indices exceed data size "
            f"(max_idx={max_idx}, max_chn_ind={indices.max()}). "
            f"Filling {np.sum(oob)} channels with NaN."
        )
        safe_indices = np.clip(indices, 0, max_idx)
        result = np.take(arr, safe_indices, axis=axis).astype(float)
        # Build a slicer that targets the OOB positions along `axis`
        slices = [slice(None)] * result.ndim
        slices[axis] = oob
        result[tuple(slices)] = np.nan
        return result
    return np.take(arr, indices, axis=axis)


class PlotData:
    def __init__(
        self,
        probe_path,
        data,
        shank_idx,
        channel_collection: ChannelCollectionView | None = None,
    ) -> None:
        self.probe_path = probe_path
        self.data = data
        self.shank_idx = shank_idx
        self.channel_collection = channel_collection

        self.channel_geometry = build_plot_channel_geometry(
            data,
            shank_idx,
            channel_collection=channel_collection,
        )
        self._apply_channel_geometry(self.channel_geometry)

        # Depths (um) of channels currently inside the brain, per the active
        # alignment (set by the GUI after this PlotData is built). None means
        # "unknown" -> probe colour levels use all channels. Channels are placed
        # along the track purely by depth, so a per-depth set is exact.
        self.in_brain_depths_um = None

        # Per-instance memo of get_* datasets (see cached()). Cleared only when
        # the unit filter actually changes, since these datasets depend on the
        # spike/channel data, not on the alignment. Keeps shank revisits cheap.
        self._img_cache: dict = {}
        self._current_filter: str | None = None

        if self.data["spikes"]["exists"]:
            self.max_spike_time = np.max(self.data["spikes"]["times"])

        if self.data["clusters"]["exists"]:
            self._shank_spike_indices = np.where(
                self.data["spike_shanks"] == shank_idx
            )[0]
            self.filter_units("all")
            self.compute_timescales()
        else:
            self._shank_spike_indices = None
            self.spike_idx = np.array([], dtype=int)
            self.kp_idx = np.array([], dtype=bool)

        logger.debug(f"Spike idx: {self.spike_idx}")
        logger.debug(f"Keep idx: {self.kp_idx}")

    def _apply_channel_geometry(self, geometry: PlotChannelGeometry) -> None:
        """Expose derived channel geometry on the legacy PlotData attributes."""
        self.chn_coords_all = geometry.chn_coords_all
        self.chn_raw_ind_all = geometry.chn_raw_ind_all
        self.chn_contact_id_all = geometry.chn_contact_id_all
        self.chn_ind_all = geometry.chn_ind_all
        self.chn_shank_ind_all = geometry.chn_shank_ind_all
        self.chn_rows = geometry.chn_rows
        self.chn_coords = geometry.chn_coords
        self.chn_ind = geometry.chn_ind
        self.chn_min = geometry.chn_min
        self.chn_max = geometry.chn_max
        self.chn_diff = geometry.chn_diff
        self.chn_full = geometry.chn_full
        self.N_BNK = geometry.n_banks
        self.idx_full = geometry.idx_full

    def cached(self, method: str, args: tuple = ()):
        """Return ``self.<method>(*args)``, memoized per PlotData instance.

        Lets plot datasets be built lazily (only when displayed) yet at most
        once per shank. The cache is cleared by :meth:`filter_units` when the
        unit subset actually changes.
        """
        key = (method, args)
        if key not in self._img_cache:
            self._img_cache[key] = getattr(self, method)(*args)
        return self._img_cache[key]

    def filter_units(self, subset: str) -> None:
        # Idempotent: re-applying the current subset (as happens on every shank
        # switch) is a no-op, so the memo cache stays warm. A genuine change
        # clears it before recomputing spike_idx/kp_idx below.
        if subset == self._current_filter:
            return
        self._current_filter = subset
        self._img_cache.clear()
        try:
            # Pre-fetch commonly used data structures (avoid repeated indexing)
            spikes: AlfBunch = self.data["spikes"]
            clusters: AlfBunch = self.data["clusters"]
            metrics: DataFrame = clusters["metrics"]
            spike_clusters: NDArray = spikes["clusters"]

            if subset == "all":
                self.spike_idx = np.arange(spike_clusters.size)
            else:
                # Map unit type string to a boolean mask over cluster metrics
                # Each mask selects clusters to keep; we then map spikes via spike_clusters
                conditions: dict[str, NDArray] = {}
                # Kilosort labels
                if "ks2_label" in metrics:
                    conditions["KS good"] = metrics.ks2_label == "good"
                    conditions["KS mua"] = metrics.ks2_label == "mua"
                # IBL curation label (1 == good)
                if "label" in metrics:
                    conditions["IBL good"] = metrics.label == 1
                # AIND QC default flag (already boolean)
                if "default_qc" in metrics:
                    conditions["aind_qc"] = metrics["default_qc"]
                # UnitRefine labels
                if "unitrefine_label" in metrics:
                    conditions["unitrefine_sua"] = metrics["unitrefine_label"] == "sua"
                    conditions["unitrefine_neural"] = (
                        metrics["unitrefine_label"] != "noise"
                    )

                if subset in conditions:
                    mask = conditions[subset]
                    # Convert mask to cluster indices
                    cluster_indices = np.where(mask)[0]
                    # Select spikes whose cluster id is in the kept set
                    self.spike_idx = np.where(np.isin(spike_clusters, cluster_indices))[
                        0
                    ]
                else:
                    # Fallback if unknown type requested
                    logger.warning(
                        f"Unknown unit filter '{subset}', returning all units"
                    )
                    self.spike_idx = np.arange(spike_clusters.size)
        except Exception:
            logger.warning(
                f"{subset} metrics not found or invalid, returning all units instead"
            )
            self.spike_idx = np.arange(self.data["spikes"]["clusters"].size)

        # Restrict to current shank (multi-shank probes)
        if self._shank_spike_indices is not None:
            self.spike_idx = np.intersect1d(self.spike_idx, self._shank_spike_indices)

        # Filter for nans in depths and also in amps
        self.kp_idx = np.where(
            ~np.isnan(self.data["spikes"]["depths"][self.spike_idx])
            & ~np.isnan(self.data["spikes"]["amps"][self.spike_idx])
        )[0]

    # Plots that require spike and cluster data
    def get_depth_data_scatter(self):
        if not self.data["spikes"]["exists"]:
            data_scatter = None
            return data_scatter
        else:
            A_BIN = 10
            amp_range = np.quantile(
                self.data["spikes"]["amps"][self.spike_idx][self.kp_idx],
                [0, 0.9],
            )
            amp_bins = np.linspace(amp_range[0], amp_range[1], A_BIN)
            colour_bin = np.linspace(0.0, 1.0, A_BIN + 1)
            colours = (
                (cm.get_cmap("BuPu")(colour_bin)[np.newaxis, :, :3][0]) * 255
            ).astype(np.int32)
            spikes_colours = np.empty(
                self.data["spikes"]["amps"][self.spike_idx][self.kp_idx].size,
                dtype=object,
            )
            spikes_size = np.empty(
                self.data["spikes"]["amps"][self.spike_idx][self.kp_idx].size
            )
            for iA in range(amp_bins.size):
                if iA == (amp_bins.size - 1):
                    idx = np.where(
                        self.data["spikes"]["amps"][self.spike_idx][self.kp_idx]
                        > amp_bins[iA]
                    )[0]
                    # Make saturated spikes a very dark purple
                    spikes_colours[idx] = (64, 0, 128)
                else:
                    idx = np.where(
                        (
                            self.data["spikes"]["amps"][self.spike_idx][self.kp_idx]
                            > amp_bins[iA]
                        )
                        & (
                            self.data["spikes"]["amps"][self.spike_idx][self.kp_idx]
                            <= amp_bins[iA + 1]
                        )
                    )[0]
                    spikes_colours[idx] = tuple(int(channel) for channel in colours[iA])

                spikes_size[idx] = iA / (A_BIN / 4)

            data_scatter = {
                "x": self.data["spikes"]["times"][self.spike_idx][self.kp_idx][
                    0:-1:100
                ],
                "y": self.data["spikes"]["depths"][self.spike_idx][self.kp_idx][
                    0:-1:100
                ],
                "levels": amp_range * 1e6,
                "colours": spikes_colours[0:-1:100],
                "pen": None,
                "size": spikes_size[0:-1:100],
                "symbol": np.array("o"),
                "xrange": np.array(
                    [
                        np.min(
                            self.data["spikes"]["times"][self.spike_idx][self.kp_idx][
                                0:-1:100
                            ]
                        ),
                        np.max(
                            self.data["spikes"]["times"][self.spike_idx][self.kp_idx][
                                0:-1:100
                            ]
                        ),
                    ]
                ),
                "xaxis": "Time (s)",
                "title": "Amplitude (uV)",
                "cmap": "BuPu",
                "cluster": False,
            }

            return data_scatter

    def get_fr_p2t_data_scatter(self):
        if not self.data["spikes"]["exists"]:
            data_fr_scatter = None
            data_p2t_scatter = None
            data_amp_scatter = None
            return data_fr_scatter, data_p2t_scatter, data_amp_scatter
        else:
            (clu, spike_depths, spike_amps, n_spikes) = self.compute_spike_average(
                (self.data["spikes"]["clusters"][self.spike_idx][self.kp_idx]),
                (self.data["spikes"]["depths"][self.spike_idx][self.kp_idx]),
                (self.data["spikes"]["amps"][self.spike_idx][self.kp_idx]),
            )
            spike_amps = spike_amps * 1e6
            fr = n_spikes / np.max(self.data["spikes"]["times"])
            fr_levels = np.quantile(fr, [0, 1])

            data_fr_scatter = {
                "x": spike_amps,
                "y": spike_depths,
                "colours": fr,
                "pen": "k",
                "size": np.array(8),
                "symbol": np.array("o"),
                "levels": fr_levels,
                "xrange": np.array(
                    [0.9 * np.min(spike_amps), 1.1 * np.max(spike_amps)]
                ),
                "xaxis": "Amplitude (uV)",
                "title": "Firing Rate (Sp/s)",
                "cmap": "hot",
                "cluster": True,
            }

            p2t = self.data["clusters"]["peakToTrough"][clu]

            # Define the p2t levels so always same colourbar across sessions
            p2t_levels = [-1.5, 1.5]
            data_p2t_scatter = {
                "x": spike_amps,
                "y": spike_depths,
                "colours": p2t,
                "pen": "k",
                "size": np.array(8),
                "symbol": np.array("o"),
                "levels": p2t_levels,
                "xrange": np.array(
                    [0.9 * np.min(spike_amps), 1.1 * np.max(spike_amps)]
                ),
                "xaxis": "Amplitude (uV)",
                "title": "Peak to Trough duration (ms)",
                "cmap": "RdYlGn",
                "cluster": True,
            }

            spike_amps_levels = np.quantile(spike_amps, [0, 1])

            data_amp_scatter = {
                "x": fr,
                "y": spike_depths,
                "colours": spike_amps,
                "pen": "k",
                "size": np.array(8),
                "symbol": np.array("o"),
                "levels": spike_amps_levels,
                "xrange": np.array([0.9 * np.min(fr), 1.1 * np.max(fr)]),
                "xaxis": "Firing Rate (Sp/s)",
                "title": "Amplitude (uV)",
                "cmap": "magma",
                "cluster": True,
            }

            return data_fr_scatter, data_p2t_scatter, data_amp_scatter

    def get_fr_img(self):
        if not self.data["spikes"]["exists"]:
            data_img = None
            return data_img
        else:
            T_BIN = 0.05
            D_BIN = 5
            chn_min = np.min(
                np.r_[
                    self.chn_min,
                    self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                ]
            )
            chn_max = np.max(
                np.r_[
                    self.chn_max,
                    self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                ]
            )
            n, times, depths = bincount2D(
                self.data["spikes"]["times"][self.spike_idx][self.kp_idx],
                self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                T_BIN,
                D_BIN,
                ylim=[chn_min, chn_max],
            )
            img = n.T / T_BIN
            xscale = (times[-1] - times[0]) / img.shape[0]
            yscale = (depths[-1] - depths[0]) / img.shape[1]

            # img columns are D_BIN-wide depth bins; constrain the colour range
            # to bins containing in-brain channels (emitted img unchanged).
            col = in_brain_depth_mask(
                depths,
                self.in_brain_depths_um,
                bin_width=D_BIN,
            )
            fr_by_depth = np.mean(img if col is None else img[:, col], axis=0)

            data_img = {
                "img": img,
                "scale": np.array([xscale, yscale]),
                "levels": np.quantile(fr_by_depth, [0, 1]),
                "offset": np.array([0, np.min(depths)]),
                "xrange": np.array([times[0], times[-1]]),
                "xaxis": "Time (s)",
                "cmap": "binary",
                "title": "Firing Rate",
            }

            return data_img

    def get_fr_amp_data_line(self):
        if not self.data["spikes"]["exists"]:
            data_fr_line = None
            data_amp_line = None
            return data_fr_line, data_amp_line
        else:
            T_BIN = np.max(self.data["spikes"]["times"])
            D_BIN = 10
            chn_min = np.min(
                np.r_[
                    self.chn_min,
                    self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                ]
            )
            chn_max = np.max(
                np.r_[
                    self.chn_max,
                    self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                ]
            )
            nspikes, times, depths = bincount2D(
                self.data["spikes"]["times"][self.spike_idx][self.kp_idx],
                self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                T_BIN,
                D_BIN,
                ylim=[chn_min, chn_max],
            )

            amp, times, depths = bincount2D(
                self.data["spikes"]["amps"][self.spike_idx][self.kp_idx],
                self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                T_BIN,
                D_BIN,
                ylim=[chn_min, chn_max],
                weights=self.data["spikes"]["amps"][self.spike_idx][self.kp_idx],
            )
            mean_fr = nspikes[:, 0] / T_BIN
            mean_amp = np.divide(amp[:, 0], nspikes[:, 0]) * 1e6
            mean_amp[np.isnan(mean_amp)] = 0
            remove_bins = np.where(nspikes[:, 0] < 50)[0]
            mean_amp[remove_bins] = 0

            data_fr_line = {
                "x": mean_fr,
                "y": depths,
                "xrange": np.array([0, np.max(mean_fr)]),
                "xaxis": "Firing Rate (Sp/s)",
            }

            data_amp_line = {
                "x": mean_amp,
                "y": depths,
                "xrange": np.array([0, np.max(mean_amp)]),
                "xaxis": "Amplitude (uV)",
            }

            return data_fr_line, data_amp_line

    def get_spike_correlation_data_img(self):
        if not self.data["spikes"]["exists"]:
            data_img = None
            return data_img
        else:
            T_BIN = 0.05
            D_BIN = 40
            chn_min = np.min(
                np.r_[
                    self.chn_min,
                    self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                ]
            )
            chn_max = np.max(
                np.r_[
                    self.chn_max,
                    self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                ]
            )
            R, _, depths = bincount2D(
                self.data["spikes"]["times"][self.spike_idx][self.kp_idx],
                self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                T_BIN,
                D_BIN,
                ylim=[chn_min, chn_max],
            )
            corr = np.corrcoef(R)
            corr[np.isnan(corr)] = 0
            np.fill_diagonal(corr, 0)
            scale = (np.max(depths) - np.min(depths)) / corr.shape[0]
            # corr is (depth-bin x depth-bin); constrain the colour range to the
            # in-brain x in-brain sub-block (emitted matrix unchanged).
            col = in_brain_depth_mask(
                depths,
                self.in_brain_depths_um,
                bin_width=D_BIN,
            )
            corr_lvl = corr if col is None else corr[np.ix_(col, col)]
            data_img = {
                "img": corr,
                "scale": np.array([scale, scale]),
                "levels": np.array([np.min(corr_lvl), np.max(corr_lvl)]),
                "offset": np.array([self.chn_min, self.chn_min]),
                "xrange": np.array([self.chn_min, self.chn_max]),
                "cmap": "viridis",
                "title": "Correlation",
                "xaxis": "Distance from probe tip (um)",
            }
            return data_img

    def get_rms_data_img_probe(self, format):
        # Finds channels that are at equivalent depth on probe and averages rms values for each
        # time point at same depth togehter

        if not self.data[f"rms_{format}"]["exists"]:
            data_img = None
            data_probe = None
            return data_img, data_probe

        _rms = _safe_take(
            self.data[f"rms_{format}"]["rms"],
            self.chn_ind,
            axis=1,
        )
        _, self.chn_depth, chn_count = np.unique(
            self.chn_coords[:, 1], return_index=True, return_counts=True
        )
        self.chn_depth_eq = np.copy(self.chn_depth)
        self.chn_depth_eq[np.where(chn_count == 2)] += 1

        def avg_chn_depth(a):
            return np.mean([a[self.chn_depth], a[self.chn_depth_eq]], axis=0)

        def get_median(a):
            return np.nanmedian(a)

        def median_subtract(a):
            return a - np.nanmedian(a)

        img = np.apply_along_axis(avg_chn_depth, 1, _rms * 1e6)
        median = np.nanmean(np.apply_along_axis(get_median, 1, img))
        # Medium subtract to remove bands, but add back average median so values make sense
        img = np.apply_along_axis(median_subtract, 1, img) + median

        img_full = np.full((img.shape[0], self.chn_full.shape[0]), np.nan)
        img_full[:, self.idx_full] = img

        # img columns are the unique channel depths (avg_chn_depth); constrain
        # the colour levels to in-brain depths so out-of-brain channels don't
        # blow out the range. Emitted img is unchanged; only levels narrow.
        unique_depths = np.unique(self.chn_coords[:, 1])
        col = in_brain_depth_mask(unique_depths, self.in_brain_depths_um)
        levels = np.nanquantile(img if col is None else img[:, col], [0.1, 0.9])
        xscale = (
            self.data[f"rms_{format}"]["timestamps"][-1]
            - self.data[f"rms_{format}"]["timestamps"][0]
        ) / img_full.shape[0]
        yscale = (self.chn_max - self.chn_min) / img_full.shape[1]

        if format == "AP":
            cmap = "plasma"
        else:
            cmap = "inferno"

        data_img = {
            "img": img_full,
            "scale": np.array([xscale, yscale]),
            "levels": levels,
            "offset": np.array([0, self.chn_min]),
            "cmap": cmap,
            "xrange": np.array(
                [
                    self.data[f"rms_{format}"]["timestamps"][0],
                    self.data[f"rms_{format}"]["timestamps"][-1],
                ]
            ),
            "xaxis": self.data[f"rms_{format}"]["xaxis"],
            "title": format + " RMS (uV)",
        }

        # Probe data
        rms_avg = (
            _safe_take(
                np.mean(self.data[f"rms_{format}"]["rms"], axis=0),
                indices=self.chn_ind,
            )
        ) * 1e6
        probe_levels = probe_colour_levels(
            rms_avg,
            channel_depths_um=self.chn_coords[:, 1],
            in_brain_depths_um=self.in_brain_depths_um,
        )
        probe_img, probe_scale, probe_offset = self.arrange_channels2banks(rms_avg)

        data_probe = {
            "img": probe_img,
            "scale": probe_scale,
            "offset": probe_offset,
            "levels": probe_levels,
            "cmap": cmap,
            "xrange": np.array([0 * BNK_SIZE, (self.N_BNK) * BNK_SIZE]),
            "title": format + " RMS (uV)",
        }

        return data_img, data_probe

    # only for IBL sorry
    def get_raw_data_image(self, pid, t0=(1000, 2000, 3000), one=None):
        def gain2level(gain):
            return 10 ** (gain / 20) * 4 * np.array([-1, 1])

        data_img = dict()

        times = [t for t in t0 if t < self.max_spike_time]

        for t in times:
            sr = Streamer(pid=pid, one=one, remove_cached=False, typ="ap")
            th = sr.geometry

            if sr.meta.get("NP2.4_shank", None) is not None:
                h = neuropixel.trace_header(sr.major_version, nshank=4)
                h = neuropixel.split_trace_header(
                    h, shank=int(sr.meta.get("NP2.4_shank"))
                )
            else:
                h = neuropixel.trace_header(
                    sr.major_version, nshank=np.unique(th["shank"]).size
                )
                idx = np.isin(h["ind"], th["ind"])
                for k in h.keys():
                    h[k] = h[k][idx]

            s0 = t * sr.fs
            tsel = slice(int(s0), int(s0) + int(1 * sr.fs))
            raw = sr[tsel, : -sr.nsync].T
            channel_labels, channel_features = voltage.detect_bad_channels(raw, sr.fs)
            raw = voltage.destripe(raw, fs=sr.fs, h=h, channel_labels=channel_labels)
            raw_image = raw[:, int((450 / 1e3) * sr.fs) : int((500 / 1e3) * sr.fs)].T
            x_range = np.array([0, raw_image.shape[0] - 1]) / sr.fs * 1e3
            levels = gain2level(-90)
            xscale = (x_range[1] - x_range[0]) / raw_image.shape[0]
            yscale = (self.chn_max - self.chn_min) / raw_image.shape[1]

            data_raw = {
                "img": raw_image,
                "scale": np.array([xscale, yscale]),
                "levels": levels,
                "offset": np.array([0, self.chn_min]),
                "cmap": "bone",
                "xrange": x_range,
                "xaxis": "Time (ms)",
                "title": "Power (uV)",
            }
            data_img[f"Raw data t={t}"] = data_raw

        return data_img

    def get_lfp_spectrum_data(self, format: str):
        freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))
        data_probe = {}

        if not self.data[f"psd_{format}"]["exists"]:
            data_img = None
            for freq in freq_bands:
                lfp_band_data = {f"{freq[0]} - {freq[1]} Hz": None}
                data_probe.update(lfp_band_data)

            return data_img, data_probe
        else:
            # Power spectrum image — log frequency, per-freq normalized
            freq_range = [0.5, 300]  # start at delta lower edge
            freq_idx = np.where(
                (self.data["psd_lf"]["freqs"] >= freq_range[0])
                & (self.data["psd_lf"]["freqs"] < freq_range[1])
            )[0]
            _lfp = _safe_take(
                self.data["psd_lf"]["power"][freq_idx],
                self.chn_ind,
                axis=1,
            )
            _lfp_dB = 10 * np.log10(np.maximum(_lfp, 1e-20))

            # Per-frequency normalization: subtract channel median
            # to remove 1/f slope and highlight spatial variation
            _lfp_dB -= np.median(_lfp_dB, axis=1, keepdims=True)

            _, self.chn_depth, chn_count = np.unique(
                self.chn_coords[:, 1], return_index=True, return_counts=True
            )
            self.chn_depth_eq = np.copy(self.chn_depth)
            self.chn_depth_eq[np.where(chn_count == 2)] += 1

            def avg_chn_depth(a):
                return np.mean([a[self.chn_depth], a[self.chn_depth_eq]], axis=0)

            img = np.apply_along_axis(avg_chn_depth, 1, _lfp_dB)

            # Resample to log-spaced frequency axis
            freqs_linear = self.data["psd_lf"]["freqs"][freq_idx]
            freqs_log = np.geomspace(freq_range[0], freq_range[1], num=img.shape[0])
            from scipy.interpolate import interp1d

            interp_fn = interp1d(
                freqs_linear,
                img,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            img_log = interp_fn(freqs_log)

            img_full = np.full((img_log.shape[0], self.chn_full.shape[0]), np.nan)
            img_full[:, self.idx_full] = img_log

            # img_log columns are the unique channel depths; constrain the
            # symmetric colour range to in-brain depths (emitted img unchanged).
            unique_depths = np.unique(self.chn_coords[:, 1])
            col = in_brain_depth_mask(unique_depths, self.in_brain_depths_um)
            level_src = img_log if col is None else img_log[:, col]
            finite_vals = level_src[np.isfinite(level_src)]
            if len(finite_vals) > 0:
                max_abs = np.quantile(np.abs(finite_vals), 0.95)
            else:
                max_abs = 1.0
            levels = np.array([-max_abs, max_abs])

            # Map to log10(freq) coordinates
            log_min = np.log10(freq_range[0])
            log_max = np.log10(freq_range[1])
            xscale = (log_max - log_min) / img_full.shape[0]
            yscale = (self.chn_max - self.chn_min) / img_full.shape[1]

            data_img = {
                "img": img_full,
                "scale": np.array([xscale, yscale]),
                "levels": levels,
                "offset": np.array([log_min, self.chn_min]),
                "cmap": "RdBu_r",
                "xrange": np.array([log_min, log_max]),
                "xaxis": "Frequency (log10 Hz)",
                "title": "PSD deviation (dB)",
            }

            # Power spectrum in bands on probe
            for freq in freq_bands:
                freq_idx = np.where(
                    (self.data["psd_lf"]["freqs"] >= freq[0])
                    & (self.data["psd_lf"]["freqs"] < freq[1])
                )[0]
                lfp_avg = _safe_take(
                    np.mean(self.data["psd_lf"]["power"][freq_idx], axis=0),
                    self.chn_ind,
                )
                lfp_avg_dB = 10 * np.log10(np.maximum(lfp_avg, 1e-20))
                probe_img, probe_scale, probe_offset = self.arrange_channels2banks(
                    lfp_avg_dB
                )
                probe_levels = probe_colour_levels(
                    lfp_avg_dB,
                    channel_depths_um=self.chn_coords[:, 1],
                    in_brain_depths_um=self.in_brain_depths_um,
                )

                lfp_band_data = {
                    f"{freq[0]} - {freq[1]} Hz": {
                        "img": probe_img,
                        "scale": probe_scale,
                        "offset": probe_offset,
                        "levels": probe_levels,
                        "cmap": "viridis",
                        "xaxis": "Time (s)",
                        "xrange": np.array([0 * BNK_SIZE, (self.N_BNK) * BNK_SIZE]),
                        "title": f"{freq[0]} - {freq[1]} Hz (dB)",
                    }
                }
                data_probe.update(lfp_band_data)

            return data_img, data_probe

    def get_lfp_correlation_data_img(self):
        """Load LFP correlation data from the band_corr folder."""
        return LfpCorrelationPlotDataBuilder(
            probe_path=self.probe_path,
            shank_idx=self.shank_idx,
            geometry=self.channel_geometry,
            in_brain_depths_um=self.in_brain_depths_um,
        ).build()

    def get_rfmap_data(self):
        data_img = dict()
        if not self.data["rf_map"]["exists"]:
            return data_img, None
        else:
            (rf_map_times, rf_map_pos, rf_stim_frames) = (
                passive.get_on_off_times_and_positions(self.data["rf_map"])
            )

            chn_min = np.min(
                np.r_[
                    self.chn_min,
                    self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                ]
            )
            chn_max = np.max(
                np.r_[
                    self.chn_max,
                    self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                ]
            )

            rf_map, _ = passive.get_rf_map_over_depth(
                rf_map_times,
                rf_map_pos,
                rf_stim_frames,
                self.data["spikes"]["times"][self.spike_idx][self.kp_idx],
                self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
                d_bin=160,
                y_lim=[chn_min, chn_max],
            )
            rfs_svd = passive.get_svd_map(rf_map)
            img = dict()
            img["on"] = np.vstack(rfs_svd["on"])
            img["off"] = np.vstack(rfs_svd["off"])
            yscale = (self.chn_max - self.chn_min) / img["on"].shape[0]

            xscale = 1
            levels = np.quantile(np.c_[img["on"], img["off"]], [0, 1])

            depths = np.linspace(self.chn_min, self.chn_max, len(rfs_svd["on"]) + 1)

            sub_type = ["on", "off"]
            for sub in sub_type:
                sub_data = {
                    sub: {
                        "img": [img[sub].T],
                        "scale": [np.array([xscale, yscale])],
                        "levels": levels,
                        "offset": [np.array([0, self.chn_min])],
                        "cmap": "viridis",
                        "xrange": np.array([0, 15]),
                        "xaxis": "Position",
                        "title": "rfmap (dB)",
                    }
                }
                data_img.update(sub_data)

            return data_img, depths

    def get_passive_events(self):
        stim_keys = ["valveOn", "toneOn", "noiseOn", "leftGabor", "rightGabor"]
        data_img = dict()
        if not self.data["pass_stim"]["exists"] and not self.data["gabor"]["exists"]:
            return data_img
        elif not self.data["pass_stim"]["exists"] and self.data["gabor"]["exists"]:
            stim_types = ["leftGabor", "rightGabor"]
            stims = self.data["gabor"]
        elif self.data["pass_stim"]["exists"] and not self.data["gabor"]["exists"]:
            stim_types = ["valveOn", "toneOn", "noiseOn"]
            stims = {
                stim_type: self.data["pass_stim"][stim_type] for stim_type in stim_types
            }
        else:
            stim_types = stim_keys
            stims = {
                stim_type: self.data["pass_stim"][stim_type]
                for stim_type in stim_types[0:3]
            }
            stims.update(self.data["gabor"])

        chn_min = np.min(
            np.r_[
                self.chn_min,
                self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
            ]
        )
        chn_max = np.max(
            np.r_[
                self.chn_max,
                self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
            ]
        )

        base_stim = 1
        pre_stim = 0.4
        post_stim = 1
        stim_events = passive.get_stim_aligned_activity(
            stims,
            self.data["spikes"]["times"][self.spike_idx][self.kp_idx],
            self.data["spikes"]["depths"][self.spike_idx][self.kp_idx],
            pre_stim=pre_stim,
            post_stim=post_stim,
            base_stim=base_stim,
            y_lim=[chn_min, chn_max],
        )

        for stim_type, z_score in stim_events.items():
            xscale = (post_stim + pre_stim) / z_score.shape[1]
            yscale = (self.chn_max - self.chn_min) / z_score.shape[0]

            levels = [-10, 10]

            stim_data = {
                stim_type: {
                    "img": z_score.T,
                    "scale": np.array([xscale, yscale]),
                    "levels": levels,
                    "offset": np.array([-1 * pre_stim, self.chn_min]),
                    "cmap": "bwr",
                    "xrange": [-1 * pre_stim, post_stim],
                    "xaxis": "Time from Stim Onset (s)",
                    "title": "Firing rate (z score)",
                }
            }
            data_img.update(stim_data)

        return data_img

    def get_autocorr(self, clust_idx):
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
        template_wf = self.data["clusters"]["waveforms"][self.clust_id[clust_idx], :, 0]
        return template_wf * 1e6

    def arrange_channels2banks(self, data):
        bnk_data = []
        bnk_scale = np.empty((self.N_BNK, 2))
        bnk_offset = np.empty((self.N_BNK, 2))
        for iX, x in enumerate(np.unique(self.chn_coords[:, 0])):
            bnk_idx = np.where(self.chn_coords[:, 0] == x)[0]

            bnk_ycoords = self.chn_coords[bnk_idx, 1]
            bnk_ycoords_unique = np.unique(bnk_ycoords)
            bnk_diff = np.min(np.abs(np.diff(bnk_ycoords_unique)))
            logger.debug(
                f"x={x}: bnk_diff={bnk_diff}, chn_diff={self.chn_diff}, "
                f"n_chns={len(bnk_ycoords)}"
            )
            bnk_full = np.arange(
                np.min(bnk_ycoords),
                np.max(bnk_ycoords) + bnk_diff,
                bnk_diff,
            )
            _bnk_vals = np.full((bnk_full.shape[0]), np.nan)
            idx_full = np.where(np.isin(bnk_full, bnk_ycoords_unique))[0]
            _bnk_vals[idx_full] = data[bnk_idx]

            _bnk_data = _bnk_vals[np.newaxis, :]

            _bnk_yscale = (self.chn_max - self.chn_min) / _bnk_data.shape[1]
            _bnk_xscale = BNK_SIZE / _bnk_data.shape[0]
            _bnk_yoffset = np.min(bnk_ycoords)
            _bnk_xoffset = BNK_SIZE * iX

            bnk_data.append(_bnk_data)
            bnk_scale[iX, :] = np.array([_bnk_xscale, _bnk_yscale])
            bnk_offset[iX, :] = np.array([_bnk_xoffset, _bnk_yoffset])

        return bnk_data, bnk_scale, bnk_offset

    def compute_spike_average(self, spike_clusters, spike_depth, spike_amp):
        clust, inverse, counts = np.unique(
            spike_clusters, return_inverse=True, return_counts=True
        )
        _spike_depth = scipy.sparse.csr_matrix(
            (spike_depth, (inverse, np.zeros(inverse.size, dtype=int)))
        )
        _spike_amp = scipy.sparse.csr_matrix(
            (spike_amp, (inverse, np.zeros(inverse.size, dtype=int)))
        )
        spike_depth_avg = np.ravel(_spike_depth.toarray()) / counts
        spike_amp_avg = np.ravel(_spike_amp.toarray()) / counts
        self.clust_id = clust
        return clust, spike_depth_avg, spike_amp_avg, counts

    def compute_timescales(self) -> None:
        self.t_autocorr = 1e3 * np.arange(
            (AUTOCORR_WIN_SIZE / 2) - AUTOCORR_WIN_SIZE,
            (AUTOCORR_WIN_SIZE / 2) + AUTOCORR_BIN_SIZE,
            AUTOCORR_BIN_SIZE,
        )
        n_template = self.data["clusters"]["waveforms"][0, :, 0].size
        self.t_template = 1e3 * (np.arange(n_template)) / FS

    def normalise_data(self, data, lquant=0, uquant=1):
        levels = np.quantile(data, [lquant, uquant])
        if np.min(data) < 0:
            data = data + np.abs(np.min(data))
        norm_data = data / np.max(data)
        norm_levels = np.quantile(norm_data, [lquant, uquant])
        norm_data[np.where(norm_data < norm_levels[0])] = 0
        norm_data[np.where(norm_data > norm_levels[1])] = 1

        return norm_data, levels
