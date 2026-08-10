"""Raw trace image plot payload builder."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry


class RawTracePlotDataBuilder:
    """Build raw trace image payloads for legacy IBL workflows."""

    def __init__(
        self,
        geometry: PlotChannelGeometry,
        *,
        max_spike_time: float | None,
    ) -> None:
        self.geometry = geometry
        self.max_spike_time = max_spike_time

    def build(self, pid, t0=(1000, 2000, 3000), one=None):
        """Return raw AP snippets keyed by requested time."""
        import neuropixel
        from brainbox.io.spikeglx import Streamer
        from neurodsp import voltage

        def gain2level(gain):
            return 10 ** (gain / 20) * 4 * np.array([-1, 1])

        if self.max_spike_time is None:
            return {}

        data_img = {}
        times = [t for t in t0 if t < self.max_spike_time]

        for t in times:
            sr = Streamer(pid=pid, one=one, remove_cached=False, typ="ap")
            th = sr.geometry

            if sr.meta.get("NP2.4_shank", None) is not None:
                h = neuropixel.trace_header(sr.major_version, nshank=4)
                h = neuropixel.split_trace_header(
                    h,
                    shank=int(sr.meta.get("NP2.4_shank")),
                )
            else:
                h = neuropixel.trace_header(
                    sr.major_version,
                    nshank=np.unique(th["shank"]).size,
                )
                idx = np.isin(h["ind"], th["ind"])
                for key in h.keys():
                    h[key] = h[key][idx]

            s0 = t * sr.fs
            tsel = slice(int(s0), int(s0) + int(1 * sr.fs))
            raw = sr[tsel, : -sr.nsync].T
            channel_labels, _channel_features = voltage.detect_bad_channels(raw, sr.fs)
            raw = voltage.destripe(raw, fs=sr.fs, h=h, channel_labels=channel_labels)
            raw_image = raw[:, int((450 / 1e3) * sr.fs) : int((500 / 1e3) * sr.fs)].T
            x_range = np.array([0, raw_image.shape[0] - 1]) / sr.fs * 1e3
            xscale = (x_range[1] - x_range[0]) / raw_image.shape[0]
            yscale = (
                self.geometry.chn_max - self.geometry.chn_min
            ) / raw_image.shape[1]

            data_img[f"Raw data t={t}"] = {
                "img": raw_image,
                "scale": np.array([xscale, yscale]),
                "levels": gain2level(-90),
                "offset": np.array([0, self.geometry.chn_min]),
                "cmap": "bone",
                "xrange": x_range,
                "xaxis": "Time (ms)",
                "title": "Power (uV)",
            }

        return data_img
