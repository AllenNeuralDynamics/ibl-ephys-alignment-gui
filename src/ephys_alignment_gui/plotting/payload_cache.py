"""Memoized ephys plot payload facade."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ephys_alignment_gui.plotting.builders.lfp_correlation import (
    LfpCorrelationPlotDataBuilder,
)
from ephys_alignment_gui.plotting.builders.lfp_spectrum import (
    LfpSpectrumPlotDataBuilder,
)
from ephys_alignment_gui.plotting.builders.raw import RawTracePlotDataBuilder
from ephys_alignment_gui.plotting.builders.rms import RmsPlotDataBuilder
from ephys_alignment_gui.plotting.builders.spikes import SpikePlotDataBuilder
from ephys_alignment_gui.plotting.builders.stimulus import StimulusPlotDataBuilder
from ephys_alignment_gui.plotting.channel_geometry import (
    PlotChannelGeometry,
    build_plot_channel_geometry,
)
from ephys_alignment_gui.plotting.probe_bank_layout import arrange_channels_to_banks
from ephys_alignment_gui.services.ephys_data import ChannelCollectionView

logger = logging.getLogger(__name__)
np.seterr(divide="ignore", invalid="ignore")


class EphysPlotPayloadCache:
    """Cache and dispatch plot payloads for one ephys channel collection."""

    def __init__(
        self,
        probe_path: Path,
        data: dict[str, Any],
        shank_idx: int,
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

        # Set by query preparation after alignment-derived brain occupancy exists.
        self.in_brain_depths_um = None

        self._img_cache: dict[tuple[str, tuple[Any, ...]], Any] = {}
        self._current_filter: str | None = None

        self.spike_builder = SpikePlotDataBuilder(
            data,
            self.channel_geometry,
            shank_idx,
        )
        if self.data["clusters"]["exists"]:
            self._current_filter = "all"
        self.rms_builder = RmsPlotDataBuilder(data, self.channel_geometry)
        self.lfp_spectrum_builder = LfpSpectrumPlotDataBuilder(
            data,
            self.channel_geometry,
        )
        self.stimulus_builder = StimulusPlotDataBuilder(
            data,
            self.channel_geometry,
            self.spike_builder,
        )

        logger.debug("Spike idx: %s", self.spike_idx)
        logger.debug("Keep idx: %s", self.kp_idx)

    def _apply_channel_geometry(self, geometry: PlotChannelGeometry) -> None:
        """Expose derived channel geometry on legacy cache attributes."""
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

    @property
    def spike_idx(self):
        """Return spike indices selected by the active unit filter."""
        return self.spike_builder.spike_idx

    @property
    def kp_idx(self):
        """Return finite spike-depth/amplitude indices inside ``spike_idx``."""
        return self.spike_builder.kp_idx

    @property
    def clust_id(self):
        """Return cluster ids for the latest cluster scatter payload."""
        return self.spike_builder.clust_id

    @property
    def t_autocorr(self):
        """Return autocorrelogram time axis."""
        return self.spike_builder.t_autocorr

    @property
    def t_template(self):
        """Return template waveform time axis."""
        return self.spike_builder.t_template

    @property
    def max_spike_time(self):
        """Return the maximum spike time, if spike data are loaded."""
        return self.spike_builder.max_spike_time

    def cached(self, method: str, args: tuple[Any, ...] = ()):
        """Return ``self.<method>(*args)``, memoized per payload cache."""
        key = (method, args)
        if key not in self._img_cache:
            self._img_cache[key] = getattr(self, method)(*args)
        return self._img_cache[key]

    def filter_units(self, subset: str) -> None:
        """Apply a unit filter and clear memoized payloads when it changes."""
        if subset == self._current_filter:
            return
        self._current_filter = subset
        self._img_cache.clear()
        self.spike_builder.filter_units(subset)

    def get_depth_data_scatter(self):
        """Return time/depth spike-amplitude scatter payload."""
        return self.spike_builder.get_depth_data_scatter()

    def get_fr_p2t_data_scatter(self):
        """Return cluster firing-rate, duration, and amplitude scatter payloads."""
        return self.spike_builder.get_fr_p2t_data_scatter()

    def get_fr_img(self):
        """Return time/depth firing-rate image payload."""
        return self.spike_builder.get_fr_img(self.in_brain_depths_um)

    def get_fr_amp_data_line(self):
        """Return firing-rate and amplitude depth-profile line payloads."""
        return self.spike_builder.get_fr_amp_data_line()

    def get_spike_correlation_data_img(self):
        """Return depth-bin spike count correlation image payload."""
        return self.spike_builder.get_spike_correlation_data_img(
            self.in_brain_depths_um
        )

    def get_rms_data_img_probe(self, format: str):
        """Return RMS image and probe payloads."""
        return self.rms_builder.build(format, self.in_brain_depths_um)

    def get_raw_data_image(self, pid, t0=(1000, 2000, 3000), one=None):
        """Return raw trace snippets for legacy IBL workflows."""
        return RawTracePlotDataBuilder(
            self.channel_geometry,
            max_spike_time=self.max_spike_time,
        ).build(pid, t0=t0, one=one)

    def get_lfp_spectrum_data(self, format: str):
        """Return LFP spectrum image and probe payloads."""
        return self.lfp_spectrum_builder.build(format, self.in_brain_depths_um)

    def get_lfp_correlation_data_img(self):
        """Return LFP correlation image payloads."""
        return LfpCorrelationPlotDataBuilder(
            probe_path=self.probe_path,
            shank_idx=self.shank_idx,
            geometry=self.channel_geometry,
            in_brain_depths_um=self.in_brain_depths_um,
        ).build()

    def get_rfmap_data(self):
        """Return receptive-field map payloads and depth bounds."""
        return self.stimulus_builder.get_rfmap_data()

    def get_passive_events(self):
        """Return stimulus-aligned passive-event image payloads."""
        return self.stimulus_builder.get_passive_events()

    def get_autocorr(self, clust_idx):
        """Return autocorrelogram and cluster id for a clicked cluster."""
        return self.spike_builder.get_autocorr(clust_idx)

    def get_template_wf(self, clust_idx):
        """Return the primary template waveform for a clicked cluster."""
        return self.spike_builder.get_template_wf(clust_idx)

    def arrange_channels2banks(self, data):
        """Arrange channel values into probe-bank image payloads."""
        return arrange_channels_to_banks(data, self.channel_geometry)
