"""Stimulus/passive plot payload builders."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np

from ephys_alignment_gui.plotting.builders.spikes import SpikePlotDataBuilder
from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry

PASSIVE_STIM_TYPES = ("valveOn", "toneOn", "noiseOn")
GABOR_STIM_TYPES = ("leftGabor", "rightGabor")
RFMAP_TYPES = ("on", "off")


def _entry_exists(entry) -> bool:
    """Return whether an ALF-style object entry is present."""
    return bool(entry and entry.get("exists", False))


def _brainbox_passive_available() -> bool:
    """Return whether the optional passive-task helpers are importable."""
    try:
        if find_spec("brainbox") is None:
            return False
        if find_spec("brainbox.task") is None:
            return False
        return find_spec("brainbox.task.passive") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _missing_brainbox_passive() -> ModuleNotFoundError:
    """Return the optional-dependency error used by registry logging."""
    return ModuleNotFoundError("No module named 'brainbox'", name="brainbox")


class StimulusPlotDataBuilder:
    """Build RF map and passive-event payloads."""

    def __init__(
        self,
        data,
        geometry: PlotChannelGeometry,
        spike_builder: SpikePlotDataBuilder,
    ) -> None:
        self.data = data
        self.geometry = geometry
        self.spike_builder = spike_builder

    def get_rfmap_data(self):
        """Return receptive-field map payloads and depth bounds."""
        data_img = {}
        if not self.data["rf_map"]["exists"]:
            return data_img, None

        from brainbox.task import passive

        rf_map_times, rf_map_pos, rf_stim_frames = (
            passive.get_on_off_times_and_positions(self.data["rf_map"])
        )
        chn_min, chn_max = self.spike_builder._spike_depth_extent()

        rf_map, _ = passive.get_rf_map_over_depth(
            rf_map_times,
            rf_map_pos,
            rf_stim_frames,
            self.data["spikes"]["times"][self.spike_builder.spike_idx][
                self.spike_builder.kp_idx
            ],
            self.data["spikes"]["depths"][self.spike_builder.spike_idx][
                self.spike_builder.kp_idx
            ],
            d_bin=160,
            y_lim=[chn_min, chn_max],
        )
        rfs_svd = passive.get_svd_map(rf_map)
        img = {
            "on": np.vstack(rfs_svd["on"]),
            "off": np.vstack(rfs_svd["off"]),
        }
        yscale = (self.geometry.chn_max - self.geometry.chn_min) / img["on"].shape[0]
        levels = np.quantile(np.c_[img["on"], img["off"]], [0, 1])
        depths = np.linspace(
            self.geometry.chn_min,
            self.geometry.chn_max,
            len(rfs_svd["on"]) + 1,
        )

        for sub_type in ["on", "off"]:
            data_img.update(
                {
                    sub_type: {
                        "img": [img[sub_type].T],
                        "scale": [np.array([1, yscale])],
                        "levels": levels,
                        "offset": [np.array([0, self.geometry.chn_min])],
                        "cmap": "viridis",
                        "xrange": np.array([0, 15]),
                        "xaxis": "Position",
                        "title": "rfmap (dB)",
                    }
                }
            )

        return data_img, depths

    def rfmap_keys(self) -> tuple[str, ...]:
        """Return cheaply discoverable RF-map payload keys."""
        if not _entry_exists(self.data.get("rf_map")):
            return ()
        if not _brainbox_passive_available():
            raise _missing_brainbox_passive()
        return RFMAP_TYPES

    def get_passive_events(self):
        """Return stimulus-aligned passive-event image payloads."""
        data_img = {}
        if not self.data["pass_stim"]["exists"] and not self.data["gabor"]["exists"]:
            return data_img

        from brainbox.task import passive

        if not self.data["pass_stim"]["exists"] and self.data["gabor"]["exists"]:
            stim_types = ["leftGabor", "rightGabor"]
            stims = self.data["gabor"]
        elif self.data["pass_stim"]["exists"] and not self.data["gabor"]["exists"]:
            stim_types = ["valveOn", "toneOn", "noiseOn"]
            stims = {
                stim_type: self.data["pass_stim"][stim_type] for stim_type in stim_types
            }
        else:
            stim_types = ["valveOn", "toneOn", "noiseOn", "leftGabor", "rightGabor"]
            stims = {
                stim_type: self.data["pass_stim"][stim_type]
                for stim_type in stim_types[0:3]
            }
            stims.update(self.data["gabor"])

        chn_min, chn_max = self.spike_builder._spike_depth_extent()

        base_stim = 1
        pre_stim = 0.4
        post_stim = 1
        stim_events = passive.get_stim_aligned_activity(
            stims,
            self.data["spikes"]["times"][self.spike_builder.spike_idx][
                self.spike_builder.kp_idx
            ],
            self.data["spikes"]["depths"][self.spike_builder.spike_idx][
                self.spike_builder.kp_idx
            ],
            pre_stim=pre_stim,
            post_stim=post_stim,
            base_stim=base_stim,
            y_lim=[chn_min, chn_max],
        )

        for stim_type, z_score in stim_events.items():
            xscale = (post_stim + pre_stim) / z_score.shape[1]
            yscale = (self.geometry.chn_max - self.geometry.chn_min) / z_score.shape[0]

            data_img.update(
                {
                    stim_type: {
                        "img": z_score.T,
                        "scale": np.array([xscale, yscale]),
                        "levels": [-10, 10],
                        "offset": np.array([-1 * pre_stim, self.geometry.chn_min]),
                        "cmap": "bwr",
                        "xrange": [-1 * pre_stim, post_stim],
                        "xaxis": "Time from Stim Onset (s)",
                        "title": "Firing rate (z score)",
                    }
                }
            )

        return data_img

    def passive_event_keys(self) -> tuple[str, ...]:
        """Return cheaply discoverable passive-event payload keys."""
        if not _brainbox_passive_available():
            if _entry_exists(self.data.get("pass_stim")) or _entry_exists(
                self.data.get("gabor")
            ):
                raise _missing_brainbox_passive()
            return ()

        keys = []
        pass_stim = self.data.get("pass_stim")
        if _entry_exists(pass_stim):
            keys.extend(
                stim_type for stim_type in PASSIVE_STIM_TYPES if stim_type in pass_stim
            )

        gabor = self.data.get("gabor")
        if _entry_exists(gabor):
            keys.extend(
                stim_type for stim_type in GABOR_STIM_TYPES if stim_type in gabor
            )

        return tuple(keys)
