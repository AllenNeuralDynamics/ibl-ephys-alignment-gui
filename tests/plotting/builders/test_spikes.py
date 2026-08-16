"""Tests for spike-derived plot payload builders."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np

import ephys_alignment_gui.plotting.builders.spikes as spikes_module
from ephys_alignment_gui.plotting.builders.spikes import SpikePlotDataBuilder
from ephys_alignment_gui.plotting.raster_request import ImageRasterRequest


def test_fr_image_reuses_binned_image_but_recomputes_levels(monkeypatch) -> None:
    calls = []

    def fake_bincount2d(*args, **kwargs):
        calls.append((args, kwargs))
        counts = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0, 10.0],
                [100.0, 100.0, 100.0, 100.0],
            ]
        )
        return counts, np.array([0.0, 1.0]), np.array([0.0, 5.0, 10.0])

    monkeypatch.setattr(spikes_module, "bincount2D", fake_bincount2d)
    builder = SpikePlotDataBuilder(
        {
            "spikes": {
                "exists": True,
                "times": np.array([0.1, 0.2, 0.3]),
                "depths": np.array([0.0, 5.0, 10.0]),
                "amps": np.array([1.0, 1.0, 1.0]),
                "clusters": np.array([0, 0, 0]),
            },
            "clusters": {
                "exists": True,
                "metrics": {},
                "waveforms": np.zeros((1, 3, 1)),
            },
            "spike_shanks": np.array([0, 0, 0]),
        },
        SimpleNamespace(chn_min=0.0, chn_max=10.0),
        shank_idx=0,
    )

    raster_request = ImageRasterRequest(
        max_time_bins=2,
        max_depth_bins=1,
        min_time_bin_s=0.0,
        min_depth_bin_um=0.0,
    )
    unmasked = builder.get_fr_img(raster_request=raster_request)
    masked = builder.get_fr_img(np.array([10.0]), raster_request=raster_request)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0][2:4], (0.1, 10.0))
    assert masked["img"] is unmasked["img"]
    assert not np.array_equal(masked["levels"], unmasked["levels"])


def test_fr_image_marks_depth_bins_without_channel_support(monkeypatch) -> None:
    def fake_bincount2d(*args, **kwargs):
        counts = np.ones((5, 2))
        return counts, np.array([0.0, 1.0]), np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    monkeypatch.setattr(spikes_module, "bincount2D", fake_bincount2d)
    builder = SpikePlotDataBuilder(
        {
            "spikes": {
                "exists": True,
                "times": np.array([0.1, 0.2, 0.3]),
                "depths": np.array([0.0, 10.0, 40.0]),
                "amps": np.array([1.0, 1.0, 1.0]),
                "clusters": np.array([0, 0, 0]),
            },
            "clusters": {
                "exists": True,
                "metrics": {},
                "waveforms": np.zeros((1, 3, 1)),
            },
            "spike_shanks": np.array([0, 0, 0]),
        },
        SimpleNamespace(
            chn_min=0.0,
            chn_max=40.0,
            chn_diff=10.0,
            chn_coords=np.array(
                [
                    [0.0, 0.0],
                    [0.0, 10.0],
                    [0.0, 40.0],
                ]
            ),
        ),
        shank_idx=0,
    )

    payload = builder.get_fr_img(
        raster_request=ImageRasterRequest(
            max_time_bins=2,
            max_depth_bins=4,
            min_time_bin_s=0.0,
            min_depth_bin_um=0.0,
        )
    )

    assert payload is not None
    assert payload["no_data_mask"].shape == payload["img"].shape
    np.testing.assert_array_equal(
        payload["no_data_mask"][0],
        np.array([False, False, True, True, False]),
    )


def test_fr_amp_line_ignores_empty_depth_bins_without_runtime_warning() -> None:
    builder = SpikePlotDataBuilder(
        {
            "spikes": {
                "exists": True,
                "times": np.array([0.1, 1.0]),
                "depths": np.array([0.0, 40.0]),
                "amps": np.array([2.0, 4.0]),
                "clusters": np.array([0, 1]),
            },
            "clusters": {
                "exists": True,
                "metrics": {},
                "waveforms": np.zeros((2, 3, 1)),
            },
            "spike_shanks": np.array([0, 0]),
        },
        SimpleNamespace(chn_min=0.0, chn_max=40.0),
        shank_idx=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fr_line, amp_line = builder.get_fr_amp_data_line()

    assert fr_line is not None
    assert amp_line is not None
    assert np.isfinite(amp_line["x"]).all()
