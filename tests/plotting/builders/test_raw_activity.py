"""Tests for raw-activity plot payloads."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.plotting.builders.raw_activity import (
    RawActivityPlotDataBuilder,
)
from ephys_alignment_gui.plotting.channel_geometry import build_plot_channel_geometry


def _two_shank_data(rate_hz: np.ndarray) -> dict:
    """Shank 0 has a column pair at each depth; shank 1 is one column."""
    local_coordinates = np.array(
        [
            [0.0, 0.0],
            [32.0, 0.0],
            [0.0, 20.0],
            [32.0, 20.0],
            [250.0, 0.0],
            [250.0, 20.0],
        ]
    )
    return {
        "channels": {
            "localCoordinates": local_coordinates,
            "shankInd": np.array([0, 0, 0, 0, 1, 1]),
        },
        "raw_activity": {"exists": True, "rate_hz": rate_hz},
    }


def _builder(data: dict, shank_idx: int) -> RawActivityPlotDataBuilder:
    return RawActivityPlotDataBuilder(
        data, build_plot_channel_geometry(data, shank_idx)
    )


def test_line_averages_contacts_sharing_a_depth_on_the_active_shank() -> None:
    rate_hz = np.array([10.0, 30.0, 50.0, np.nan, 900.0, 900.0])

    line, probe = _builder(_two_shank_data(rate_hz), 0).build()

    np.testing.assert_array_equal(line["y"], [0.0, 20.0])
    np.testing.assert_allclose(line["x"], [20.0, 50.0])
    np.testing.assert_allclose(line["xrange"], [0.0, 50.0])
    assert line["xaxis"] == probe["title"] == "Raw Activity (Hz)"
    np.testing.assert_allclose(
        probe["levels"], np.nanquantile([10, 30, 50], [0.1, 0.9])
    )


def test_probe_keeps_rejected_contacts_blank() -> None:
    rate_hz = np.array([10.0, 30.0, 50.0, np.nan, 900.0, 900.0])

    _, probe = _builder(_two_shank_data(rate_hz), 0).build()

    values = np.concatenate([bank.ravel() for bank in probe["img"]])
    assert np.isnan(values).sum() == 1
    np.testing.assert_allclose(np.sort(values[np.isfinite(values)]), [10, 30, 50])


def test_shank_without_measured_rates_has_no_payload() -> None:
    rate_hz = np.array([10.0, 30.0, 50.0, 70.0, np.nan, np.nan])
    builder = _builder(_two_shank_data(rate_hz), 1)

    assert builder.rates_hz() is None
    assert builder.build() == (None, None)


def test_missing_raw_activity_has_no_payload() -> None:
    data = _two_shank_data(np.zeros(6))
    data["raw_activity"] = {"exists": False}

    assert _builder(data, 0).build() == (None, None)
