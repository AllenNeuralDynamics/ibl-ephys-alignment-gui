"""Tests for LFP correlation plot-data loading."""

from __future__ import annotations

import json

import numpy as np

from ephys_alignment_gui.plotting.payload_cache import EphysPlotPayloadCache


def test_lfp_correlation_builder_slices_full_matrix_to_shank_rows(tmp_path) -> None:
    band_corr = tmp_path / "band_corr"
    band_corr.mkdir()
    (band_corr / "row_channels.json").write_text(
        json.dumps({"shanks": {"1": {"rows": [1, 3]}}})
    )
    matrix = np.array(
        [
            [1.0, 0.1, 0.2, 0.3],
            [0.1, 1.0, 0.4, 0.6],
            [0.2, 0.4, 1.0, 0.8],
            [0.3, 0.6, 0.8, 1.0],
        ]
    )
    np.save(band_corr / "spont_theta_mean_corr.npy", matrix)
    data = {
        "channels": {
            "localCoordinates": np.array(
                [
                    [0.0, 0.0],
                    [250.0, 0.0],
                    [0.0, 20.0],
                    [250.0, 20.0],
                ]
            ),
            "rawInd": np.array([100, 101, 102, 103]),
            "shankInd": np.array([0, 1, 0, 1]),
        },
        "spikes": {"exists": False},
        "clusters": {"exists": False},
    }
    plot_data = EphysPlotPayloadCache(tmp_path, data, shank_idx=1)

    result = plot_data.get_lfp_correlation_data_img()

    assert list(result) == ["spont_theta"]
    payload = result["spont_theta"]
    np.testing.assert_allclose(payload["img"], np.array([[0.0, 0.6], [0.6, 0.0]]))
    np.testing.assert_allclose(payload["scale"], np.array([20.0, 20.0]))
    np.testing.assert_allclose(payload["offset"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(payload["xrange"], np.array([0.0, 40.0]))
    np.testing.assert_allclose(payload["levels"], np.array([-0.6, 0.6]))
