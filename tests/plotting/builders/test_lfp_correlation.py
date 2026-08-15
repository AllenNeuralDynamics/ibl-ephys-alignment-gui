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


def test_lfp_correlation_builder_emits_one_image_per_recording_block(
    tmp_path,
) -> None:
    """Blocks with different contact density get their own affine.

    A single image over both blocks would apply one averaged pitch to a matrix
    whose halves are sampled differently, placing every contact in the denser
    block at the wrong depth.
    """
    band_corr = tmp_path / "band_corr"
    band_corr.mkdir()
    # Block "main": 4 contacts over 2 depths (two columns per depth, pitch 20).
    # Block "surface": 2 contacts over 2 depths (one column per depth, pitch 20).
    (band_corr / "row_channels.json").write_text(
        json.dumps(
            {
                "shanks": {
                    "0": {
                        "rows": [0, 1, 2, 3, 4, 5],
                        "blocks": [
                            {"label": "main", "rows": [0, 1, 2, 3]},
                            {"label": "surface", "rows": [4, 5]},
                        ],
                    }
                }
            }
        )
    )
    rng = np.random.default_rng(0)
    matrix = rng.random((6, 6))
    matrix = (matrix + matrix.T) / 2
    np.save(band_corr / "spont_theta_mean_corr.npy", matrix)
    data = {
        "channels": {
            "localCoordinates": np.array(
                [
                    [0.0, 0.0],
                    [32.0, 0.0],
                    [0.0, 20.0],
                    [32.0, 20.0],
                    [0.0, 40.0],
                    [0.0, 60.0],
                ]
            ),
            "rawInd": np.arange(6),
            "shankInd": np.zeros(6, dtype=int),
        },
        "spikes": {"exists": False},
        "clusters": {"exists": False},
    }
    plot_data = EphysPlotPayloadCache(tmp_path, data, shank_idx=0)

    payload = plot_data.get_lfp_correlation_data_img()["spont_theta"]

    assert isinstance(payload["img"], list)
    assert [img.shape for img in payload["img"]] == [(4, 4), (2, 2)]
    # main: 4 rows over depths 0-20 -> two rows straddle each 20um slab
    np.testing.assert_allclose(payload["scale"][0], np.array([10.0, 10.0]))
    np.testing.assert_allclose(payload["offset"][0], np.array([0.0, 0.0]))
    # surface: 2 rows over depths 40-60 -> full pitch per row
    np.testing.assert_allclose(payload["scale"][1], np.array([20.0, 20.0]))
    np.testing.assert_allclose(payload["offset"][1], np.array([40.0, 40.0]))
    # xrange spans both blocks; levels are shared so the colours are comparable
    np.testing.assert_allclose(payload["xrange"], np.array([0.0, 80.0]))
    assert payload["levels"][0] == -payload["levels"][1]


def test_lfp_correlation_builder_single_image_when_blocks_agree(tmp_path) -> None:
    """One unique block row-set means one image, not a degenerate list."""
    band_corr = tmp_path / "band_corr"
    band_corr.mkdir()
    (band_corr / "row_channels.json").write_text(
        json.dumps(
            {
                "shanks": {
                    "0": {
                        "rows": [0, 1],
                        "blocks": [
                            {"label": "main", "rows": [0, 1]},
                            {"label": "main2", "rows": [0, 1]},
                        ],
                    }
                }
            }
        )
    )
    np.save(
        band_corr / "spont_theta_mean_corr.npy",
        np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    data = {
        "channels": {
            "localCoordinates": np.array([[0.0, 0.0], [0.0, 20.0]]),
            "rawInd": np.arange(2),
            "shankInd": np.zeros(2, dtype=int),
        },
        "spikes": {"exists": False},
        "clusters": {"exists": False},
    }
    plot_data = EphysPlotPayloadCache(tmp_path, data, shank_idx=0)

    payload = plot_data.get_lfp_correlation_data_img()["spont_theta"]

    assert not isinstance(payload["img"], list)
