"""Tests for interference-line amplitude payloads."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.io.brain_surface import parse_brain_surface
from ephys_alignment_gui.plotting.builders.line_amplitude import (
    LineAmplitudePlotDataBuilder,
    _logistic,
)
from ephys_alignment_gui.plotting.channel_geometry import build_plot_channel_geometry

N_ROWS = 40
PITCH_UM = 20.0
DEPTHS = np.arange(N_ROWS) * PITCH_UM
MIDPOINT_UM = 410.0  # between two rows of the upper block
BASELINES = {0: -120.0, 1: -116.0}
TREND_DB_PER_MM = 0.5
STEP_DB = 18.0
WIDTH_ABOVE_UM = 15.0


def _model_db(block: int) -> np.ndarray:
    return (
        BASELINES[block]
        + TREND_DB_PER_MM * DEPTHS / 1000.0
        + STEP_DB * _logistic(DEPTHS, MIDPOINT_UM, 0.0, WIDTH_ABOVE_UM)
    )


def _tiled_amps() -> np.ndarray:
    """One line recorded by two blocks tiling the shank, lower then upper,
    exactly on the fitted model; a second line never measured on it."""
    amps = np.full((4, N_ROWS), np.nan)
    lower = DEPTHS < N_ROWS / 2 * PITCH_UM
    for block, rows in ((0, lower), (1, ~lower)):
        amps[block, rows] = 10 ** (_model_db(block)[rows] / 20)
    return amps


def _surface(midpoint_um: float = MIDPOINT_UM, extra_candidates=()):
    line = {
        "line_index": 0,
        "freq_hz": 3870.25,
        "amplitude_rows": {"0": 0, "1": 1},
        "step_db": STEP_DB,
        "width_below_um": 0.0,
        "width_above_um": WIDTH_ABOVE_UM,
        "support": 4000.0,
        "agrees": True,
        "baseline_db": {str(b): v for b, v in BASELINES.items()},
        "trend_db_per_mm": TREND_DB_PER_MM,
    }
    rows = [(0, 0, 3870.25), (0, 1, 3870.25), (1, 0, 7503.1), (1, 1, 7503.1)]
    return parse_brain_surface(
        {
            "version": "1.0",
            "shanks": {
                "0": {
                    "candidates": [
                        {
                            "method": "lines",
                            "block_indices": [0, 1],
                            "midpoint_um": midpoint_um,
                            "width_um": 30.0,
                            "interval_um": [400.0, 420.0],
                            "delta_bic": 3000.0,
                            "n_agreeing_lines": 1,
                            "n_lines": 2,
                            "lines": [line],
                        },
                        *extra_candidates,
                    ]
                }
            },
            "lines": [
                {
                    "line_index": j,
                    "block_index": b,
                    "freq_hz": f,
                    "label": None,
                    "excess_db": 10.0,
                }
                for j, b, f in rows
            ],
        }
    )


def _builder(amps=None, surface=None) -> LineAmplitudePlotDataBuilder:
    data = {
        "channels": {"localCoordinates": np.c_[np.zeros(N_ROWS), DEPTHS]},
        "line_amp": {"exists": True, "amps": _tiled_amps() if amps is None else amps},
        "brain_surface": _surface() if surface is None else surface,
    }
    return LineAmplitudePlotDataBuilder(
        data, build_plot_channel_geometry(data, 0), shank_idx=0
    )


def test_line_keys_list_only_lines_measured_on_the_shank():
    assert _builder().line_keys() == ("3870.2 Hz",)


def test_no_lines_without_a_matching_lines_table():
    builder = _builder(amps=_tiled_amps()[:3])

    assert builder.line_keys() == ()
    assert builder.build_image() is None


def test_profile_is_referenced_to_each_block_median():
    (profile,) = [p["x"] for p in _builder().build_lines().values()]
    lower = DEPTHS < N_ROWS / 2 * PITCH_UM

    for block, rows in ((0, lower), (1, ~lower)):
        expected = _model_db(block)[rows]
        np.testing.assert_allclose(profile[rows], expected - np.median(expected))


def test_fitted_curve_overlays_the_data_it_was_fitted_to():
    (payload,) = _builder().build_lines().values()

    fit = payload["fits"]["lines"]
    np.testing.assert_allclose(fit["x"], payload["x"], atol=1e-9)
    np.testing.assert_array_equal(fit["y"], payload["y"])


def test_line_markers_carry_curve_level_and_step_extent():
    (payload,) = _builder().build_lines().values()
    (marker,) = payload["surface_candidates"]

    assert marker["agrees"] is True
    assert marker["extent_um"] == (
        MIDPOINT_UM,
        MIDPOINT_UM + np.log(9.0) * WIDTH_ABOVE_UM,
    )
    expected = np.interp(MIDPOINT_UM, payload["y"], payload["fits"]["lines"]["x"])
    assert marker["curve_db"] == expected


def test_replicate_blocks_average_after_referencing():
    amps = np.full((4, N_ROWS), np.nan)
    amps[0] = 10 ** (_model_db(0) / 20)
    amps[1] = 10 ** ((_model_db(0) + 6.0) / 20)  # same channels, 6 dB louder
    (payload,) = _builder(amps=amps).build_lines().values()

    expected = _model_db(0)
    np.testing.assert_allclose(payload["x"], expected - np.median(expected))


def test_image_has_one_column_per_line_with_candidate_markers():
    image = _builder().build_image()

    assert image["img"].shape == (1, N_ROWS)
    assert image["line_keys"] == ("3870.2 Hz",)
    assert image["levels"][0] == -image["levels"][1]
    (marker,) = image["surface_candidates"]
    assert marker["midpoint_um"] == MIDPOINT_UM
    assert "curve_db" not in marker


def test_plateau_candidate_draws_its_own_curve():
    # Bath level above the onset, 3 dB lower below it, in the upper block.
    onset_um = 600.0
    plateau = {
        "method": "plateau",
        "block_indices": [1],
        "midpoint_um": onset_um,
        "width_um": None,
        "delta_bic": 200.0,
        "n_agreeing_lines": 1,
        "n_lines": 1,
        "lines": [
            {
                "line_index": 0,
                "freq_hz": 3870.25,
                "amplitude_rows": {"1": 1},
                "step_db": 3.0,
                "width_below_um": 0.0,
                "width_above_um": 0.0,
                "support": 50.0,
                "agrees": True,
                "baseline_db": {"1": -110.0},
                "trend_db_per_mm": 0.0,
            }
        ],
    }
    builder = _builder(surface=_surface(extra_candidates=(plateau,)))
    (payload,) = builder.build_lines().values()

    curve = payload["fits"]["plateau"]["x"]
    upper = DEPTHS >= N_ROWS / 2 * PITCH_UM
    assert np.isnan(curve[~upper]).all()
    above = curve[upper & (DEPTHS > onset_um)]
    below = curve[upper & (DEPTHS < onset_um)]
    np.testing.assert_allclose(above - below[0], 3.0)
    (_, marker) = payload["surface_candidates"]
    assert marker["method"] == "plateau"
    assert marker["curve_db"] == np.interp(onset_um, payload["y"], curve)
