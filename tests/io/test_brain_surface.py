"""Tests for brain-surface candidate normalization."""

from __future__ import annotations

import json
import logging

import pytest

from ephys_alignment_gui.io.brain_surface import (
    BrainSurfaceError,
    load_brain_surface,
    parse_brain_surface,
)


def surface_json(**overrides):
    """A producer ``brain_surface.json`` with one candidate on shank 1."""
    raw = {
        "version": "1.0",
        "depth_frame": "channels.localCoordinates y, um",
        "shanks": {
            "0": {"candidates": []},
            "1": {
                "candidates": [
                    {
                        "method": "lines",
                        "block_indices": [0, 1],
                        "midpoint_um": 2122.5,
                        "width_um": 30.0,
                        "interval_um": [2110.0, 2135.0],
                        "delta_bic": 3269.0,
                        "n_agreeing_lines": 1,
                        "n_lines": 1,
                        "lines": [
                            {
                                "line_index": 0,
                                "freq_hz": 3870.25,
                                "amplitude_rows": {"0": 0, "1": 1},
                                "step_db": 18.0,
                                "width_below_um": 0.0,
                                "width_above_um": 7.5,
                                "support": 4000.0,
                                "agrees": True,
                                "baseline_db": {"0": -120.0, "1": -116.0},
                                "trend_db_per_mm": 0.5,
                            }
                        ],
                    }
                ]
            },
        },
        "lines": [
            {
                "freq_hz": 3870.25,
                "line_index": 0,
                "block_index": 0,
                "label": "main",
                "excess_db": 12.0,
            },
            {
                "freq_hz": 3870.25,
                "line_index": 0,
                "block_index": 1,
                "label": "surface",
                "excess_db": None,
            },
        ],
    }
    raw.update(overrides)
    return raw


def test_parse_normalizes_candidates_per_integer_shank():
    surface = parse_brain_surface(surface_json())

    assert surface.shank_candidates(0) == ()
    (candidate,) = surface.shank_candidates(1)
    assert candidate.method == "lines"
    assert candidate.interval_um == (2110.0, 2135.0)
    assert candidate.block_indices == (0, 1)
    (line,) = candidate.lines
    assert line.baseline_db == {0: -120.0, 1: -116.0}
    assert line.amplitude_rows == {0: 0, 1: 1}
    assert [row.block_index for row in surface.lines] == [0, 1]
    assert surface.lines[1].excess_db is None
    assert surface.shank_candidates(7) == ()


def test_parse_rejects_unsupported_major_version():
    with pytest.raises(BrainSurfaceError, match="not supported"):
        parse_brain_surface(surface_json(version="2.0"))


def test_parse_rejects_malformed_candidate():
    raw = surface_json()
    del raw["shanks"]["1"]["candidates"][0]["midpoint_um"]

    with pytest.raises(BrainSurfaceError, match="malformed"):
        parse_brain_surface(raw)


def test_load_returns_none_for_missing_file(tmp_path):
    assert load_brain_surface(tmp_path / "brain_surface.json") is None


def test_load_warns_and_returns_none_for_invalid_file(tmp_path, caplog):
    path = tmp_path / "brain_surface.json"
    path.write_text(json.dumps(surface_json(version="9.0")))
    caplog.set_level(logging.WARNING, logger="ephys_alignment_gui.io.brain_surface")

    assert load_brain_surface(path) is None
    assert "Ignoring unreadable" in caplog.text
