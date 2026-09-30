"""Tests for the brain-surface candidate read model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ephys_alignment_gui.application.queries.ephys_plot import EphysPlotQueries
from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.io.brain_surface import CandidateLine, SurfaceCandidate
from ephys_alignment_gui.plotting.registry import line_amplitude_line_key
from ephys_alignment_gui.services.alignment_derived_data import (
    AlignmentDerivedDataService,
)


def _line(line_index: int, freq_hz: float) -> CandidateLine:
    return CandidateLine(
        line_index=line_index,
        freq_hz=freq_hz,
        step_db=18.0,
        width_below_um=0.0,
        width_above_um=10.0,
        support=900.0,
        agrees=True,
        baseline_db={0: -120.0},
        trend_db_per_mm=0.0,
        amplitude_rows={0: line_index},
    )


class FakePayloadCache:
    def get_line_amplitude_labels_by_index(self) -> dict[int, str]:
        return {0: "3870.2 Hz"}

    def get_surface_candidates(self) -> tuple[SurfaceCandidate, ...]:
        return (
            SurfaceCandidate(
                method="lines",
                midpoint_um=2100.0,
                width_um=30.0,
                interval_um=(2090.0, 2110.0),
                delta_bic=3000.0,
                n_agreeing_lines=1,
                n_lines=2,
                block_indices=(0,),
                # The second line was fitted but has no amplitudes here.
                lines=(_line(0, 3870.25), _line(1, 7503.1)),
            ),
        )


def _queries(stream_runtime) -> EphysPlotQueries:
    context = SimpleNamespace(
        runtime=SimpleNamespace(active_stream_runtime=stream_runtime),
        active_shank_idx=lambda: 0,
    )
    return EphysPlotQueries(
        context=context,
        display_state=AlignmentDisplayState(),
        derived_data_service=AlignmentDerivedDataService(),
    )


def test_candidates_carry_line_extents_and_plot_keys() -> None:
    cache = FakePayloadCache()
    queries = _queries(SimpleNamespace(plot_payload_cache_for_shank=lambda _i: cache))

    (candidate,) = queries.active_surface_candidates()
    measured, unmeasured = candidate.lines

    assert candidate.midpoint_um == 2100.0
    assert measured.line_plot_key == line_amplitude_line_key("3870.2 Hz")
    assert measured.extent_um == pytest.approx((2100.0, 2100.0 + 10 * np.log(9)))
    assert unmeasured.line_plot_key is None


def test_no_candidates_without_an_active_stream() -> None:
    assert _queries(None).active_surface_candidates() == ()
