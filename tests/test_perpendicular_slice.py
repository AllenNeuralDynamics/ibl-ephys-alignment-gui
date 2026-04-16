"""Tests for the perpendicular-slice helpers."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.perpendicular_slice import (
    _ATLAS_AP_AXIS,
    _ATLAS_ML_AXIS,
    arc_lengths,
    perpendicular_direction,
    position_and_tangent_at_arc_lengths,
    smoothed_tangents,
)


def _piecewise_linear_trajectory() -> np.ndarray:
    """Two linear segments meeting at a 90-degree corner.

    Segment 1: straight down along -DV (tangent = (0, 0, -1))
    Segment 2: straight in +AP  (tangent = (0, 1,  0))
    """
    n = 100
    seg1 = np.stack(
        [np.zeros(n), np.zeros(n), np.linspace(0.0, -0.001, n)], axis=1
    )  # (0, 0, 0)->(0, 0, -1mm)
    seg2 = np.stack(
        [np.zeros(n), np.linspace(0.0, 0.001, n), np.full(n, -0.001)], axis=1
    )  # turn AP at the corner
    seg2 = seg2[1:]  # avoid duplicating the corner point
    return np.vstack([seg1, seg2])


def test_smoothed_tangents_shape_and_norm():
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((80, 3))
    tans = smoothed_tangents(xyz, sigma_samples=2.0)
    assert tans.shape == xyz.shape
    np.testing.assert_allclose(np.linalg.norm(tans, axis=1), 1.0, atol=1e-10)


def test_smoothed_tangents_rounds_corners():
    """At the xyz-picks corner, raw tangent jumps; smoothed tangent varies continuously."""
    traj = _piecewise_linear_trajectory()

    # Raw tangent: piecewise constant, steps at the join.
    diffs = np.diff(traj, axis=0)
    raw = diffs / np.linalg.norm(diffs, axis=1, keepdims=True)
    raw_jump = np.linalg.norm(raw[len(raw) // 2] - raw[len(raw) // 2 - 1])

    tans = smoothed_tangents(traj, sigma_samples=3.0)
    smooth_jump = np.linalg.norm(tans[len(tans) // 2] - tans[len(tans) // 2 - 1])

    # The raw step is a full right-angle (~sqrt(2)). Smoothing with sigma=3
    # samples spreads the transition over ~6 samples on each side, so the
    # per-step delta should drop by ~5x compared to the raw discontinuity.
    assert raw_jump > 1.0
    assert smooth_jump < raw_jump / 3, (
        f"smoothing didn't flatten the corner: raw={raw_jump:.3f}, smooth={smooth_jump:.3f}"
    )


def test_smoothed_tangents_leaves_straight_segment_untouched():
    """Away from corners the smoothed tangent equals the raw (PL) tangent."""
    n = 200
    # Straight DV trajectory, no corners anywhere.
    traj = np.stack([np.zeros(n), np.zeros(n), np.linspace(0.0, -0.004, n)], axis=1)
    tans = smoothed_tangents(traj, sigma_samples=2.0)
    # Expected unit tangent is (0, 0, -1) everywhere.
    expected = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    np.testing.assert_allclose(tans, expected, atol=1e-10)


def test_perpendicular_direction_pure_dv_tangent_gives_ml():
    """For a probe going straight down, the perp direction is exactly +ML."""
    tangent = np.tile(np.array([0.0, 0.0, -1.0]), (5, 1))
    perp = perpendicular_direction(tangent)
    np.testing.assert_allclose(perp, np.tile(_ATLAS_ML_AXIS, (5, 1)), atol=1e-12)


def test_perpendicular_direction_stays_perpendicular_to_tangent():
    """For arbitrary tangents, output is unit-norm and orthogonal to tangent."""
    rng = np.random.default_rng(1)
    # Random unit tangents away from ML so the default projection doesn't
    # trip the guard.
    raw = rng.standard_normal((20, 3))
    # Bias away from the ML axis to keep projection non-degenerate.
    raw[:, 0] *= 0.1
    tans = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    perp = perpendicular_direction(tans)
    np.testing.assert_allclose(np.linalg.norm(perp, axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.sum(perp * tans, axis=1), 0.0, atol=1e-10)


def test_perpendicular_direction_falls_back_when_tangent_parallel_to_ml():
    """Tangent along +ML -> ref projection collapses; fallback (AP) kicks in."""
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (3, 1))
    perp = perpendicular_direction(tangent)
    # AP axis is already perpendicular to ML; projection is itself after sign fix.
    np.testing.assert_allclose(
        perp, np.tile(_ATLAS_AP_AXIS, (3, 1)), atol=1e-12
    )


def test_perpendicular_direction_sign_stable():
    """Tangents close to ML shouldn't flip the perp across consecutive rows."""
    theta = np.linspace(0.1, np.pi / 2 - 0.1, 50)
    # Rotate tangent in the ML-DV plane.
    tangents = np.stack([np.cos(theta), np.zeros_like(theta), -np.sin(theta)], axis=1)
    perp = perpendicular_direction(tangents)
    # Dot products with +ML should all be positive (sign-fixed).
    dots = perp @ _ATLAS_ML_AXIS
    assert np.all(dots > 0), f"sign flipped: {dots[dots <= 0]}"


def test_position_and_tangent_at_arc_lengths_matches_interp():
    """Sanity: the sampler returns np.interp-equivalent positions."""
    n = 80
    arc = np.linspace(0.0, 4e-3, n)  # metres
    traj = np.stack(
        [np.linspace(0.0, 1e-4, n), np.zeros(n), -arc], axis=1
    )
    q = np.array([0.0, 1e-3, 2e-3, 3e-3])
    pos, tan = position_and_tangent_at_arc_lengths(traj, arc, q)
    # Positions are linear in arc so interp exact.
    expected_pos = np.stack(
        [
            np.interp(q, arc, traj[:, 0]),
            np.interp(q, arc, traj[:, 1]),
            np.interp(q, arc, traj[:, 2]),
        ],
        axis=1,
    )
    np.testing.assert_allclose(pos, expected_pos, atol=1e-12)
    # Tangent unit-norm.
    np.testing.assert_allclose(np.linalg.norm(tan, axis=1), 1.0, atol=1e-10)


def test_arc_lengths_monotone_and_starts_at_zero():
    traj = np.cumsum(np.ones((30, 3)) * 0.025, axis=0)
    s = arc_lengths(traj)
    assert s[0] == 0.0
    assert np.all(np.diff(s) >= 0.0)
