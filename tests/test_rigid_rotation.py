"""Tests for the rigid rotation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ephys_alignment_gui.rigid_rotation import (
    polar_rotation,
    rotate_points,
    rotation_transform,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    # Random 3D rotation via QR of a random normal matrix.
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure det = +1 (proper rotation, not reflection).
    Q *= np.sign(np.linalg.det(Q))
    return Q


def test_polar_rotation_recovers_pure_rotation():
    rng = np.random.default_rng(0)
    R_true = _random_rotation(rng)
    # A pure rotation should polar-decompose to itself.
    R_recovered = polar_rotation(R_true)
    np.testing.assert_allclose(R_recovered, R_true, atol=1e-12)


def test_polar_rotation_strips_scale_and_shear():
    rng = np.random.default_rng(1)
    R_true = _random_rotation(rng)
    # Non-uniform scale + small shear.
    S = np.diag([1.3, 0.8, 1.05]) + 0.05 * rng.standard_normal((3, 3))
    A = R_true @ S
    R_recovered = polar_rotation(A)
    # The recovered R should still be a proper rotation.
    np.testing.assert_allclose(R_recovered.T @ R_recovered, np.eye(3), atol=1e-10)
    assert np.linalg.det(R_recovered) > 0


def test_polar_rotation_fixes_reflection():
    rng = np.random.default_rng(2)
    R_true = _random_rotation(rng)
    # Flip one axis to introduce a reflection component.
    reflect = np.diag([1.0, 1.0, -1.0])
    A = R_true @ reflect @ np.diag([1.1, 0.9, 1.0])
    R_recovered = polar_rotation(A)
    assert np.linalg.det(R_recovered) == pytest.approx(1.0, abs=1e-10)


def test_rotate_points_roundtrip():
    """Pick -> R -> R^{-1} must land exactly where we started (rotation is isometric)."""
    rng = np.random.default_rng(3)
    R = _random_rotation(rng)
    center = rng.standard_normal(3) * 5.0
    points = rng.standard_normal((50, 3)) * 10.0

    rotated = rotate_points(points, R, center)
    unrotated = rotate_points(rotated, R.T, center)

    np.testing.assert_allclose(unrotated, points, atol=1e-12)


def test_rotate_points_matches_sitk_transform():
    """Our point rotation should agree with sitk.AffineTransform.TransformPoint."""
    sitk = pytest.importorskip("SimpleITK")
    rng = np.random.default_rng(4)
    R = _random_rotation(rng)
    center = rng.standard_normal(3) * 2.0

    tx = rotation_transform(R, center)
    points = rng.standard_normal((20, 3))
    expected = np.array([tx.TransformPoint(p.tolist()) for p in points])
    got = rotate_points(points, R, center)

    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_rotate_image_preserves_content_mass_and_center():
    """A rotated volume should preserve total mass and its (rotated) centroid."""
    sitk = pytest.importorskip("SimpleITK")
    from ephys_alignment_gui.rigid_rotation import (
        image_center_physical,
        rotate_image,
    )

    rng = np.random.default_rng(5)
    xx, yy, zz = np.meshgrid(
        np.arange(32), np.arange(28), np.arange(24), indexing="ij"
    )
    # Off-center gaussian so rotation actually moves the centroid.
    arr = np.exp(
        -((xx - 10) ** 2 + (yy - 8) ** 2 + (zz - 6) ** 2) / 20.0
    ).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.025, 0.025, 0.025))

    R = _random_rotation(rng)
    center = image_center_physical(img)

    forward = rotate_image(img, R, center, spacing_mm=0.025)

    arr_orig = sitk.GetArrayFromImage(img)
    arr_fwd = sitk.GetArrayFromImage(forward)

    mass_orig = arr_orig.sum()
    mass_fwd = arr_fwd.sum()
    assert mass_fwd == pytest.approx(mass_orig, rel=0.05), (
        f"mass drift: {mass_fwd:.2f} vs {mass_orig:.2f}"
    )

    # Centroid of the forward-rotated volume should equal R applied to the
    # original centroid (both in physical coords).
    def _centroid_phys(image) -> np.ndarray:
        arr = sitk.GetArrayFromImage(image)  # (z, y, x)
        total = arr.sum()
        idx_z, idx_y, idx_x = np.indices(arr.shape)
        cx = (arr * idx_x).sum() / total
        cy = (arr * idx_y).sum() / total
        cz = (arr * idx_z).sum() / total
        return np.asarray(
            image.TransformContinuousIndexToPhysicalPoint(
                [float(cx), float(cy), float(cz)]
            )
        )

    c_orig = _centroid_phys(img)
    c_fwd = _centroid_phys(forward)
    c_expected = R @ (c_orig - center) + center

    np.testing.assert_allclose(c_fwd, c_expected, atol=0.03)  # within ~1 voxel
