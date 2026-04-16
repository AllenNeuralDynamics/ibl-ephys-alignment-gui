"""Rigid rotation of SPIM-native volumes into an atlas-oriented canonical frame.

The image-space histology produced by the capsule-converter lives in a SPIM
frame whose axes are tilted relative to the atlas template. Displaying or
sampling along array axes in that frame does not correspond to anatomical
ML/AP/DV. We rectify this at load time by extracting the rotation-only part
of the SPIM->template affine (via polar decomposition) and resampling every
image-space asset — plus xyz_picks — into the rotated canonical frame. The
inverse rotation is applied only at I/O boundaries (saving xyz_picks,
composing with the SPIM-native ANTs CCF transform chain).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from scipy.linalg import polar


def load_affine_matrix(affine_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load an ITK/ANTs 3D affine transform and return (3x3 linear, 3 translation).

    The .mat file written by ANTs is an ITK `AffineTransform_double_3_3` or
    similar; SimpleITK can read it directly.
    """
    tx = sitk.ReadTransform(str(affine_path))
    affine = sitk.AffineTransform(tx)
    if affine.GetDimension() != 3:
        raise ValueError(
            f"Expected a 3D affine transform at {affine_path}, got "
            f"dimension {affine.GetDimension()}"
        )
    matrix = np.asarray(affine.GetMatrix(), dtype=np.float64).reshape(3, 3)
    translation = np.asarray(affine.GetTranslation(), dtype=np.float64)
    return matrix, translation


def polar_rotation(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Closest rotation to ``matrix`` via polar decomposition.

    Falls back to the reflection-corrected rotation if the raw polar factor
    has det = -1 (mirror). The fix flips the singular vector corresponding to
    the smallest singular value.
    """
    R, _ = polar(matrix)
    if np.linalg.det(R) < 0:
        u, _, vt = np.linalg.svd(matrix)
        d = np.ones(3)
        d[-1] = np.sign(np.linalg.det(u @ vt))
        R = u @ np.diag(d) @ vt
    return R


def image_center_physical(image: sitk.Image) -> NDArray[np.float64]:
    """Physical-space center of ``image`` (LPS mm)."""
    size = np.asarray(image.GetSize(), dtype=np.float64)
    center_index = (size - 1) / 2.0
    return np.asarray(
        image.TransformContinuousIndexToPhysicalPoint(center_index.tolist()),
        dtype=np.float64,
    )


def rotation_transform(
    R: NDArray[np.float64], center: NDArray[np.float64]
) -> sitk.AffineTransform:
    """SimpleITK affine transform representing rotation ``R`` about ``center``."""
    tx = sitk.AffineTransform(3)
    tx.SetMatrix(R.flatten().tolist())
    tx.SetCenter(center.tolist())
    return tx


def rotated_bounding_box(
    image: sitk.Image, R: NDArray[np.float64], center: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the axis-aligned bounding box of ``image`` after rotation.

    Returns (min_corner, max_corner) in physical space (mm, LPS).
    """
    size = np.asarray(image.GetSize(), dtype=np.float64)
    # The 8 corners of the input image in physical space.
    idx_corners = np.array(
        [
            [0, 0, 0],
            [size[0] - 1, 0, 0],
            [0, size[1] - 1, 0],
            [0, 0, size[2] - 1],
            [size[0] - 1, size[1] - 1, 0],
            [size[0] - 1, 0, size[2] - 1],
            [0, size[1] - 1, size[2] - 1],
            [size[0] - 1, size[1] - 1, size[2] - 1],
        ],
        dtype=np.float64,
    )
    phys_corners = np.array(
        [image.TransformContinuousIndexToPhysicalPoint(c.tolist()) for c in idx_corners]
    )
    # Apply rotation about center: p' = R @ (p - center) + center
    rotated = (phys_corners - center) @ R.T + center
    return rotated.min(axis=0), rotated.max(axis=0)


def rotate_image(
    image: sitk.Image,
    R: NDArray[np.float64],
    center: NDArray[np.float64],
    spacing_mm: float = 0.025,
    interpolator: Literal["linear", "nearest"] = "linear",
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample ``image`` into the rotated canonical frame.

    Output grid is axis-aligned (identity direction), isotropic at
    ``spacing_mm``, with extent chosen to contain all 8 rotated corners of the
    input. The output is the image as seen after applying rotation R about
    ``center``; for downstream array-axis slicing this corresponds to
    anatomical ML/AP/DV.
    """
    interp = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
    }[interpolator]

    min_corner, max_corner = rotated_bounding_box(image, R, center)
    extent = max_corner - min_corner
    out_size = np.ceil(extent / spacing_mm).astype(int) + 1

    reference = sitk.Image(out_size.tolist(), image.GetPixelID())
    reference.SetSpacing([spacing_mm] * 3)
    reference.SetOrigin(min_corner.tolist())
    reference.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    # sitk.Resample applies the inverse of the supplied transform to the output
    # grid to find the input sampling point. We want the output point p_out to
    # pull from the input point R^{-1}(p_out - center) + center; that is the
    # transform "rotate by R about center" applied to the input coords to get
    # output coords. SitK's Resample takes the transform that maps *output* to
    # *input*, so we pass R^T (inverse rotation) about the same center.
    R_inv = R.T
    resample_tx = rotation_transform(R_inv, center)

    return sitk.Resample(
        image,
        reference,
        resample_tx,
        interp,
        default_value,
    )


def rotate_points(
    points: NDArray[np.float64],
    R: NDArray[np.float64],
    center: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply ``R`` about ``center`` to a batch of 3D points (N, 3)."""
    return (points - center) @ R.T + center
