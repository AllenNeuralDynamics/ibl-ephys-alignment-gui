"""Perpendicular histology-slice sampling for the alignment GUI.

The perp slice shows, for each depth along the probe trajectory, a line of
tissue perpendicular to the (smoothed) trajectory tangent. Its purpose is
diagnostic: a misalignment between histology and the CCF-driven region
boundaries in ``fig_hist`` will appear visually as the probe "missing" its
expected anatomy.

Design notes (see conversation leading to this module):

* The trajectory positions come from ``EphysAlignment.track_interpolation_ras``
  — the dense, annotator-invariant piecewise-linear trajectory. We do NOT
  fit a smoothing spline to the positions, because that would move them off
  the user's picks.
* Only the *tangent direction* is smoothed — via a small-sigma Gaussian
  along arc length — to eliminate the discontinuity at xyz-picks corners
  that otherwise shows up as stripes in the perp image.
* The perpendicular line at each depth lies in the plane spanned by the
  atlas ML axis and the tangent, projected onto the orthogonal complement
  of the tangent. For the usual (near-DV) IBL probe orientations this is
  ~atlas ML; for tilted probes it rotates smoothly to stay truly
  perpendicular.
* Sampling is feature-space driven: we pick a uniform feature-space grid,
  invert to track distance via ``feature2track``, and look up position and
  tangent at those arc lengths. No row-resample is needed and the result
  lines up with ``fig_hist`` y-axis for free.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

# In the IBL RAS convention x=ML, y=AP, z=DV.
_ATLAS_ML_AXIS: NDArray[np.float64] = np.array([1.0, 0.0, 0.0])
_ATLAS_AP_AXIS: NDArray[np.float64] = np.array([0.0, 1.0, 0.0])


def arc_lengths(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative Euclidean arc length along the rows of ``xyz``. Shape (N,)."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3); got shape {xyz.shape}")
    if xyz.shape[0] < 2:
        return np.zeros(xyz.shape[0], dtype=np.float64)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def smoothed_tangents(
    xyz: NDArray[np.float64], sigma_samples: float = 2.0
) -> NDArray[np.float64]:
    """Unit tangents at each sample, smoothed along arc length.

    Tangent for row ``i`` is the forward finite difference
    ``xyz[i+1] - xyz[i]`` (backward at the last row). Each component is then
    Gaussian-smoothed with ``sigma_samples`` (in *samples*, i.e. rows, not
    metres), and the resulting vectors are renormalised.

    Positions themselves are untouched — only the derivative is smoothed. For
    the piecewise-linear ``track_interpolation_ras`` this removes the hard
    discontinuity at xyz-picks corners while keeping every sampled voxel
    exactly at the pick-interpolated location.

    Parameters
    ----------
    xyz : NDArray (N, 3)
        Dense trajectory positions (e.g. ``track_interpolation_ras``).
    sigma_samples : float
        Gaussian sigma for the 1D smoothing, in units of input rows. Default
        2.0 (at DV-voxel sampling density this is ~50 um of smoothing).

    Returns
    -------
    NDArray (N, 3), unit-norm per row.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3); got shape {xyz.shape}")
    if xyz.shape[0] < 2:
        raise ValueError("Need at least two trajectory samples")

    # Forward diff for all but the last sample; repeat the last segment for the
    # tail so the returned array has the same length as xyz.
    diffs = np.diff(xyz, axis=0)
    raw = np.vstack([diffs, diffs[-1:]])
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    # Avoid divide-by-zero on degenerate consecutive duplicates. A row of
    # zeros will get renormalised further down via the smoothing + fallback.
    raw = np.where(norms > 0, raw / norms, raw)

    smoothed = np.stack(
        [gaussian_filter1d(raw[:, k], sigma=sigma_samples, mode="nearest") for k in range(3)],
        axis=1,
    )
    smoothed_norm = np.linalg.norm(smoothed, axis=1, keepdims=True)
    # If smoothing happened to cancel everything to ~zero, fall back to the
    # raw direction at that row.
    degenerate = (smoothed_norm < 1e-12).flatten()
    smoothed = np.where(smoothed_norm > 1e-12, smoothed / smoothed_norm, raw)
    if degenerate.any():
        # Renormalise the raw rows we substituted (they're already ~unit, but
        # be defensive in case the raw row was itself zero).
        raw_norms = np.linalg.norm(smoothed[degenerate], axis=1, keepdims=True)
        smoothed[degenerate] = np.where(
            raw_norms > 0, smoothed[degenerate] / raw_norms, _ATLAS_AP_AXIS
        )
    return smoothed


def perpendicular_direction(
    tangent: NDArray[np.float64],
    ref_axis: NDArray[np.float64] = _ATLAS_ML_AXIS,
    fallback_axis: NDArray[np.float64] = _ATLAS_AP_AXIS,
    eps: float = 1e-3,
) -> NDArray[np.float64]:
    """Project ``ref_axis`` onto the orthogonal complement of each tangent row.

    For tangents far from ``ref_axis`` this returns an approximately-``ref_axis``
    direction rotated the minimal amount to be truly perpendicular. For
    tangents parallel to ``ref_axis`` the projection collapses; ``fallback_axis``
    is used in its place (and itself projected to be perpendicular).

    A sign fix forces the output to point in the positive ``ref_axis``
    direction so consecutive rows don't flip.

    Parameters
    ----------
    tangent : NDArray (N, 3)
        Unit tangent vectors, one per depth.
    ref_axis, fallback_axis : NDArray (3,)
        The "preferred" perpendicular reference direction (default atlas ML)
        and the fallback for degenerate cases (default atlas AP).
    eps : float
        Threshold on the ref-axis projection's norm below which we switch to
        the fallback axis.

    Returns
    -------
    NDArray (N, 3), unit-norm per row.
    """
    t = np.asarray(tangent, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"tangent must be (N, 3); got shape {t.shape}")

    def _project(axis: NDArray[np.float64]) -> NDArray[np.float64]:
        dot = t @ axis  # (N,)
        return axis[None, :] - dot[:, None] * t

    ref_perp = _project(ref_axis)
    ref_norms = np.linalg.norm(ref_perp, axis=1, keepdims=True)
    fallback_perp = _project(fallback_axis)
    fallback_norms = np.linalg.norm(fallback_perp, axis=1, keepdims=True)

    use_fallback = (ref_norms < eps).flatten()
    out = np.where(use_fallback[:, None], fallback_perp, ref_perp)
    out_norms = np.where(use_fallback[:, None], fallback_norms, ref_norms)

    # Defensive: any residual zero row gets the fallback axis itself (even if
    # projection collapses against the tangent, which should be rare).
    zero_row = (out_norms < eps).flatten()
    if zero_row.any():
        out[zero_row] = fallback_axis

    out = out / np.linalg.norm(out, axis=1, keepdims=True)

    # Sign fix: force positive dot with ref_axis so the output direction
    # doesn't flip across rows where the projection crossed zero.
    flip = (out @ ref_axis) < 0
    out[flip] = -out[flip]
    return out


def position_and_tangent_at_arc_lengths(
    track_interpolation_ras: NDArray[np.float64],
    ephys_depths_along_track: NDArray[np.float64],
    arc_lengths_query: NDArray[np.float64],
    sigma_samples: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate 3D position and smoothed unit tangent at query arc lengths.

    The dense ``track_interpolation_ras`` gives piecewise-linear positions
    indexed by ``ephys_depths_along_track``. For each query arc length we:

    1. Look up the 3D position via linear interpolation of the dense
       trajectory — exactly preserves the user's picks.
    2. Look up the pre-computed smoothed tangent at the same arc length by
       linear interpolation, then renormalise.

    Parameters
    ----------
    track_interpolation_ras : NDArray (N, 3)
        Dense trajectory positions, in metres RAS, from
        ``EphysAlignment.track_interpolation_ras``.
    ephys_depths_along_track : NDArray (N,)
        Parallel arc-length coordinates (metres), from
        ``EphysAlignment.ephys_depths_along_track``.
    arc_lengths_query : NDArray (M,)
        Arc lengths at which to evaluate. Outside the range of
        ``ephys_depths_along_track`` this extrapolates (``np.interp`` clips).
    sigma_samples : float
        Forwarded to :func:`smoothed_tangents`.

    Returns
    -------
    positions : NDArray (M, 3) in metres.
    tangents  : NDArray (M, 3) unit-norm.
    """
    tangents_dense = smoothed_tangents(track_interpolation_ras, sigma_samples=sigma_samples)
    positions = np.stack(
        [
            np.interp(arc_lengths_query, ephys_depths_along_track, track_interpolation_ras[:, k])
            for k in range(3)
        ],
        axis=1,
    )
    tangents_interp = np.stack(
        [
            np.interp(arc_lengths_query, ephys_depths_along_track, tangents_dense[:, k])
            for k in range(3)
        ],
        axis=1,
    )
    tangents_interp /= np.linalg.norm(tangents_interp, axis=1, keepdims=True)
    return positions, tangents_interp


def sample_perpendicular_lines(
    volume_arr: NDArray,
    brain_atlas,
    positions_ras_m: NDArray[np.float64],
    perp_dirs_ras_m: NDArray[np.float64],
    extent_m: float,
    n_samples: int,
) -> NDArray[np.float64]:
    """Sample ``n_samples`` voxels along each perpendicular line.

    Returns an ``(n_samples, n_depths)`` array. Out-of-volume samples are set
    to NaN so the caller (plot) can render them transparent.

    Parameters
    ----------
    volume_arr : NDArray, 3D
        The image array to sample from — ``brain_atlas.image``,
        ``brain_atlas.label``, or any array with the same shape and axis
        convention as the atlas (same dims2xyz / xyz2dims mapping).
    brain_atlas : BrainAtlasAnatomical
        Provides ``physical_points_to_indices`` for the RAS-m -> array-index
        mapping and ``bc.xlim/ylim/zlim`` for the OOB check.
    positions_ras_m : NDArray (N, 3)
        Trajectory positions at each depth, RAS metres.
    perp_dirs_ras_m : NDArray (N, 3)
        Unit perpendicular directions at each depth, RAS metres.
    extent_m : float
        Half-window in metres. Samples cover ``[-extent_m, +extent_m]``.
    n_samples : int
        Number of samples across the full window (including endpoints).

    Returns
    -------
    NDArray (n_samples, n_depths) float64, with NaN in OOB cells.
    """
    n_depths = positions_ras_m.shape[0]
    offsets = np.linspace(-extent_m, extent_m, n_samples)
    # (n_samples, n_depths, 3) — broadcast positions (n_depths, 3) with
    # offsets (n_samples,) scaled by perp_dirs (n_depths, 3).
    pts = (
        positions_ras_m[None, :, :]
        + offsets[:, None, None] * perp_dirs_ras_m[None, :, :]
    )
    flat = pts.reshape(-1, 3)

    # Physical-space OOB mask. bc.x/y/zlim can be reverse-ordered when the
    # underlying dxyz has a negative sign on that axis, so normalise first.
    def _bounds(lim: np.ndarray) -> tuple[float, float]:
        return float(min(lim)), float(max(lim))

    xlo, xhi = _bounds(brain_atlas.bc.xlim)
    ylo, yhi = _bounds(brain_atlas.bc.ylim)
    zlo, zhi = _bounds(brain_atlas.bc.zlim)
    in_bounds = (
        (flat[:, 0] >= xlo)
        & (flat[:, 0] <= xhi)
        & (flat[:, 1] >= ylo)
        & (flat[:, 1] <= yhi)
        & (flat[:, 2] >= zlo)
        & (flat[:, 2] <= zhi)
    )

    idx = brain_atlas.physical_points_to_indices(flat, round=True, mode="clip")
    idx = idx.astype(np.int64)

    values = volume_arr[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
    values[~in_bounds] = np.nan
    return values.reshape(n_samples, n_depths)


def build_perpendicular_slice(
    volume_arr: NDArray,
    brain_atlas,
    track_interpolation_ras: NDArray[np.float64],
    ephys_depths_along_track: NDArray[np.float64],
    feature_ref: NDArray[np.float64],
    track_ref: NDArray[np.float64],
    feature_grid_m: NDArray[np.float64],
    extent_m: float = 500e-6,
    n_perp_samples: int = 41,
    sigma_samples: float = 2.0,
) -> NDArray[np.float64]:
    """Assemble the perpendicular slice image for the current alignment.

    Orchestrates the full feature-space-driven sample:

    1. Map each feature-space depth to a track-arc-length via
       :meth:`EphysAlignment.feature2track` (piecewise-linear, defined by the
       user's alignment reference points ``feature_ref`` / ``track_ref``).
    2. Look up 3D position and smoothed unit tangent at those arc lengths.
    3. Compute the perpendicular direction at each row (atlas-ML projected
       onto the tangent's orthogonal complement, with an AP fallback when
       degenerate).
    4. Sample the 3D volume along each perpendicular line with NaN fill for
       out-of-volume samples.

    Parameters
    ----------
    volume_arr : NDArray (3D)
        Image array to sample; same axis convention as
        ``brain_atlas.image`` / ``brain_atlas.label`` (i.e. the blessed
        DICOM orientation used by :class:`BrainAtlasAnatomical`).
    brain_atlas : BrainAtlasAnatomical
        Used for RAS-m -> array-index mapping and physical-bounds check.
    track_interpolation_ras, ephys_depths_along_track : NDArray
        From the current :class:`EphysAlignment`.
    feature_ref, track_ref : NDArray
        Current alignment reference points (``features[idx]``, ``track[idx]``).
    feature_grid_m : NDArray (M,)
        Feature-space depths at which to sample. Usually uniform in feature
        space so the output y-axis lines up with ``fig_hist``.
    extent_m : float
        Half-window of the perpendicular line, in metres. Default 500 um.
    n_perp_samples : int
        Samples across the perp window. Default 41 (yields 25 um at 500 um
        extent, matching the atlas resolution).
    sigma_samples : float
        Tangent-smoothing sigma, in samples of ``track_interpolation_ras``.

    Returns
    -------
    NDArray (n_perp_samples, M) float64, NaN for OOB samples.
    """
    # Feature grid -> track arc length. Delayed import of EphysAlignment to
    # avoid pulling it at module import time.
    from ephys_alignment_gui.ephys_alignment import EphysAlignment

    arc_lengths_query = EphysAlignment.feature2track(
        feature_grid_m, feature_ref, track_ref
    )
    positions, tangents = position_and_tangent_at_arc_lengths(
        track_interpolation_ras,
        ephys_depths_along_track,
        np.asarray(arc_lengths_query, dtype=np.float64),
        sigma_samples=sigma_samples,
    )
    perp_dirs = perpendicular_direction(tangents)
    return sample_perpendicular_lines(
        volume_arr=volume_arr,
        brain_atlas=brain_atlas,
        positions_ras_m=positions,
        perp_dirs_ras_m=perp_dirs,
        extent_m=extent_m,
        n_samples=n_perp_samples,
    )
