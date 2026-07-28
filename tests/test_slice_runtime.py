"""Tests for anatomical slice runtime cache ownership."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.slice_runtime import SliceRuntime


def test_coronal_slice_cache_hits_by_alignment_key_and_track() -> None:
    runtime = SliceRuntime()
    key = AlignmentKey("rec1", "streamA", 0)
    track = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    slice_data = {"ccf": np.array([[1.0]])}

    runtime.set_coronal_slice(
        alignment_key=key,
        track_interpolation_ras=track,
        slice_data=slice_data,
        fp_slice_data=None,
    )
    hit = runtime.cached_coronal_slice(
        alignment_key=key,
        track_interpolation_ras=track.copy(),
    )

    assert hit is not None
    assert hit.slice_data is slice_data


def test_coronal_slice_cache_misses_when_track_changes() -> None:
    runtime = SliceRuntime()
    key = AlignmentKey("rec1", "streamA", 0)
    track = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    runtime.set_coronal_slice(
        alignment_key=key,
        track_interpolation_ras=track,
        slice_data={"ccf": np.array([[1.0]])},
        fp_slice_data=None,
    )

    assert (
        runtime.cached_coronal_slice(
            alignment_key=key,
            track_interpolation_ras=track + 1.0,
        )
        is None
    )


def test_perpendicular_slice_cache_keys_include_alignment_and_channel() -> None:
    runtime = SliceRuntime()
    key = AlignmentKey("rec1", "streamA", 0)
    other_key = AlignmentKey("rec1", "streamA", 1)
    track = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    depths = np.array([0.0, 1.0])
    feature = np.array([0.0, 1.0])
    track_ref = np.array([0.0, 1.0])
    grid = np.linspace(0.0, 1.0, 5)
    calls: list[str] = []

    cache_key = runtime.perpendicular_key(
        alignment_key=key,
        channel_name="ccf",
        track_interpolation_ras=track,
        ephys_depths_along_track=depths,
        feature_ref=feature,
        track_ref=track_ref,
        feature_grid_m=grid,
        extent_m=500e-6,
        n_perp_samples=41,
    )
    first = runtime.get_or_build_perpendicular_slice(
        key=cache_key,
        builder=lambda: calls.append("build") or np.ones((2, 5)),
    )
    second = runtime.get_or_build_perpendicular_slice(
        key=cache_key,
        builder=lambda: calls.append("rebuild") or np.zeros((2, 5)),
    )
    other_alignment_key = runtime.perpendicular_key(
        alignment_key=other_key,
        channel_name="ccf",
        track_interpolation_ras=track,
        ephys_depths_along_track=depths,
        feature_ref=feature,
        track_ref=track_ref,
        feature_grid_m=grid,
        extent_m=500e-6,
        n_perp_samples=41,
    )
    other_alignment = runtime.get_or_build_perpendicular_slice(
        key=other_alignment_key,
        builder=lambda: calls.append("other") or np.zeros((2, 5)),
    )

    assert first is second
    assert other_alignment is not first
    assert calls == ["build", "other"]


def test_invalidate_alignment_removes_only_matching_cached_slices() -> None:
    runtime = SliceRuntime()
    key0 = AlignmentKey("rec1", "streamA", 0)
    key1 = AlignmentKey("rec1", "streamA", 1)
    track = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    runtime.set_coronal_slice(
        alignment_key=key0,
        track_interpolation_ras=track,
        slice_data={"ccf": np.array([[0.0]])},
        fp_slice_data=None,
    )
    runtime.set_coronal_slice(
        alignment_key=key1,
        track_interpolation_ras=track,
        slice_data={"ccf": np.array([[1.0]])},
        fp_slice_data=None,
    )

    runtime.invalidate_alignment(key0)

    assert (
        runtime.cached_coronal_slice(
            alignment_key=key0,
            track_interpolation_ras=track,
        )
        is None
    )
    assert (
        runtime.cached_coronal_slice(
            alignment_key=key1,
            track_interpolation_ras=track,
        )
        is not None
    )
