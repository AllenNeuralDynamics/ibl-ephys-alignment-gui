"""Tests for shank alignment runtime initialization."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ephys_alignment_gui.services.alignment_runtime import AlignmentRuntimeService


class FakeEphysAlignment:
    calls = []

    def __init__(self, **kwargs) -> None:
        self.calls.append(kwargs)
        self.track_interpolation_ras = np.array([[1.0, 2.0, 3.0]])
        self.ephys_depths_along_track = np.array([4.0])
        self.track_annos_and_ends_ras = (
            np.asarray(
                kwargs["track_annotations_ras"],
                dtype=float,
            )
            + 1.0
        )
        if "feature_prev" in kwargs:
            self.feature_init = np.asarray(kwargs["feature_prev"], dtype=float)
            self.track_init = np.asarray(kwargs["track_prev"], dtype=float)
        else:
            self.feature_init = np.array([0.0, 1.0])
            self.track_init = np.array([2.0, 3.0])

    def get_track_and_feature(self):
        return self.feature_init, self.track_init, self.track_annos_and_ends_ras

    @staticmethod
    def get_histology_regions(track_interpolation_ras, ephys_depths_along_track, atlas):
        return "region", "label", "colour", "ignored"


def test_alignment_runtime_service_initializes_and_attaches_shank_runtime() -> None:
    FakeEphysAlignment.calls = []
    service = AlignmentRuntimeService(alignment_cls=FakeEphysAlignment)
    shank_runtime = SimpleNamespace(
        chn_depths=np.array([10.0, 20.0]),
        nearby_boundaries="cached",
    )
    track_annotations_ras = np.array([[0.0, 0.0, 0.0]])

    result = service.initialize_shank_runtime(
        shank_runtime,
        track_annotations_ras=track_annotations_ras,
        brain_atlas="atlas",
        feature_prev=np.array([5.0, 6.0]),
        track_prev=np.array([7.0, 8.0]),
    )

    assert FakeEphysAlignment.calls[0]["brain_atlas"] == "atlas"
    np.testing.assert_array_equal(
        FakeEphysAlignment.calls[0]["chn_depths"], [10.0, 20.0]
    )
    np.testing.assert_array_equal(
        FakeEphysAlignment.calls[0]["feature_prev"],
        [5.0, 6.0],
    )
    np.testing.assert_array_equal(result.feature_init, [5.0, 6.0])
    np.testing.assert_array_equal(result.track_init, [7.0, 8.0])
    assert shank_runtime.ephysalign is result.ephysalign
    assert shank_runtime.region_fp == "region"
    assert shank_runtime.region_label_fp == "label"
    assert shank_runtime.region_colour_fp == "colour"
    assert shank_runtime.nearby_boundaries is None
    np.testing.assert_array_equal(
        shank_runtime.track_annotations_ras, [[0.0, 0.0, 0.0]]
    )
    np.testing.assert_array_equal(
        shank_runtime.track_annos_and_ends_ras,
        [[1.0, 1.0, 1.0]],
    )


def test_alignment_runtime_service_omits_empty_previous_alignment() -> None:
    FakeEphysAlignment.calls = []
    service = AlignmentRuntimeService(alignment_cls=FakeEphysAlignment)
    shank_runtime = SimpleNamespace(chn_depths=np.array([10.0, 20.0]))

    result = service.initialize_shank_runtime(
        shank_runtime,
        track_annotations_ras=np.array([[0.0, 0.0, 0.0]]),
        brain_atlas="atlas",
        feature_prev=None,
        track_prev=None,
    )

    assert "feature_prev" not in FakeEphysAlignment.calls[0]
    assert "track_prev" not in FakeEphysAlignment.calls[0]
    np.testing.assert_array_equal(result.feature_init, [0.0, 1.0])
    np.testing.assert_array_equal(result.track_init, [2.0, 3.0])
