"""Tests for Qt-free active-alignment derived data computations."""

from __future__ import annotations

import numpy as np
import pytest

from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentDerivedDataService,
)


class FakeEphysAlignment:
    track_extent = np.array([0.0, 4.0])

    def __init__(self) -> None:
        self.region_colour = None
        self.scale_histology_calls = []
        self.scale_factor_calls = []
        self.nearby_boundary_calls = []
        self.arrange_calls = []
        self.track_interpolation_ras = np.array([[1.0, 2.0, 3.0]])
        self.ephys_depths_along_track = np.array([0.0, 1.0])

    def scale_histology_regions(self, feature, track, region=None, region_label=None):
        self.scale_histology_calls.append((feature, track, region, region_label))
        if region is None:
            self.region_colour = np.array([[1, 2, 3]])
            return np.array([[10.0, 20.0]]), np.array([[15.0, "A"]], dtype=object)
        return np.array([[30.0, 40.0]]), np.array([[35.0, "F"]], dtype=object)

    def get_scale_factor(self, region, region_orig=None):
        self.scale_factor_calls.append((region, region_orig))
        return np.array([[1.0, 2.0]]), np.array([1.25])

    @staticmethod
    def get_channel_locations(feature, track):
        return np.array([[feature[0], 0.0, track[0]], [feature[-1], 0.0, track[-1]]])

    @staticmethod
    def get_tip_location(feature, track):
        return np.array([feature[0], -1.0, track[0]])

    @staticmethod
    def get_perp_vector(feature, track):
        return [np.array([[feature[0], 0.0, track[0]], [feature[-1], 0.0, track[-1]]])]

    def get_nearest_boundary(self, track_interpolation_ras, allen, **kwargs):
        self.nearby_boundary_calls.append((track_interpolation_ras, allen, kwargs))
        return {
            "id": np.array([1.0, 2.0]),
            "dist": np.array([10.0, 20.0]),
            "col": ["red", "blue"],
            "parent_id": np.array([3.0, 4.0]),
            "parent_dist": np.array([30.0, 40.0]),
            "parent_col": ["pink", "cyan"],
        }

    def arrange_into_regions(self, depths, ids, distances, colours):
        self.arrange_calls.append((depths, ids, distances, colours))
        return ids, distances, colours


def test_compute_histology_for_allen_annotation_source() -> None:
    ephysalign = FakeEphysAlignment()

    derived = AlignmentDerivedDataService().compute_histology(
        ephysalign=ephysalign,
        feature=np.array([0.0, 4.0]),
        track=np.array([10.0, 14.0]),
        region_annotation_source="Allen",
    )

    np.testing.assert_array_equal(derived.histology.region, [[10.0, 20.0]])
    np.testing.assert_array_equal(
        derived.histology.axis_label,
        np.array([[15.0, "A"]], dtype=object),
    )
    np.testing.assert_array_equal(derived.histology.colour, [[1, 2, 3]])
    np.testing.assert_array_equal(derived.reference_histology.region, [[10.0, 20.0]])
    np.testing.assert_array_equal(derived.scale.region, [[1.0, 2.0]])
    np.testing.assert_array_equal(derived.scale.scale, [1.25])
    assert ephysalign.scale_factor_calls[0][1] is None


def test_compute_histology_for_franklin_paxinos_annotation_source() -> None:
    ephysalign = FakeEphysAlignment()
    region_fp = np.array([[0.0, 1.0]])
    region_label_fp = np.array([[0.5, "FP"]], dtype=object)
    region_colour_fp = np.array([[4, 5, 6]])

    derived = AlignmentDerivedDataService().compute_histology(
        ephysalign=ephysalign,
        feature=np.array([0.0, 4.0]),
        track=np.array([10.0, 14.0]),
        region_annotation_source="FranklinPaxinos",
        region_fp=region_fp,
        region_label_fp=region_label_fp,
        region_colour_fp=region_colour_fp,
    )

    np.testing.assert_array_equal(derived.histology.region, [[30.0, 40.0]])
    np.testing.assert_array_equal(
        derived.histology.axis_label,
        np.array([[35.0, "F"]], dtype=object),
    )
    np.testing.assert_array_equal(derived.histology.colour, region_colour_fp)
    np.testing.assert_array_equal(derived.reference_histology.colour, region_colour_fp)
    assert ephysalign.scale_factor_calls[0][1] is region_fp


def test_compute_histology_rejects_unknown_annotation_source() -> None:
    with pytest.raises(ValueError, match="Unknown region annotation source"):
        AlignmentDerivedDataService().compute_histology(
            ephysalign=FakeEphysAlignment(),
            feature=np.array([0.0, 4.0]),
            track=np.array([10.0, 14.0]),
            region_annotation_source="Other",
        )


def test_compute_channel_projection() -> None:
    derived = AlignmentDerivedDataService().compute_channel_projection(
        ephysalign=FakeEphysAlignment(),
        feature=np.array([0.0, 4.0]),
        track=np.array([10.0, 14.0]),
    )

    np.testing.assert_array_equal(
        derived.channel_locations_ras,
        [[0.0, 0.0, 10.0], [4.0, 0.0, 14.0]],
    )
    np.testing.assert_array_equal(derived.tip_location_ras, [0.0, -1.0, 10.0])
    assert len(derived.perpendicular_vectors) == 1


def test_compute_channel_locations_only() -> None:
    locations = AlignmentDerivedDataService().compute_channel_locations(
        ephysalign=FakeEphysAlignment(),
        feature=np.array([0.0, 4.0]),
        track=np.array([10.0, 14.0]),
    )

    np.testing.assert_array_equal(
        locations,
        [[0.0, 0.0, 10.0], [4.0, 0.0, 14.0]],
    )


def test_compute_nearby_boundaries() -> None:
    ephysalign = FakeEphysAlignment()

    nearby = AlignmentDerivedDataService().compute_nearby_boundaries(
        ephysalign=ephysalign,
        allen="allen-table",
        brain_atlas="atlas",
        steps=7,
    )

    assert ephysalign.nearby_boundary_calls[0][1] == "allen-table"
    assert ephysalign.nearby_boundary_calls[0][2] == {
        "steps": 7,
        "brain_atlas": "atlas",
    }
    np.testing.assert_array_equal(nearby.x, [1.0, 2.0])
    np.testing.assert_array_equal(nearby.y, [10.0, 20.0])
    assert nearby.colours == ["red", "blue"]
    np.testing.assert_array_equal(nearby.parent_x, [3.0, 4.0])
    np.testing.assert_array_equal(nearby.parent_y, [30.0, 40.0])
    assert nearby.parent_colours == ["pink", "cyan"]
