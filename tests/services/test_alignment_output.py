"""Tests for alignment output construction service."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas
import pytest
from iblutil.util import Bunch

import ephys_alignment_gui.services.alignment_output as alignment_output_service
from ephys_alignment_gui.core.alignment_output import ChannelOutputIdentity
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.services.alignment_output import AlignmentOutputService
from ephys_alignment_gui.services.histology_data import HistologyDataContext


def test_alignment_output_service_creates_channel_dict() -> None:
    brain_regions = Bunch(
        id=np.array([10, 20]),
        xyz=np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]]),
        axial=np.array([20.0, 40.0]),
        lateral=np.array([5.0, 7.0]),
        acronym=np.array(["VISp", "LGd"]),
    )

    channel_dict = AlignmentOutputService.create_channel_dict(brain_regions)

    assert channel_dict["channel_0"] == {
        "x": np.float64(1000.0),
        "y": np.float64(2000.0),
        "z": np.float64(3000.0),
        "axial": np.float64(20.0),
        "lateral": np.float64(5.0),
        "raw_ind": 0,
        "contact_id": None,
        "shank_idx": 0,
        "brain_region_id": 10,
        "brain_region": "VISp",
    }
    assert channel_dict["channel_1"]["brain_region"] == "LGd"


def test_alignment_output_service_creates_channel_dict_with_identity() -> None:
    brain_regions = Bunch(
        id=np.array([10]),
        xyz=np.array([[0.001, 0.002, 0.003]]),
        axial=np.array([20.0]),
        lateral=np.array([5.0]),
        raw_ind=np.array([42]),
        contact_id=np.array([142]),
        shank_idx=np.array([1]),
        acronym=np.array(["VISp"]),
    )

    channel_dict = AlignmentOutputService.create_channel_dict(brain_regions)

    assert channel_dict["channel_0"]["raw_ind"] == 42
    assert channel_dict["channel_0"]["contact_id"] == 142
    assert channel_dict["channel_0"]["shank_idx"] == 1


def test_alignment_output_service_requires_loaded_histology() -> None:
    service = AlignmentOutputService(
        AlignmentDataContext(),
        HistologyDataContext(),
    )

    with pytest.raises(ValueError, match="Brain atlas"):
        service.get_alignment_results(
            np.zeros((1, 3), dtype=float),
            np.zeros((1, 2), dtype=float),
        )


class FakeImage:
    def TransformPhysicalPointToContinuousIndex(self, point):
        return tuple(point)

    def TransformContinuousIndexToPhysicalPoint(self, index):
        return tuple(index)


class FakeRegions:
    @staticmethod
    def get(labels):
        labels = np.asarray(labels)
        return Bunch(
            id=labels,
            xyz=np.zeros((labels.size, 3)),
            axial=np.zeros(labels.size),
            lateral=np.zeros(labels.size),
            acronym=np.array([f"R{label}" for label in labels]),
        )


class FakeBrainAtlas:
    regions = FakeRegions()
    intensity_sitk_image_spim_native = FakeImage()
    pipeline_sitk_image_spim_native = FakeImage()

    @staticmethod
    def get_labels(channel_locations_ras):
        return np.arange(1, len(channel_locations_ras) + 1)

    @staticmethod
    def unrotate_to_spim_native(channel_locations_ras):
        return np.asarray(channel_locations_ras)


def test_alignment_output_service_batches_ants_transforms(monkeypatch) -> None:
    calls = []

    def fake_apply_transforms_to_points(
        dimension,
        points,
        transforms,
        whichtoinvert,
    ):
        calls.append((dimension, points.copy(), transforms, whichtoinvert))
        return pandas.DataFrame(
            {
                "x": np.arange(len(points), dtype=float) + 100.0,
                "y": np.arange(len(points), dtype=float) + 200.0,
                "z": np.arange(len(points), dtype=float) + 300.0,
            }
        )

    monkeypatch.setattr(
        alignment_output_service.ants,
        "apply_transforms_to_points",
        fake_apply_transforms_to_points,
    )
    data_context = AlignmentDataContext()
    data_context.mouse_root = SimpleNamespace(
        transforms=SimpleNamespace(
            image_to_template_affine="image_affine.mat",
            image_to_template_warp="image_warp.nii.gz",
            template_to_ccf_affine="ccf_affine.mat",
            template_to_ccf_warp="ccf_warp.nii.gz",
        )
    )
    data_context.channel_table = SimpleNamespace(n_shanks=2)
    histology_context = HistologyDataContext(
        runtime_data=SimpleNamespace(
            brain_atlas=FakeBrainAtlas(),
            histology_images={},
            lazy_channel_paths={},
        )
    )
    service = AlignmentOutputService(data_context, histology_context)

    first_key = ("rec1", "streamA", 0)
    second_key = ("rec1", "streamA", 1)
    results = service.get_alignment_results_batch(
        {
            first_key: (
                np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]),
                np.array([[5.0, 10.0], [6.0, 20.0]]),
                ChannelOutputIdentity(
                    raw_ind=np.array([5, 6]),
                    contact_id=np.array([105, 106]),
                    shank_idx=np.array([0, 0]),
                ),
            ),
            second_key: (
                np.array([[0.002, 0.0, 0.0]]),
                np.array([[7.0, 30.0]]),
            ),
        }
    )

    assert len(calls) == 1
    assert len(calls[0][1]) == 3
    first_ccf = results[first_key][1]
    second_ccf = results[second_key][1]
    assert first_ccf["channel_0"]["x"] == 100.0
    assert first_ccf["channel_1"]["x"] == 101.0
    assert first_ccf["channel_0"]["raw_ind"] == 5
    assert first_ccf["channel_0"]["contact_id"] == 105
    assert first_ccf["channel_0"]["shank_idx"] == 0
    assert second_ccf["channel_0"]["x"] == 102.0
    assert second_ccf["channel_0"]["raw_ind"] == 0
    assert second_ccf["channel_0"]["contact_id"] is None
    assert second_ccf["channel_0"]["shank_idx"] == 0
    assert results[first_key][2]
