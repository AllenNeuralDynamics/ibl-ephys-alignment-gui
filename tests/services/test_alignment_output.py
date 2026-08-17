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
from ephys_alignment_gui.services.ants_points_transform import (
    AntsPointTransformCancelled,
)
from ephys_alignment_gui.services.ccf_transform_frame import (
    PIPELINE_FRAME,
    SPIM_NATIVE_FRAME,
)
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


class OffsetPipelineImage(FakeImage):
    def TransformContinuousIndexToPhysicalPoint(self, index):
        return (index[0] + 10.0, index[1], index[2])


class FakeRegions:
    @staticmethod
    def get(labels):
        labels = np.asarray(labels)
        return Bunch(
            id=labels,
            xyz=np.zeros((labels.size, 3)),
            axial=np.zeros(labels.size),
            lateral=np.zeros(labels.size),
            acronym=np.array(
                ["void" if label == 0 else f"R{label}" for label in labels]
            ),
        )


class FakeBrainAtlas:
    regions = FakeRegions()
    intensity_sitk_image_spim_native = FakeImage()
    pipeline_sitk_image_spim_native = FakeImage()
    ccf_transform_input_frame = PIPELINE_FRAME
    ccf_transform_input_frame_reason = "test frame"

    @staticmethod
    def get_labels(channel_locations_ras):
        return np.arange(1, len(channel_locations_ras) + 1)

    @staticmethod
    def unrotate_to_spim_native(channel_locations_ras):
        return np.asarray(channel_locations_ras)


class FakePipelineFrameBrainAtlas(FakeBrainAtlas):
    pipeline_sitk_image_spim_native = OffsetPipelineImage()
    ccf_transform_input_frame = PIPELINE_FRAME


class FakeSpimNativeFrameBrainAtlas(FakePipelineFrameBrainAtlas):
    ccf_transform_input_frame = SPIM_NATIVE_FRAME


class FakeVoidSecondChannelBrainAtlas(FakeBrainAtlas):
    @staticmethod
    def get_labels(channel_locations_ras):
        return np.array([1, 0])


def test_alignment_output_service_batches_ants_transforms(monkeypatch) -> None:
    calls = []

    def fake_apply_transforms_to_points(
        points,
        *,
        dimension,
        transforms,
        whichtoinvert,
        cancel_token=None,
    ):
        calls.append(
            (
                dimension,
                pandas.DataFrame(points, columns=list("xyz")),
                transforms,
                whichtoinvert,
                cancel_token,
            )
        )
        return np.column_stack(
            [
                np.arange(len(points), dtype=float) * 0.1,
                np.arange(len(points), dtype=float) * 0.2,
                np.arange(len(points), dtype=float) * 0.3,
            ]
        )

    monkeypatch.setattr(
        alignment_output_service,
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
    assert first_ccf["channel_0"]["x"] == 0.0
    assert first_ccf["channel_1"]["x"] == 0.1
    assert first_ccf["channel_0"]["raw_ind"] == 5
    assert first_ccf["channel_0"]["contact_id"] == 105
    assert first_ccf["channel_0"]["shank_idx"] == 0
    assert second_ccf["channel_0"]["x"] == 0.2
    assert second_ccf["channel_0"]["raw_ind"] == 0
    assert second_ccf["channel_0"]["contact_id"] is None
    assert second_ccf["channel_0"]["shank_idx"] == 0
    assert results[first_key][2]


def test_alignment_output_service_rejects_ccf_shape_mismatch() -> None:
    channel_dict = {
        "channel_0": {
            "axial": 0.0,
            "lateral": 0.0,
            "raw_ind": 0,
            "contact_id": None,
            "shank_idx": 0,
            "brain_region_id": 1,
            "brain_region": "R1",
        },
        "channel_1": {
            "axial": 20.0,
            "lateral": 0.0,
            "raw_ind": 1,
            "contact_id": None,
            "shank_idx": 0,
            "brain_region_id": 2,
            "brain_region": "R2",
        },
    }

    with pytest.raises(RuntimeError, match="different number of points"):
        AlignmentOutputService._create_ccf_channel_dict(
            channel_dict,
            np.array([[0.0, 0.0, 0.0]]),
        )


def test_alignment_output_service_warns_but_keeps_out_of_bounds_ccf_ml(
    caplog,
    monkeypatch,
) -> None:
    def fake_apply_transforms_to_points(
        points,
        *,
        dimension,
        transforms,
        whichtoinvert,
        cancel_token=None,
    ):
        return np.asarray(
            [
                [9.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    monkeypatch.setattr(
        alignment_output_service,
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
    histology_context = HistologyDataContext(
        runtime_data=SimpleNamespace(
            brain_atlas=FakeBrainAtlas(),
            histology_images={},
            lazy_channel_paths={},
        )
    )
    service = AlignmentOutputService(data_context, histology_context)

    with caplog.at_level("WARNING", logger=alignment_output_service.__name__):
        results = service.get_alignment_results_batch(
            {
                ("rec", "bad", 0): (
                    np.array([[0.0, 0.0, 0.0]]),
                    np.array([[0.0, 0.0]]),
                ),
                ("rec", "good", 0): (
                    np.array([[0.001, 0.0, 0.0]]),
                    np.array([[0.0, 0.0]]),
                ),
            }
        )

    channel_results, ccf_results, multi_shank = results[("rec", "bad", 0)]
    assert channel_results["channel_0"]["raw_ind"] == 0
    # Kept, not trimmed: the breach is evidence of a bad transform frame, and
    # deleting it would leave the rest of the shank looking clean.
    assert ccf_results["channel_0"]["x"] == 9.5
    assert results[("rec", "good", 0)][1]["channel_0"]["x"] == 0.0
    assert not multi_shank
    assert "in-brain ML coordinates outside Allen CCF bounds" in caplog.text
    bad_status = service.ccf_export_status_by_key[("rec", "bad", 0)]
    good_status = service.ccf_export_status_by_key[("rec", "good", 0)]
    assert bad_status.status == "complete"
    assert bad_status.omitted_channel_count == 0
    assert bad_status.issues[0].reason == "in_brain_ml_out_of_ccf_bounds"
    assert good_status.status == "complete"


def test_alignment_output_service_exports_out_of_brain_ccf_rows(
    monkeypatch,
) -> None:
    def fake_apply_transforms_to_points(
        points,
        *,
        dimension,
        transforms,
        whichtoinvert,
        cancel_token=None,
    ):
        return np.asarray(
            [
                [0.25, 0.0, 0.0],
                [9.5, 0.0, 0.0],
            ],
            dtype=float,
        )

    monkeypatch.setattr(
        alignment_output_service,
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
    histology_context = HistologyDataContext(
        runtime_data=SimpleNamespace(
            brain_atlas=FakeVoidSecondChannelBrainAtlas(),
            histology_images={},
            lazy_channel_paths={},
        )
    )
    service = AlignmentOutputService(data_context, histology_context)

    results = service.get_alignment_results_batch(
        {
            ("rec", "stream", 0): (
                np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, 20.0]]),
            )
        }
    )

    _channel_results, ccf_results, _multi_shank = results[("rec", "stream", 0)]
    # channel_1 is void and lands past the ML bound; the track past the pia is
    # real geometry, so it is exported and only recorded as an issue.
    assert list(ccf_results) == ["channel_0", "channel_1"]
    assert ccf_results["channel_0"]["x"] == 0.25
    assert ccf_results["channel_1"]["x"] == 9.5
    status = service.ccf_export_status_by_key[("rec", "stream", 0)]
    assert status.status == "complete"
    assert status.total_channel_count == 2
    assert status.ccf_channel_count == 2
    assert status.in_brain_channel_count == 1
    assert [i.reason for i in status.issues] == ["out_of_brain_channel_location"]
    assert status.omitted_channel_count == 0
    assert status.in_brain_channel_count == 1
    assert status.issues[0].reason == "out_of_brain_channel_location"
    assert status.issues[0].ml_range_mm == (9.5, 9.5)


def test_alignment_output_service_propagates_ants_cancellation(monkeypatch) -> None:
    def fake_apply_transforms_to_points(
        points,
        *,
        dimension,
        transforms,
        whichtoinvert,
        cancel_token=None,
    ):
        raise AntsPointTransformCancelled("cancelled by user")

    monkeypatch.setattr(
        alignment_output_service,
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
    histology_context = HistologyDataContext(
        runtime_data=SimpleNamespace(
            brain_atlas=FakeBrainAtlas(),
            histology_images={},
            lazy_channel_paths={},
        )
    )
    service = AlignmentOutputService(data_context, histology_context)

    with pytest.raises(AntsPointTransformCancelled):
        service.get_alignment_results_batch(
            {
                ("rec", "stream", 0): (
                    np.array([[0.0, 0.0, 0.0]]),
                    np.array([[0.0, 0.0]]),
                )
            }
        )


@pytest.mark.parametrize(
    ("brain_atlas", "expected_x"),
    [
        (FakePipelineFrameBrainAtlas(), 11.0),
        (FakeSpimNativeFrameBrainAtlas(), 1.0),
    ],
)
def test_alignment_output_service_conditionally_regrids_for_ccf_frame(
    monkeypatch,
    brain_atlas,
    expected_x,
) -> None:
    calls = []

    def fake_apply_transforms_to_points(
        points,
        *,
        dimension,
        transforms,
        whichtoinvert,
        cancel_token=None,
    ):
        calls.append(pandas.DataFrame(points, columns=list("xyz")))
        return np.asarray([[0.0, 0.0, 0.0]], dtype=float)

    monkeypatch.setattr(
        alignment_output_service,
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
    histology_context = HistologyDataContext(
        runtime_data=SimpleNamespace(
            brain_atlas=brain_atlas,
            histology_images={},
            lazy_channel_paths={},
        )
    )
    service = AlignmentOutputService(data_context, histology_context)

    service.get_alignment_results_batch(
        {
            ("rec", "stream", 0): (
                np.array([[-0.001, -0.002, 0.003]]),
                np.array([[0.0, 0.0]]),
            )
        }
    )

    assert len(calls) == 1
    assert calls[0].loc[0, "x"] == expected_x
    assert calls[0].loc[0, "y"] == 2.0
    assert calls[0].loc[0, "z"] == 3.0
