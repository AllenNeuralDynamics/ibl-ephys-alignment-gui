"""Tests for histology runtime data boundaries."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import AnatomicalHeader

from ephys_alignment_gui.services.histology_data import (
    HistologyDataContext,
    HistologyRuntimeData,
    _load_pipeline_geometry_image,
    _pipeline_geometry_stub_from_sidecar,
)

IRP_DIRECTION = (0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0)


def test_histology_data_context_stores_runtime_data(tmp_path) -> None:
    image = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.uint8))
    brain_atlas = object()
    histology_data = HistologyRuntimeData(
        brain_atlas=brain_atlas,
        histology_images={"histology_registration": image},
        lazy_channel_paths={"fluor": tmp_path / "fluor.nii.gz"},
    )
    histology_context = HistologyDataContext()

    histology_context.set(histology_data)

    assert histology_context.brain_atlas is brain_atlas
    assert histology_context.histology_images == {"histology_registration": image}
    assert histology_context.lazy_channel_paths == {"fluor": tmp_path / "fluor.nii.gz"}


def _volume() -> sitk.Image:
    img = sitk.GetImageFromArray(np.zeros((7, 9, 11), dtype=np.uint16))
    img.SetSpacing((0.030, 0.031, 0.032))
    img.SetOrigin((11.82, -1.5, 1.5))
    img.SetDirection(IRP_DIRECTION)
    return img


def _write_sidecar(path, img: sitk.Image, **overrides) -> None:
    header = AnatomicalHeader.from_sitk(img)
    payload = {
        "schema": "anatomical-header/1",
        "space": "left-posterior-superior",
        "units": "millimeter",
        "header": {
            "origin": [float(v) for v in header.origin],
            "spacing": [float(v) for v in header.spacing],
            "direction": [float(v) for v in header.direction_tuple()],
            "size_ijk": [int(v) for v in header.size_ijk],
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "index",
    [(0, 0, 0), (10, 8, 6), (5.5, 2.25, 3.75), (99.0, -12.0, 3.0)],
)
def test_pipeline_geometry_sidecar_stub_maps_like_full_volume(tmp_path, index) -> None:
    volume = _volume()
    sidecar = tmp_path / "histology_registration_pipeline.json"
    _write_sidecar(sidecar, volume)

    stub = _pipeline_geometry_stub_from_sidecar(sidecar, volume)

    assert stub.GetSize() == (1, 1, 1)
    assert stub.GetSpacing() == pytest.approx(volume.GetSpacing())
    assert stub.GetOrigin() == pytest.approx(volume.GetOrigin())
    assert stub.GetDirection() == pytest.approx(volume.GetDirection())
    assert stub.TransformContinuousIndexToPhysicalPoint(index) == pytest.approx(
        volume.TransformContinuousIndexToPhysicalPoint(index)
    )


def test_pipeline_geometry_sidecar_size_must_match_base_image(tmp_path) -> None:
    volume = _volume()
    base_image = sitk.Image([1, 2, 3], sitk.sitkUInt8)
    sidecar = tmp_path / "histology_registration_pipeline.json"
    _write_sidecar(sidecar, volume)

    with pytest.raises(ValueError, match="does not match base image"):
        _pipeline_geometry_stub_from_sidecar(sidecar, base_image)


def test_pipeline_geometry_sidecar_declares_units_and_space(tmp_path) -> None:
    volume = _volume()
    sidecar = tmp_path / "histology_registration_pipeline.json"
    _write_sidecar(sidecar, volume, units="micrometer")

    with pytest.raises(ValueError, match="Unsupported pipeline geometry units"):
        _pipeline_geometry_stub_from_sidecar(sidecar, volume)


def test_pipeline_geometry_sidecar_avoids_opening_pipeline_volume(tmp_path) -> None:
    histology = _volume()
    sidecar = tmp_path / "histology_registration_pipeline.json"
    _write_sidecar(sidecar, histology)
    hist = SimpleNamespace(
        registration_pipeline=None,
        registration_pipeline_geometry=sidecar,
    )

    stub = _load_pipeline_geometry_image(hist, histology)

    assert stub.GetSize() == (1, 1, 1)


def test_pipeline_geometry_sidecar_validates_present_pipeline_volume(tmp_path) -> None:
    volume = _volume()
    pipeline_path = tmp_path / "histology_registration_pipeline.nrrd"
    sitk.WriteImage(volume, str(pipeline_path))
    sidecar = tmp_path / "histology_registration_pipeline.json"
    _write_sidecar(sidecar, volume)
    hist = SimpleNamespace(
        registration_pipeline=pipeline_path,
        registration_pipeline_geometry=sidecar,
    )

    stub = _load_pipeline_geometry_image(hist, volume)

    assert stub.TransformContinuousIndexToPhysicalPoint((5.0, 2.0, 1.0)) == (
        pytest.approx(volume.TransformContinuousIndexToPhysicalPoint((5.0, 2.0, 1.0)))
    )


def test_pipeline_geometry_sidecar_rejects_pipeline_volume_drift(tmp_path) -> None:
    sidecar_source = _volume()
    volume = _volume()
    volume.SetOrigin((99.0, -1.5, 1.5))
    pipeline_path = tmp_path / "histology_registration_pipeline.nrrd"
    sitk.WriteImage(volume, str(pipeline_path))
    sidecar = tmp_path / "histology_registration_pipeline.json"
    _write_sidecar(sidecar, sidecar_source)
    hist = SimpleNamespace(
        registration_pipeline=pipeline_path,
        registration_pipeline_geometry=sidecar,
    )

    with pytest.raises(ValueError, match="origin"):
        _load_pipeline_geometry_image(hist, sidecar_source)
