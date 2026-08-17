"""Tests for detecting CCF transform input frame from sidecar domains."""

from __future__ import annotations

from types import SimpleNamespace

import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import AnatomicalHeader
from aind_ants_transform_sidecar import SynTriplet, TransformSidecarV1, dump_package
from aind_registration_utils.domains import ImageDomainAxisAligned, ImageHeader

from ephys_alignment_gui.services.ccf_transform_frame import (
    PIPELINE_FRAME,
    SPIM_NATIVE_FRAME,
    detect_ccf_transform_input_frame,
)


def test_detect_ccf_transform_frame_defaults_to_pipeline_without_sidecar(tmp_path):
    affine = tmp_path / "ls_to_template_SyN_0GenericAffine.mat"
    mouse_root = _mouse_root(affine)

    decision = detect_ccf_transform_input_frame(
        mouse_root,
        spim_native_image=_image(origin=(0.0, 0.0, 0.0)),
        pipeline_header=AnatomicalHeader.from_sitk(_image(origin=(10.0, 0.0, 0.0))),
    )

    assert decision.frame == PIPELINE_FRAME
    assert decision.sidecar_path is None
    assert "no image-to-template transform sidecar" in decision.reason


def test_detect_ccf_transform_frame_uses_pipeline_for_pipeline_sidecar(tmp_path):
    affine = tmp_path / "ls_to_template_SyN_0GenericAffine.mat"
    mouse_root = _mouse_root(affine)
    native = _image(origin=(0.0, 0.0, 0.0))
    pipeline = _image(origin=(10.0, 0.0, 0.0))
    _write_transform_sidecar(
        tmp_path / "ls_to_template_transform_information.json",
        moving_image=pipeline,
    )

    decision = detect_ccf_transform_input_frame(
        mouse_root,
        spim_native_image=native,
        pipeline_header=AnatomicalHeader.from_sitk(pipeline),
    )

    assert decision.frame == PIPELINE_FRAME
    assert (
        decision.sidecar_path == tmp_path / "ls_to_template_transform_information.json"
    )
    assert "moving_domain matched pipeline" in decision.reason


def test_detect_ccf_transform_frame_uses_native_for_native_sidecar(tmp_path):
    affine = tmp_path / "ls_to_template_SyN_0GenericAffine.mat"
    mouse_root = _mouse_root(affine)
    native = _image(origin=(0.0, 0.0, 0.0))
    pipeline = _image(origin=(10.0, 0.0, 0.0))
    _write_transform_sidecar(
        tmp_path / "ls_to_template_transform_information.json",
        moving_image=native,
    )

    decision = detect_ccf_transform_input_frame(
        mouse_root,
        spim_native_image=native,
        pipeline_header=AnatomicalHeader.from_sitk(pipeline),
    )

    assert decision.frame == SPIM_NATIVE_FRAME
    assert "moving_domain matched spim_native" in decision.reason


def _image(origin: tuple[float, float, float]) -> sitk.Image:
    image = sitk.Image([4, 5, 6], sitk.sitkUInt8)
    image.SetOrigin(origin)
    image.SetSpacing((0.025, 0.025, 0.025))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _mouse_root(affine_path):
    return SimpleNamespace(
        transforms=SimpleNamespace(image_to_template_affine=affine_path)
    )


def _write_transform_sidecar(path, *, moving_image: sitk.Image) -> None:
    moving_domain = ImageDomainAxisAligned.from_header(
        ImageHeader.from_sitk(moving_image)
    ).to_sidecar()
    sidecar = TransformSidecarV1(
        fixed_domain=moving_domain,
        moving_domain=moving_domain,
        transform=SynTriplet(
            affine="ls_to_template_SyN_0GenericAffine.mat",
            warp="ls_to_template_SyN_1Warp.nii.gz",
            inverse_warp="ls_to_template_SyN_1InverseWarp.nii.gz",
        ),
    )
    path.write_text(dump_package(sidecar))
