"""Detect the image frame expected by image-to-template ANTs transforms."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import AnatomicalHeader
from aind_ants_transform_sidecar import Domain, load_package
from aind_registration_utils.domains import ImageDomainAxisAligned, ImageHeader

from ephys_alignment_gui.io.datapackage_loader import MouseRoot

logger = logging.getLogger(__name__)

CcfTransformInputFrame = Literal["pipeline", "spim_native"]
PIPELINE_FRAME: CcfTransformInputFrame = "pipeline"
SPIM_NATIVE_FRAME: CcfTransformInputFrame = "spim_native"
_DOMAIN_ATOL = 1e-6


@dataclass(frozen=True)
class CcfTransformFrameDecision:
    """Chosen physical frame for image-to-template transform input points."""

    frame: CcfTransformInputFrame
    reason: str
    sidecar_path: Path | None = None


def detect_ccf_transform_input_frame(
    mouse_root: MouseRoot,
    *,
    spim_native_image: sitk.Image,
    pipeline_header: AnatomicalHeader,
) -> CcfTransformFrameDecision:
    """Detect whether CCF export points should be regridded to pipeline space.

    The image-to-template transform sidecar, when present, records the moving
    domain used during registration. Matching it against the SPIM-native and
    pipeline-anchored candidate image domains tells us whether the GUI must
    preserve the current pipeline regrid or skip it for override registrations
    that were computed directly in SPIM-native space.
    """
    sidecar_path = find_image_to_template_sidecar(
        mouse_root.transforms.image_to_template_affine
    )
    if sidecar_path is None:
        return CcfTransformFrameDecision(
            frame=PIPELINE_FRAME,
            reason=(
                "no image-to-template transform sidecar found; defaulting to "
                "pipeline geometry"
            ),
        )

    sidecar = load_package(sidecar_path.read_text())
    moving_domain = sidecar.moving_domain
    if moving_domain is None:
        return CcfTransformFrameDecision(
            frame=PIPELINE_FRAME,
            reason=(
                f"{sidecar_path.name} has no moving_domain; defaulting to "
                "pipeline geometry"
            ),
            sidecar_path=sidecar_path,
        )

    native_domain = _domain_from_sitk_image(spim_native_image)
    pipeline_domain = _domain_from_anatomical_header(pipeline_header)
    native_distance = _domain_distance(moving_domain, native_domain)
    pipeline_distance = _domain_distance(moving_domain, pipeline_domain)

    native_matches = _domains_close(moving_domain, native_domain)
    pipeline_matches = _domains_close(moving_domain, pipeline_domain)
    if native_matches and not pipeline_matches:
        frame = SPIM_NATIVE_FRAME
    elif pipeline_matches:
        frame = PIPELINE_FRAME
    elif native_distance < pipeline_distance:
        frame = SPIM_NATIVE_FRAME
    elif pipeline_distance < native_distance:
        frame = PIPELINE_FRAME
    else:
        raise ValueError(
            "Image-to-template transform sidecar moving_domain is ambiguous: "
            f"native_distance={native_distance:.6g}, "
            f"pipeline_distance={pipeline_distance:.6g}, "
            f"sidecar={sidecar_path}"
        )

    reason = (
        f"{sidecar_path.name} moving_domain matched {frame} "
        f"(native_distance={native_distance:.6g}, "
        f"pipeline_distance={pipeline_distance:.6g})"
    )
    logger.info("Detected CCF transform input frame: %s; %s", frame, reason)
    return CcfTransformFrameDecision(
        frame=frame,
        reason=reason,
        sidecar_path=sidecar_path,
    )


def find_image_to_template_sidecar(affine_path: Path) -> Path | None:
    """Return the first known sidecar path for an image-to-template affine."""
    for candidate in _image_to_template_sidecar_candidates(affine_path):
        if candidate.is_file():
            return candidate
    return None


def _image_to_template_sidecar_candidates(affine_path: Path) -> tuple[Path, ...]:
    parent = affine_path.parent
    name = affine_path.name
    stem = affine_path.stem
    prefixes: list[str] = []
    for suffix in (
        "_SyN_0GenericAffine.mat",
        "_0GenericAffine.mat",
        "0GenericAffine.mat",
    ):
        if name.endswith(suffix):
            prefix = name[: -len(suffix)].rstrip("_")
            if prefix:
                prefixes.append(prefix)
    prefixes.append(stem)

    names = ["ls_to_template_transform_information.json", "transform_information.json"]
    for prefix in prefixes:
        names.append(f"{prefix}_transform_information.json")
        names.append(f"{prefix}.json")

    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate_name in names:
        candidate = parent / candidate_name
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return tuple(ordered)


def _domain_from_sitk_image(image: sitk.Image) -> Domain:
    return ImageDomainAxisAligned.from_header(ImageHeader.from_sitk(image)).to_sidecar()


def _domain_from_anatomical_header(header: AnatomicalHeader) -> Domain:
    image_header = ImageHeader(
        origin=header.origin,
        spacing=header.spacing,
        direction=header.direction_tuple(),
        size=header.size_ijk,
    )
    return ImageDomainAxisAligned.from_header(image_header).to_sidecar()


def _domains_close(left: Domain, right: Domain) -> bool:
    return _domain_distance(left, right) <= _DOMAIN_ATOL


def _domain_distance(left: Domain, right: Domain) -> float:
    left_values = _domain_numeric_values(left)
    right_values = _domain_numeric_values(right)
    value_distance = float(np.max(np.abs(left_values - right_values)))
    if (
        left.shape_canonical is not None
        and right.shape_canonical is not None
        and tuple(left.shape_canonical) != tuple(right.shape_canonical)
    ):
        return value_distance + 1e6
    return value_distance


def _domain_numeric_values(domain: Domain) -> np.ndarray:
    return np.asarray(
        [
            *domain.spacing_LPS,
            *domain.bbox.L,
            *domain.bbox.P,
            *domain.bbox.S,
        ],
        dtype=np.float64,
    )
