"""Qt-free service for loading atlas and histology runtime data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import AnatomicalHeader

from ephys_alignment_gui.geometry.anatomical_atlas import (
    _BLESSED_DIRECTION,
    BrainAtlasAnatomical,
)
from ephys_alignment_gui.geometry.rigid_rotation import (
    display_spacing_mm_from_image,
    image_center_physical,
    load_affine_matrix,
    polar_rotation,
    rotate_image,
)
from ephys_alignment_gui.io.datapackage_loader import HistologyImagePaths, MouseRoot

logger = logging.getLogger(__name__)
_GEOMETRY_SCHEMA = "anatomical-header/1"
_GEOMETRY_SPACE = "left-posterior-superior"
_GEOMETRY_UNITS = "millimeter"


@dataclass(frozen=True)
class HistologyRuntimeData:
    """Runtime atlas/histology data loaded from a mouse root."""

    brain_atlas: BrainAtlasAnatomical
    histology_images: dict[str, sitk.Image]
    lazy_channel_paths: dict[str, Path]


@dataclass
class HistologyDataContext:
    """Mutable holder for currently loaded histology runtime data."""

    runtime_data: HistologyRuntimeData | None = None
    source_mouse_root: Path | None = None

    def set(
        self,
        data: HistologyRuntimeData,
        *,
        mouse_root: MouseRoot | Path | None = None,
    ) -> None:
        """Store loaded histology runtime data."""
        self.runtime_data = data
        if isinstance(mouse_root, MouseRoot):
            self.source_mouse_root = mouse_root.root
        elif mouse_root is not None:
            self.source_mouse_root = Path(mouse_root)
        else:
            self.source_mouse_root = None

    def clear(self) -> None:
        """Clear loaded histology runtime data."""
        self.runtime_data = None
        self.source_mouse_root = None

    def is_loaded_for(self, mouse_root: MouseRoot | Path) -> bool:
        """Return whether histology is loaded for one mouse root."""
        if self.runtime_data is None:
            return False
        if self.source_mouse_root is None:
            return True
        if isinstance(mouse_root, MouseRoot):
            return self.source_mouse_root == mouse_root.root
        return self.source_mouse_root == Path(mouse_root)

    @property
    def brain_atlas(self) -> BrainAtlasAnatomical | None:
        """Loaded anatomical atlas, if available."""
        if self.runtime_data is None:
            return None
        return self.runtime_data.brain_atlas

    @property
    def histology_images(self) -> dict[str, sitk.Image]:
        """Loaded histology image channels."""
        if self.runtime_data is None:
            return {}
        return self.runtime_data.histology_images

    @property
    def lazy_channel_paths(self) -> dict[str, Path]:
        """Additional histology channels available for lazy loading."""
        if self.runtime_data is None:
            return {}
        return self.runtime_data.lazy_channel_paths


class HistologyDataService:
    """Load atlas and histology images for a resolved mouse root."""

    def load(self, mouse_root: MouseRoot) -> HistologyRuntimeData:
        """Load atlas + default histology channel from datapackage paths."""
        hist = mouse_root.histology
        logger.debug("Loading atlas and histology from %s", hist.registration.parent)

        intensity_image = sitk.ReadImage(str(hist.ccf_template))
        label_image = sitk.ReadImage(str(hist.labels))
        histology_image = sitk.ReadImage(str(hist.registration))
        pipeline_image = _load_pipeline_geometry_image(hist, intensity_image)

        # Extract the rotational part of the SPIM->template affine and apply
        # it to every image-space asset, so the canonical in-memory frame has
        # atlas-aligned axes. SPIM-native recovery (for saving xyz_picks and
        # composing with the ANTs CCF chain) is done via R^T through the
        # BrainAtlasAnatomical.unrotate_to_spim_native helper.
        linear, _ = load_affine_matrix(mouse_root.transforms.image_to_template_affine)
        # An ANTs 0GenericAffine.mat maps points fixed->moving. The
        # ls_to_template registration has fixed=template, moving=SPIM, so this
        # linear part is the template->SPIM map. We want to rotate SPIM data
        # *into* the template-aligned canonical frame, i.e. the SPIM->template
        # rotation, which is the transpose (inverse) of the extracted rotation.
        R = polar_rotation(linear).T
        rotation_center = image_center_physical(intensity_image)
        display_spacing_mm = display_spacing_mm_from_image(intensity_image)
        logger.debug(
            "Computed SPIM->template display rotation (det=%.6f, spacing=%.3f mm)",
            np.linalg.det(R),
            display_spacing_mm,
        )

        intensity_image_rot = rotate_image(
            intensity_image,
            R,
            rotation_center,
            spacing_mm=display_spacing_mm,
            interpolator="linear",
        )
        label_image_rot = rotate_image(
            label_image,
            R,
            rotation_center,
            spacing_mm=display_spacing_mm,
            interpolator="nearest",
        )
        histology_image_rot = rotate_image(
            histology_image,
            R,
            rotation_center,
            spacing_mm=display_spacing_mm,
            interpolator="linear",
        )

        brain_atlas = BrainAtlasAnatomical(
            intensity_img=intensity_image_rot,
            label_img=label_image_rot,
            display_rotation=R,
            display_rotation_center=rotation_center,
            intensity_img_spim_native=intensity_image,
            pipeline_img_spim_native=pipeline_image,
        )

        # Ensure the rotated histology is in the blessed DICOM orientation
        # consumed by the rest of the pipeline (rotate_image emits identity
        # direction, so DICOMOrient only does a cheap axis permutation).
        dicom_orient_str = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                histology_image_rot.GetDirection()
            )
        )
        if dicom_orient_str != _BLESSED_DIRECTION:
            histology_image_rot = sitk.DICOMOrient(
                histology_image_rot, _BLESSED_DIRECTION
            )

        lazy_channel_paths = dict(hist.additional_channels)
        logger.debug("Setup lazy loading for %d channels", len(lazy_channel_paths))
        return HistologyRuntimeData(
            brain_atlas=brain_atlas,
            histology_images={"histology_registration": histology_image_rot},
            lazy_channel_paths=lazy_channel_paths,
        )


def _load_pipeline_geometry_image(
    hist: HistologyImagePaths,
    base_image: sitk.Image,
) -> sitk.Image:
    """Load the pipeline geometry image, preferring the sidecar when present."""
    sidecar_path = hist.registration_pipeline_geometry
    if sidecar_path is not None:
        header = _pipeline_geometry_header_from_sidecar(sidecar_path, base_image)
        if hist.registration_pipeline is not None:
            _validate_pipeline_volume_matches_header(hist.registration_pipeline, header)
        return header.as_sitk_stub()

    if hist.registration_pipeline is not None:
        return sitk.ReadImage(str(hist.registration_pipeline))

    raise ValueError(
        "Datapackage histology image_space must include either "
        "registration_pipeline_geometry or registration_pipeline"
    )


def _pipeline_geometry_stub_from_sidecar(
    sidecar_path: Path,
    base_image: sitk.Image,
) -> sitk.Image:
    """Rehydrate a pipeline geometry sidecar as a 1x1x1 SimpleITK stub."""
    return _pipeline_geometry_header_from_sidecar(
        sidecar_path, base_image
    ).as_sitk_stub()


def _pipeline_geometry_header_from_sidecar(
    sidecar_path: Path,
    base_image: sitk.Image,
) -> AnatomicalHeader:
    """Rehydrate a pipeline geometry sidecar as an anatomical header."""
    payload = _load_geometry_payload(sidecar_path)
    _validate_geometry_payload_conventions(payload, sidecar_path)
    header_payload = payload["header"]
    header = _anatomical_header_from_payload(header_payload, sidecar_path)
    if header.size_ijk != tuple(base_image.GetSize()):
        raise ValueError(
            f"pipeline geometry {header.size_ijk} does not match base image "
            f"{base_image.GetSize()}; the index handed between frames "
            "would be wrong"
        )
    return header


def _validate_pipeline_volume_matches_header(
    pipeline_path: Path,
    header: AnatomicalHeader,
) -> None:
    """Assert the 3.2 transition sidecar agrees with the still-shipped volume."""
    volume = sitk.ReadImage(str(pipeline_path))
    if header.size_ijk != tuple(volume.GetSize()):
        raise ValueError(
            f"pipeline geometry {header.size_ijk} does not match pipeline volume "
            f"{volume.GetSize()}"
        )
    if not np.allclose(header.spacing, volume.GetSpacing()):
        raise ValueError(
            f"pipeline geometry spacing {header.spacing} does not match "
            f"pipeline volume {volume.GetSpacing()}"
        )
    if not np.allclose(header.origin, volume.GetOrigin()):
        raise ValueError(
            f"pipeline geometry origin {header.origin} does not match "
            f"pipeline volume {volume.GetOrigin()}"
        )
    if not np.allclose(header.direction_tuple(), volume.GetDirection()):
        raise ValueError(
            "pipeline geometry direction does not match pipeline volume "
            f"{volume.GetDirection()}"
        )


def _load_geometry_payload(sidecar_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Malformed pipeline geometry sidecar {sidecar_path}: {e}"
        ) from e
    if not isinstance(payload, dict):
        raise ValueError(
            f"Pipeline geometry sidecar {sidecar_path} must contain a JSON object"
        )
    return payload


def _validate_geometry_payload_conventions(
    payload: dict[str, object],
    sidecar_path: Path,
) -> None:
    if payload.get("schema") != _GEOMETRY_SCHEMA:
        raise ValueError(
            f"Unsupported pipeline geometry schema in {sidecar_path}: "
            f"{payload.get('schema')!r}"
        )
    if payload.get("space") != _GEOMETRY_SPACE:
        raise ValueError(
            f"Unsupported pipeline geometry space in {sidecar_path}: "
            f"{payload.get('space')!r}"
        )
    if payload.get("units") != _GEOMETRY_UNITS:
        raise ValueError(
            f"Unsupported pipeline geometry units in {sidecar_path}: "
            f"{payload.get('units')!r}"
        )
    if not isinstance(payload.get("header"), dict):
        raise ValueError(
            f"Pipeline geometry sidecar {sidecar_path} missing header object"
        )


def _anatomical_header_from_payload(
    header_payload: object,
    sidecar_path: Path,
) -> AnatomicalHeader:
    if not isinstance(header_payload, dict):
        raise ValueError(
            f"Pipeline geometry sidecar {sidecar_path} missing header object"
        )
    try:
        origin = _float_triplet(header_payload["origin"], "origin")
        spacing = _float_triplet(header_payload["spacing"], "spacing")
        direction = np.array(header_payload["direction"], dtype=np.float64).reshape(
            3, 3
        )
        size_ijk = _int_triplet(header_payload["size_ijk"], "size_ijk")
    except KeyError as e:
        raise ValueError(
            f"Pipeline geometry sidecar {sidecar_path} missing header field {e}"
        ) from e
    except ValueError as e:
        raise ValueError(
            f"Invalid pipeline geometry sidecar {sidecar_path}: {e}"
        ) from e
    return AnatomicalHeader(
        origin=origin,
        spacing=spacing,
        direction=direction,
        size_ijk=size_ijk,
    )


def _float_triplet(
    value: object,
    field: str,
) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{field} must be a length-3 sequence")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field} must contain numeric values") from e


def _int_triplet(
    value: object,
    field: str,
) -> tuple[int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{field} must be a length-3 sequence")
    try:
        return (int(value[0]), int(value[1]), int(value[2]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field} must contain integer values") from e
