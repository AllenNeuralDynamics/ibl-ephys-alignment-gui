"""Qt-free service for loading atlas and histology runtime data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ephys_alignment_gui.geometry.anatomical_atlas import (
    _BLESSED_DIRECTION,
    BrainAtlasAnatomical,
)
from ephys_alignment_gui.geometry.rigid_rotation import (
    image_center_physical,
    load_affine_matrix,
    polar_rotation,
    rotate_image,
)
from ephys_alignment_gui.io.datapackage_loader import MouseRoot

logger = logging.getLogger(__name__)


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
        pipeline_image = sitk.ReadImage(str(hist.registration_pipeline))
        histology_image = sitk.ReadImage(str(hist.registration))

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
        logger.debug(
            "Computed SPIM->template display rotation (det=%.6f)",
            np.linalg.det(R),
        )

        intensity_image_rot = rotate_image(
            intensity_image, R, rotation_center, interpolator="linear"
        )
        label_image_rot = rotate_image(
            label_image, R, rotation_center, interpolator="nearest"
        )
        pipeline_image_rot = rotate_image(
            pipeline_image, R, rotation_center, interpolator="linear"
        )
        histology_image_rot = rotate_image(
            histology_image, R, rotation_center, interpolator="linear"
        )

        brain_atlas = BrainAtlasAnatomical(
            intensity_img=intensity_image_rot,
            label_img=label_image_rot,
            pipeline_img=pipeline_image_rot,
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
