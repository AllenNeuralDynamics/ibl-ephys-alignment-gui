"""Anatomical and histology slice runtime models."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from ephys_alignment_gui.geometry.anatomical_atlas import _BLESSED_DIRECTION
from ephys_alignment_gui.geometry.perpendicular_slice import build_perpendicular_slice
from ephys_alignment_gui.geometry.rigid_rotation import (
    display_spacing_mm_from_image,
    rotate_image,
)

logger = logging.getLogger(__name__)

_METADATA_KEYS = frozenset({"ccf", "label", "annotation_ids", "scale", "offset"})


class SliceSet(dict):
    """Dict-compatible anatomical slice collection.

    The GUI still treats slice data as a mapping. This class keeps that API but
    gives the object a domain name and owns lazy channel slicing.
    """

    def __init__(
        self,
        eager_data: Mapping[str, Any],
        lazy_channel_names: list[str],
        load_slice_callback: Callable[[str], NDArray],
    ) -> None:
        super().__init__(eager_data)
        self._lazy_channel_names = set(lazy_channel_names)
        self._load_slice_callback = load_slice_callback

        for channel in lazy_channel_names:
            if channel not in self:
                super().__setitem__(channel, None)

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        if key in self._lazy_channel_names and value is None:
            logger.info("Lazy loading and slicing channel: %s", key)
            value = self._load_slice_callback(key)
            super().__setitem__(key, value)
        return value

    @property
    def annotation_ids(self) -> NDArray | None:
        """Raw CCF annotation-ID slice, used for brain masking."""
        return self.get("annotation_ids")

    @property
    def image_channels(self) -> list[str]:
        """Slice keys that represent plottable image channels."""
        return [key for key in self.keys() if key not in _METADATA_KEYS]


@dataclass
class SliceService:
    """Build and sample anatomical/histology slices in atlas space."""

    max_loaded_histology_images: int | None = 3

    def __post_init__(self) -> None:
        if (
            self.max_loaded_histology_images is not None
            and self.max_loaded_histology_images < 1
        ):
            raise ValueError("max_loaded_histology_images must be positive or None")

    def build_slice_set(
        self,
        *,
        brain_atlas: Any,
        histology_images: dict[str, sitk.Image],
        lazy_channel_paths: Mapping[str, Path] | None,
        track_interpolation_ras: NDArray,
    ) -> SliceSet:
        """Build the coronal slice set for one aligned shank track."""
        index = brain_atlas.physical_points_to_indices(
            track_interpolation_ras, round=True
        )

        ccf_slice = cut_slice_from_atlas_image(brain_atlas.image, index)
        annotation_slice = cut_slice_from_atlas_image(brain_atlas.label, index)
        label_slice = brain_atlas._label2rgb(annotation_slice)
        x_dimno = brain_atlas.xyz2dims[0]
        width = [
            brain_atlas.bc.i2x(0),
            brain_atlas.bc.i2x(brain_atlas.image.shape[x_dimno]),
        ]
        height = [
            brain_atlas.bc.i2z(index[0, 2]),
            brain_atlas.bc.i2z(index[-1, 2]),
        ]

        eager_data = {
            "ccf": ccf_slice,
            "label": label_slice,
            "annotation_ids": annotation_slice,
            "scale": np.array(
                [
                    (width[-1] - width[0]) / ccf_slice.shape[0],
                    (height[-1] - height[0]) / ccf_slice.shape[1],
                ]
            ),
            "offset": np.array([width[0], height[0]]),
        }

        if "histology_registration" in histology_images:
            eager_data["histology_registration"] = self.slice_channel(
                brain_atlas=brain_atlas,
                histology_images=histology_images,
                lazy_channel_paths=lazy_channel_paths,
                channel_name="histology_registration",
                index=index,
            )

        lazy_channel_names = list(lazy_channel_paths or {})
        if lazy_channel_names:
            logger.debug(
                "Setting up lazy slicing for %d channels", len(lazy_channel_names)
            )

        return SliceSet(
            eager_data=eager_data,
            lazy_channel_names=lazy_channel_names,
            load_slice_callback=lambda channel_name: self.slice_channel(
                brain_atlas=brain_atlas,
                histology_images=histology_images,
                lazy_channel_paths=lazy_channel_paths,
                channel_name=channel_name,
                index=index,
            ),
        )

    def slice_channel(
        self,
        *,
        brain_atlas: Any,
        histology_images: dict[str, sitk.Image],
        lazy_channel_paths: Mapping[str, Path] | None,
        channel_name: str,
        index: NDArray,
    ) -> NDArray:
        """Return one histology channel sliced at already-computed indices."""
        hist_image = self.load_channel_image(
            brain_atlas=brain_atlas,
            histology_images=histology_images,
            lazy_channel_paths=lazy_channel_paths,
            channel_name=channel_name,
        )
        hist_arr = sitk.GetArrayViewFromImage(hist_image)
        hist_slice = cut_slice_from_atlas_image(hist_arr, index)
        logger.debug("Computed slice for %s: shape %s", channel_name, hist_slice.shape)
        return hist_slice

    def volume_for_channel(
        self,
        *,
        brain_atlas: Any,
        histology_images: dict[str, sitk.Image],
        lazy_channel_paths: Mapping[str, Path] | None,
        channel_name: str,
    ) -> NDArray:
        """Return a scalar 3-D volume for perpendicular sampling."""
        if channel_name == "ccf":
            return brain_atlas.image

        hist_image = self.load_channel_image(
            brain_atlas=brain_atlas,
            histology_images=histology_images,
            lazy_channel_paths=lazy_channel_paths,
            channel_name=channel_name,
        )
        return sitk.GetArrayViewFromImage(hist_image)

    def build_perpendicular_slice_image(
        self,
        *,
        brain_atlas: Any,
        histology_images: dict[str, sitk.Image],
        lazy_channel_paths: Mapping[str, Path] | None,
        ephysalign: Any,
        feature_ref: NDArray,
        track_ref: NDArray,
        feature_grid_m: NDArray,
        channel_name: str,
        extent_m: float = 500e-6,
        n_perp_samples: int = 41,
        sigma_samples: float = 2.0,
    ) -> NDArray[np.float64]:
        """Build a perpendicular slice image for the requested scalar channel."""
        volume_arr = self.volume_for_channel(
            brain_atlas=brain_atlas,
            histology_images=histology_images,
            lazy_channel_paths=lazy_channel_paths,
            channel_name=channel_name,
        )

        return build_perpendicular_slice(
            volume_arr=volume_arr,
            brain_atlas=brain_atlas,
            track_interpolation_ras=ephysalign.track_interpolation_ras,
            ephys_depths_along_track=ephysalign.ephys_depths_along_track,
            feature_ref=np.asarray(feature_ref, dtype=np.float64),
            track_ref=np.asarray(track_ref, dtype=np.float64),
            feature_grid_m=np.asarray(feature_grid_m, dtype=np.float64),
            extent_m=extent_m,
            n_perp_samples=n_perp_samples,
            sigma_samples=sigma_samples,
        )

    def load_channel_image(
        self,
        *,
        brain_atlas: Any,
        histology_images: dict[str, sitk.Image],
        lazy_channel_paths: Mapping[str, Path] | None,
        channel_name: str,
    ) -> sitk.Image:
        """Load and cache a histology channel image in canonical atlas space."""
        if channel_name in histology_images:
            logger.debug("Using cached image for %s", channel_name)
            channel_image = self._touch_loaded_image(histology_images, channel_name)
            self._enforce_loaded_image_limit(histology_images)
            return channel_image

        lazy_channel_paths = lazy_channel_paths or {}
        if channel_name not in lazy_channel_paths:
            raise ValueError(f"Unknown channel: {channel_name}")

        channel_path = lazy_channel_paths[channel_name]
        logger.info("Loading channel image from disk: %s", channel_path.name)
        channel_image = sitk.ReadImage(str(channel_path))

        if (
            brain_atlas is not None
            and brain_atlas.display_rotation is not None
            and brain_atlas.display_rotation_center is not None
        ):
            spacing_mm = _canonical_display_spacing_mm(brain_atlas, channel_image)
            channel_image = rotate_image(
                channel_image,
                brain_atlas.display_rotation,
                brain_atlas.display_rotation_center,
                spacing_mm=spacing_mm,
                interpolator="linear",
            )

        dicom_orient_str = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                channel_image.GetDirection()
            )
        )
        if dicom_orient_str != _BLESSED_DIRECTION:
            channel_image = sitk.DICOMOrient(channel_image, _BLESSED_DIRECTION)

        histology_images[channel_name] = channel_image
        self._touch_loaded_image(histology_images, channel_name)
        self._enforce_loaded_image_limit(histology_images)
        logger.debug("Cached %s in histology_images", channel_name)
        return channel_image

    @staticmethod
    def _touch_loaded_image(
        histology_images: dict[str, sitk.Image],
        channel_name: str,
    ) -> sitk.Image:
        """Mark one cached histology image as recently used."""
        image = histology_images.pop(channel_name)
        histology_images[channel_name] = image
        return image

    def _enforce_loaded_image_limit(
        self,
        histology_images: dict[str, sitk.Image],
    ) -> None:
        """Evict least-recently-used histology images until within budget."""
        if self.max_loaded_histology_images is None:
            return

        while len(histology_images) > self.max_loaded_histology_images:
            oldest_key = next(iter(histology_images))
            histology_images.pop(oldest_key, None)


def _canonical_display_spacing_mm(
    brain_atlas: Any,
    source_image: sitk.Image,
) -> float:
    """Return the canonical display spacing for lazy histology channels."""
    atlas_image = getattr(brain_atlas, "intensity_sitk_image", None)
    if atlas_image is not None:
        return display_spacing_mm_from_image(atlas_image)
    return display_spacing_mm_from_image(source_image)


def cut_slice_from_atlas_image(
    atlas_array: NDArray,
    xyz_channel_indices: NDArray,
    func: Callable[[NDArray], NDArray] | None = None,
) -> NDArray:
    """Extract a tilted slice from an atlas-oriented image volume."""
    slice_image = atlas_array[xyz_channel_indices[:, 0], :, xyz_channel_indices[:, 2]]
    if func is not None:
        slice_image = func(slice_image)
    return np.swapaxes(slice_image, 0, 1)
