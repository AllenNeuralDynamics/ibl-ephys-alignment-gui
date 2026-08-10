"""Tests for anatomical slice runtime models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ephys_alignment_gui.geometry.anatomical_atlas import _BLESSED_DIRECTION
from ephys_alignment_gui.services.slice import (
    SliceService,
    SliceSet,
    cut_slice_from_atlas_image,
)


class _FakeBrainCoordinates:
    def i2x(self, index):
        return float(index)

    def i2z(self, index):
        return float(index)


class _FakeAtlas:
    def __init__(self) -> None:
        self.image = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        self.label = np.zeros((3, 4, 5), dtype=np.uint16)
        self.label[1, :, 2] = 997
        self.xyz2dims = (0, 1, 2)
        self.bc = _FakeBrainCoordinates()
        self.display_rotation = None
        self.display_rotation_center = None

    def physical_points_to_indices(self, points, round=True):
        return np.asarray(points, dtype=np.int64)

    def _label2rgb(self, annotation_slice):
        return np.repeat(annotation_slice[..., None], 3, axis=2).astype(np.uint8)


def _write_image(path: Path, image: np.ndarray) -> Path:
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetDirection(
        sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
            _BLESSED_DIRECTION
        )
    )
    sitk.WriteImage(sitk_image, str(path))
    return path


def test_cut_slice_from_atlas_image_uses_track_xz_indices() -> None:
    volume = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    indices = np.array([[0, 0, 0], [1, 0, 2], [2, 0, 4]])

    image_slice = cut_slice_from_atlas_image(volume, indices)

    expected = np.swapaxes(volume[[0, 1, 2], :, [0, 2, 4]], 0, 1)
    np.testing.assert_array_equal(image_slice, expected)


def test_build_slice_set_returns_ccf_annotation_and_histology_slices() -> None:
    atlas = _FakeAtlas()
    histology = atlas.image + 1000
    service = SliceService()

    slice_set = service.build_slice_set(
        brain_atlas=atlas,
        histology_images={"histology_registration": sitk.GetImageFromArray(histology)},
        lazy_channel_paths={},
        track_interpolation_ras=np.array([[0, 0, 0], [1, 0, 2], [2, 0, 4]]),
    )

    assert isinstance(slice_set, SliceSet)
    np.testing.assert_array_equal(
        slice_set["ccf"],
        np.swapaxes(atlas.image[[0, 1, 2], :, [0, 2, 4]], 0, 1),
    )
    np.testing.assert_array_equal(
        slice_set.annotation_ids,
        np.swapaxes(atlas.label[[0, 1, 2], :, [0, 2, 4]], 0, 1),
    )
    np.testing.assert_array_equal(
        slice_set["histology_registration"],
        np.swapaxes(histology[[0, 1, 2], :, [0, 2, 4]], 0, 1),
    )
    assert slice_set.image_channels == ["histology_registration"]


def test_slice_set_lazy_channel_loads_on_first_access(tmp_path) -> None:
    atlas = _FakeAtlas()
    service = SliceService()
    channel = atlas.image + 2000
    channel_path = _write_image(tmp_path / "fluor.mha", channel)

    slice_set = service.build_slice_set(
        brain_atlas=atlas,
        histology_images={},
        lazy_channel_paths={"fluor": channel_path},
        track_interpolation_ras=np.array([[0, 0, 0], [1, 0, 2], [2, 0, 4]]),
    )

    assert "fluor" in slice_set
    assert dict.__getitem__(slice_set, "fluor") is None

    np.testing.assert_array_equal(
        slice_set["fluor"],
        np.swapaxes(channel[[0, 1, 2], :, [0, 2, 4]], 0, 1),
    )
    np.testing.assert_array_equal(
        dict.__getitem__(slice_set, "fluor"),
        np.swapaxes(channel[[0, 1, 2], :, [0, 2, 4]], 0, 1),
    )


def test_volume_for_channel_loads_without_existing_slice_index(tmp_path) -> None:
    atlas = _FakeAtlas()
    service = SliceService()
    channel = atlas.image + 3000
    channel_path = _write_image(tmp_path / "fluor.mha", channel)
    histology_images = {}

    volume = service.volume_for_channel(
        brain_atlas=atlas,
        histology_images=histology_images,
        lazy_channel_paths={"fluor": channel_path},
        channel_name="fluor",
    )

    np.testing.assert_array_equal(volume, channel)
    assert "fluor" in histology_images
