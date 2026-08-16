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
        self.intensity_sitk_image = sitk.GetImageFromArray(self.image)
        self.intensity_sitk_image.SetSpacing((0.030, 0.030, 0.030))

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


def test_volume_for_channel_uses_array_view(monkeypatch, tmp_path) -> None:
    atlas = _FakeAtlas()
    service = SliceService()
    channel = atlas.image + 3000
    channel_path = _write_image(tmp_path / "fluor.mha", channel)

    def fail_copy(_image):
        raise AssertionError("volume_for_channel should not copy SITK image arrays")

    monkeypatch.setattr(
        "ephys_alignment_gui.services.slice.sitk.GetArrayFromImage",
        fail_copy,
    )
    histology_images = {}

    volume = service.volume_for_channel(
        brain_atlas=atlas,
        histology_images=histology_images,
        lazy_channel_paths={"fluor": channel_path},
        channel_name="fluor",
    )

    np.testing.assert_array_equal(volume, channel)
    assert "fluor" in histology_images


def test_load_channel_image_evicts_least_recently_used_histology_images(tmp_path) -> None:
    atlas = _FakeAtlas()
    service = SliceService(max_loaded_histology_images=3)
    histology_images = {
        "histology_registration": sitk.GetImageFromArray(atlas.image + 1000)
    }
    paths = {
        name: _write_image(tmp_path / f"{name}.mha", atlas.image + offset)
        for name, offset in [
            ("a", 2000),
            ("b", 3000),
            ("c", 4000),
            ("d", 5000),
        ]
    }

    service.load_channel_image(
        brain_atlas=atlas,
        histology_images=histology_images,
        lazy_channel_paths=paths,
        channel_name="a",
    )
    service.load_channel_image(
        brain_atlas=atlas,
        histology_images=histology_images,
        lazy_channel_paths=paths,
        channel_name="b",
    )
    service.load_channel_image(
        brain_atlas=atlas,
        histology_images=histology_images,
        lazy_channel_paths=paths,
        channel_name="c",
    )

    assert list(histology_images) == ["a", "b", "c"]

    service.load_channel_image(
        brain_atlas=atlas,
        histology_images=histology_images,
        lazy_channel_paths=paths,
        channel_name="a",
    )
    service.load_channel_image(
        brain_atlas=atlas,
        histology_images=histology_images,
        lazy_channel_paths=paths,
        channel_name="d",
    )

    assert list(histology_images) == ["c", "a", "d"]


def test_lazy_channel_rotation_uses_atlas_canonical_spacing(
    monkeypatch,
    tmp_path,
) -> None:
    atlas = _FakeAtlas()
    atlas.display_rotation = np.eye(3)
    atlas.display_rotation_center = np.zeros(3)
    atlas.intensity_sitk_image.SetSpacing((0.030, 0.030, 0.030))
    service = SliceService()
    channel = atlas.image + 4000
    channel_path = _write_image(tmp_path / "fluor.mha", channel)
    channel_image = sitk.ReadImage(str(channel_path))
    channel_image.SetSpacing((0.010, 0.010, 0.010))
    sitk.WriteImage(channel_image, str(channel_path))
    calls: list[float | None] = []

    def fake_rotate_image(
        image,
        rotation,
        center,
        *,
        spacing_mm=None,
        interpolator="linear",
        default_value=0.0,
    ):
        calls.append(spacing_mm)
        return image

    monkeypatch.setattr("ephys_alignment_gui.services.slice.rotate_image", fake_rotate_image)

    service.load_channel_image(
        brain_atlas=atlas,
        histology_images={},
        lazy_channel_paths={"fluor": channel_path},
        channel_name="fluor",
    )

    assert calls == [0.030]
