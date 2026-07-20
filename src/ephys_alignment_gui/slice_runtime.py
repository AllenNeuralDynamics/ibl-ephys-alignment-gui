"""Runtime cache for anatomical and perpendicular slice data."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ArrayFingerprint:
    """Stable identity for numeric array content used in slice cache keys."""

    shape: tuple[int, ...]
    digest: str


@dataclass(frozen=True)
class CoronalSliceKey:
    """Cache key for a coronal/track-following slice set."""

    alignment_key: Hashable
    track_interpolation: ArrayFingerprint


@dataclass(frozen=True)
class PerpendicularSliceKey:
    """Cache key for a perpendicular slice image."""

    alignment_key: Hashable
    channel_name: str
    track_interpolation: ArrayFingerprint
    ephys_depths_along_track: ArrayFingerprint
    feature_ref: ArrayFingerprint
    track_ref: ArrayFingerprint
    feature_grid: ArrayFingerprint
    extent_m: float
    n_perp_samples: int
    sigma_samples: float


@dataclass(frozen=True)
class SliceCacheEntry:
    """Cached coronal slice data for one alignment key and track."""

    slice_data: Any
    fp_slice_data: Any


def array_fingerprint(values: Any, *, decimals: int = 9) -> ArrayFingerprint:
    """Return a rounded numeric-array fingerprint for cache identity."""
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.round(arr, decimals=decimals)
    contiguous = np.ascontiguousarray(rounded)
    digest = hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
    return ArrayFingerprint(shape=tuple(contiguous.shape), digest=digest)


class SliceRuntime:
    """Own cached slice arrays for one shank/channel collection runtime."""

    def __init__(self) -> None:
        self._coronal_slices: dict[CoronalSliceKey, SliceCacheEntry] = {}
        self._perpendicular_slices: dict[PerpendicularSliceKey, NDArray] = {}
        self._active_coronal_entry: SliceCacheEntry = SliceCacheEntry(None, None)
        self._active_coronal_key: CoronalSliceKey | None = None

    @property
    def active_slice_data(self) -> Any:
        """Slice data projected for legacy view/session call sites."""
        return self._active_coronal_entry.slice_data

    @property
    def active_fp_slice_data(self) -> Any:
        """Feature-space slice data projected for legacy call sites."""
        return self._active_coronal_entry.fp_slice_data

    def set_active_slice_data(self, slice_data: Any, fp_slice_data: Any) -> None:
        """Set the active legacy slice projection without adding a cache key."""
        self._active_coronal_entry = SliceCacheEntry(slice_data, fp_slice_data)
        self._active_coronal_key = None
        if slice_data is None and fp_slice_data is None:
            self.clear()

    def coronal_key(
        self,
        *,
        alignment_key: Hashable,
        track_interpolation_ras: NDArray,
    ) -> CoronalSliceKey:
        """Build a cache key for a coronal slice set."""
        return CoronalSliceKey(
            alignment_key=alignment_key,
            track_interpolation=array_fingerprint(track_interpolation_ras),
        )

    def cached_coronal_slice(
        self,
        *,
        alignment_key: Hashable,
        track_interpolation_ras: NDArray,
    ) -> SliceCacheEntry | None:
        """Return a cached coronal slice set and make it active."""
        key = self.coronal_key(
            alignment_key=alignment_key,
            track_interpolation_ras=track_interpolation_ras,
        )
        entry = self._coronal_slices.get(key)
        if entry is not None:
            self._active_coronal_key = key
            self._active_coronal_entry = entry
        return entry

    def set_coronal_slice(
        self,
        *,
        alignment_key: Hashable,
        track_interpolation_ras: NDArray,
        slice_data: Any,
        fp_slice_data: Any,
    ) -> SliceCacheEntry:
        """Cache and activate a coronal slice set."""
        key = self.coronal_key(
            alignment_key=alignment_key,
            track_interpolation_ras=track_interpolation_ras,
        )
        entry = SliceCacheEntry(slice_data, fp_slice_data)
        self._coronal_slices[key] = entry
        self._active_coronal_key = key
        self._active_coronal_entry = entry
        return entry

    def get_or_build_coronal_slice(
        self,
        *,
        alignment_key: Hashable,
        track_interpolation_ras: NDArray,
        builder: Callable[[], SliceCacheEntry],
    ) -> SliceCacheEntry:
        """Return a cached coronal slice set or build/cache it."""
        cached = self.cached_coronal_slice(
            alignment_key=alignment_key,
            track_interpolation_ras=track_interpolation_ras,
        )
        if cached is not None:
            return cached
        entry = builder()
        return self.set_coronal_slice(
            alignment_key=alignment_key,
            track_interpolation_ras=track_interpolation_ras,
            slice_data=entry.slice_data,
            fp_slice_data=entry.fp_slice_data,
        )

    def perpendicular_key(
        self,
        *,
        alignment_key: Hashable,
        channel_name: str,
        track_interpolation_ras: NDArray,
        ephys_depths_along_track: NDArray,
        feature_ref: NDArray,
        track_ref: NDArray,
        feature_grid_m: NDArray,
        extent_m: float,
        n_perp_samples: int,
        sigma_samples: float = 2.0,
    ) -> PerpendicularSliceKey:
        """Build a cache key for a perpendicular slice image."""
        return PerpendicularSliceKey(
            alignment_key=alignment_key,
            channel_name=channel_name,
            track_interpolation=array_fingerprint(track_interpolation_ras),
            ephys_depths_along_track=array_fingerprint(ephys_depths_along_track),
            feature_ref=array_fingerprint(feature_ref),
            track_ref=array_fingerprint(track_ref),
            feature_grid=array_fingerprint(feature_grid_m),
            extent_m=float(extent_m),
            n_perp_samples=int(n_perp_samples),
            sigma_samples=float(sigma_samples),
        )

    def get_or_build_perpendicular_slice(
        self,
        *,
        key: PerpendicularSliceKey,
        builder: Callable[[], NDArray],
    ) -> NDArray:
        """Return a cached perpendicular slice image or build/cache it."""
        cached = self._perpendicular_slices.get(key)
        if cached is not None:
            return cached
        image = builder()
        self._perpendicular_slices[key] = image
        return image

    def invalidate_alignment(self, alignment_key: Hashable) -> None:
        """Remove cached slices for one alignment key."""
        self._coronal_slices = {
            key: value
            for key, value in self._coronal_slices.items()
            if key.alignment_key != alignment_key
        }
        self._perpendicular_slices = {
            key: value
            for key, value in self._perpendicular_slices.items()
            if key.alignment_key != alignment_key
        }
        if (
            self._active_coronal_key is not None
            and self._active_coronal_key.alignment_key == alignment_key
        ):
            self._active_coronal_key = None
            self._active_coronal_entry = SliceCacheEntry(None, None)

    def clear(self) -> None:
        """Clear all cached slice data."""
        self._coronal_slices.clear()
        self._perpendicular_slices.clear()
        self._active_coronal_key = None
        self._active_coronal_entry = SliceCacheEntry(None, None)
