"""Active alignment value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ActiveAlignment:
    """Current feature/track control points for one shank alignment."""

    feature: NDArray[np.floating[Any]]
    track: NDArray[np.floating[Any]]
    lin_fit: bool = False

    def __post_init__(self) -> None:
        feature = np.array(self.feature, dtype=float, copy=True)
        track = np.array(self.track, dtype=float, copy=True)
        if feature.ndim != 1 or track.ndim != 1:
            raise ValueError("feature and track must be 1D control-point arrays")
        if feature.shape != track.shape:
            raise ValueError("feature and track must have matching shapes")
        feature.setflags(write=False)
        track.setflags(write=False)
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "track", track)

    @classmethod
    def from_values(
        cls,
        feature: Any,
        track: Any,
        *,
        lin_fit: bool = False,
    ) -> ActiveAlignment | None:
        """Create an alignment from legacy buffers, returning None for blanks."""
        if feature is None or track is None:
            return None
        feature_arr = np.asarray(feature)
        track_arr = np.asarray(track)
        if feature_arr.ndim == 0 or track_arr.ndim == 0:
            return None
        return cls(feature_arr, track_arr, lin_fit=lin_fit)

    def feature_copy(self) -> NDArray[np.floating[Any]]:
        """Return a mutable copy of feature control points."""
        return np.array(self.feature, copy=True)

    def track_copy(self) -> NDArray[np.floating[Any]]:
        """Return a mutable copy of track control points."""
        return np.array(self.track, copy=True)
