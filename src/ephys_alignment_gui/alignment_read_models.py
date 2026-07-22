"""Qt-free read models for alignment rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentHistologyData,
    ChannelProjectionData,
)
from ephys_alignment_gui.document import AlignmentKey


@dataclass(frozen=True)
class ActiveAlignmentRenderState:
    """Derived data needed to render the active alignment."""

    key: AlignmentKey
    active_alignment: ActiveAlignment
    histology: AlignmentHistologyData
    projection: ChannelProjectionData


@dataclass(frozen=True)
class ActiveSliceDataState:
    """Runtime slice data available for the active alignment."""

    key: AlignmentKey
    slice_data: Any
    fp_slice_data: Any

    @property
    def data_by_attr(self) -> dict[str, Any]:
        """Return legacy slice data keyed by menu payload data-attr names."""
        return {
            "slice_data": self.slice_data,
            "fp_slice_data": self.fp_slice_data,
        }


@dataclass(frozen=True)
class PerpendicularSliceRenderState:
    """Perpendicular slice image and geometry ready for frontend rendering."""

    key: AlignmentKey
    channel_name: str
    image: NDArray[Any]
    extent_um: float
    feature_min_um: float
    feature_max_um: float
    n_perp_samples: int
    n_depths: int
    channel_depths_um: NDArray[Any]

    @property
    def scale_x_um(self) -> float:
        """Micrometres per perpendicular image sample."""
        if self.n_perp_samples <= 1:
            return 1.0
        return (2 * self.extent_um) / (self.n_perp_samples - 1)

    @property
    def scale_y_um(self) -> float:
        """Micrometres per depth image sample."""
        if self.n_depths <= 1:
            return 1.0
        return (self.feature_max_um - self.feature_min_um) / (self.n_depths - 1)
