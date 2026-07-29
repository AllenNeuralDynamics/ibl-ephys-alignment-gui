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
from ephys_alignment_gui.plot_menu_state import PlotMenuState
from ephys_alignment_gui.slice_display_policy import (
    SliceMenuItem,
    SliceRenderDecision,
    SliceSelection,
    SliceSelectionDecision,
)


@dataclass(frozen=True)
class ActiveAlignmentRenderState:
    """Derived data needed to render the active alignment."""

    key: AlignmentKey
    active_alignment: ActiveAlignment
    histology: AlignmentHistologyData
    projection: ChannelProjectionData


@dataclass(frozen=True)
class ProbeExtentRenderState:
    """Probe extent and interaction bounds in feature-depth units."""

    probe_tip_um: float
    probe_top_um: float
    probe_extra_um: float
    feature_min_um: float
    feature_max_um: float
    tip_bounds_um: tuple[float, float]
    top_bounds_um: tuple[float, float]


@dataclass(frozen=True)
class HistologyPanelRenderState:
    """Histology region data ready for frontend panel rendering."""

    key: AlignmentKey
    histology: AlignmentHistologyData
    probe_extent: ProbeExtentRenderState


@dataclass(frozen=True)
class ScaleFactorRenderState:
    """Scale-factor regions ready for frontend rendering."""

    key: AlignmentKey
    region: Any
    scale: Any
    probe_extent: ProbeExtentRenderState


@dataclass(frozen=True)
class FitPlotRenderState:
    """Feature/track fit curve data ready for frontend rendering."""

    key: AlignmentKey
    feature_um: NDArray[Any]
    track_um: NDArray[Any]
    linear_feature_um: NDArray[Any] | None = None
    linear_track_um: NDArray[Any] | None = None


@dataclass(frozen=True)
class ClusterDetailRenderState:
    """Cluster autocorrelogram and template waveform data for frontend rendering."""

    cluster_no: Any
    autocorr: NDArray[Any]
    t_autocorr: NDArray[Any]
    template_waveform: NDArray[Any]
    t_template: NDArray[Any]


@dataclass(frozen=True)
class NearbyBoundaryRenderState:
    """Nearby boundary curves ready for frontend rendering."""

    key: AlignmentKey | None
    x: Any
    y: Any
    colours: Any
    parent_x: Any
    parent_y: Any
    parent_colours: Any
    probe_extent: ProbeExtentRenderState


@dataclass(frozen=True)
class ActiveReferenceLineRenderState:
    """Reference-line coordinates ready for frontend overlay rendering."""

    feature_positions_um: Any
    track_positions_um: Any | None = None


@dataclass(frozen=True)
class ActiveShankPlotDataState:
    """Prepared ephys plot-data bounds for the active shank."""

    key: AlignmentKey | None
    shank_idx: int
    unit_filter: str
    channel_min_um: float
    channel_max_um: float
    in_brain_depths_um: Any


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
class ActiveSliceMenuState:
    """Available coronal slice menu items and selected fallback policy."""

    key: AlignmentKey
    items: tuple[SliceMenuItem, ...]
    default_selection: SliceSelection
    selection: SliceSelectionDecision


@dataclass(frozen=True)
class ActiveSliceRenderState:
    """Coronal slice image and geometry ready for frontend rendering."""

    key: AlignmentKey
    selection: SliceSelection
    image: Any
    scale: NDArray[Any]
    offset: NDArray[Any]
    decision: SliceRenderDecision
    track_annos_and_ends_ras: NDArray[Any]
    projection: ChannelProjectionData

    @property
    def scalar_channel(self) -> str | None:
        """Scalar volume channel, when this selection can drive perpendicular view."""
        return self.decision.scalar_channel


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


@dataclass(frozen=True)
class ActiveShankScreenState:
    """Qt-free state needed to render the active shank screen."""

    shank_idx: int
    shank_id: int
    alignment_key: AlignmentKey | None
    data_loaded: bool
    preserve_plot_selection: bool
    unit_filter: str
    plot_menu: PlotMenuState
    slice_menu: ActiveSliceMenuState | None
