"""Query-side application facade for the alignment workspace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.active_shank_screen_queries import ActiveShankScreenQueries
from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.alignment_query_context import AlignmentQueryContext
from ephys_alignment_gui.alignment_read_models import (
    ActiveAlignmentEditScreenState,
    ActiveAlignmentRenderState,
    ActiveReferenceLineRenderState,
    ActiveShankPlotDataState,
    ActiveShankScreenState,
    ActiveSliceDataState,
    ActiveSliceMenuState,
    ActiveSliceRenderState,
    ClusterDetailRenderState,
    FitPlotRenderState,
    HistologyPanelRenderState,
    NearbyBoundaryRenderState,
    PerpendicularSliceRenderState,
    PreparedActiveShankScreenState,
    ProbeExtentRenderState,
    ScaleFactorRenderState,
)
from ephys_alignment_gui.alignment_render_queries import AlignmentRenderQueries
from ephys_alignment_gui.app_results import (
    ShankSelectionState,
)
from ephys_alignment_gui.document import AlignmentDocument
from ephys_alignment_gui.ephys_plot_queries import EphysPlotQueries
from ephys_alignment_gui.ephys_stream_runtime import StreamKey
from ephys_alignment_gui.plot_menu_state import PlotMenuState
from ephys_alignment_gui.plot_registry import PlotMenu, PlotSpec
from ephys_alignment_gui.session_runtime import (
    LoadDataPlan,
    SessionRuntime,
)
from ephys_alignment_gui.slice_data_runtime_service import SliceDataRuntimeService
from ephys_alignment_gui.slice_display_policy import SliceDisplayPolicy, SliceSelection
from ephys_alignment_gui.slice_queries import SliceQueries
from ephys_alignment_gui.workspace_state_queries import WorkspaceStateQueries


@dataclass
class AlignmentQueries:
    """Query/read-model app port for UI rendering state."""

    document: AlignmentDocument
    runtime: SessionRuntime
    data_context: AlignmentDataContext | None = None
    display_state: AlignmentDisplayState = field(default_factory=AlignmentDisplayState)
    derived_data_service: AlignmentDerivedDataService = field(
        default_factory=AlignmentDerivedDataService
    )
    slice_data_runtime_service: SliceDataRuntimeService = field(
        default_factory=SliceDataRuntimeService
    )
    histology_context: Any | None = None
    slice_service: Any | None = None
    region_lookup_service: Any | None = None
    slice_display_policy: SliceDisplayPolicy = field(default_factory=SliceDisplayPolicy)
    query_context: AlignmentQueryContext = field(init=False)
    ephys_plot_queries: EphysPlotQueries = field(init=False)
    alignment_render_queries: AlignmentRenderQueries = field(init=False)
    slice_queries: SliceQueries = field(init=False)
    workspace_state_queries: WorkspaceStateQueries = field(init=False)
    active_shank_screen_queries: ActiveShankScreenQueries = field(init=False)

    def __post_init__(self) -> None:
        self.query_context = AlignmentQueryContext(
            document=self.document,
            runtime=self.runtime,
        )
        self.workspace_state_queries = WorkspaceStateQueries(
            context=self.query_context,
            data_context=self.data_context,
            display_state=self.display_state,
            histology_context=self.histology_context,
            region_lookup_service=self.region_lookup_service,
        )
        self.ephys_plot_queries = EphysPlotQueries(
            context=self.query_context,
            display_state=self.display_state,
            derived_data_service=self.derived_data_service,
            histology_context=self.histology_context,
        )
        self.alignment_render_queries = AlignmentRenderQueries(
            context=self.query_context,
            display_state=self.display_state,
            derived_data_service=self.derived_data_service,
        )
        self.slice_queries = SliceQueries(
            context=self.query_context,
            render_queries=self.alignment_render_queries,
            derived_data_service=self.derived_data_service,
            slice_data_runtime_service=self.slice_data_runtime_service,
            histology_context=self.histology_context,
            slice_service=self.slice_service,
            slice_display_policy=self.slice_display_policy,
        )
        self.active_shank_screen_queries = ActiveShankScreenQueries(
            workspace_state_queries=self.workspace_state_queries,
            ephys_plot_queries=self.ephys_plot_queries,
            slice_queries=self.slice_queries,
        )

    def active_shank_selection(self) -> ShankSelectionState:
        """Return the current document-owned shank selection."""
        return self.workspace_state_queries.active_shank_selection()

    def active_reference_line_state(
        self,
        shank_idx: int | None = None,
    ) -> ActiveReferenceLineRenderState | None:
        """Return pending or previous-alignment reference lines for rendering."""
        return self.workspace_state_queries.active_reference_line_state(shank_idx)

    def is_loaded_stream_shank(
        self,
        stream_key: StreamKey | None,
        shank_idx: int,
    ) -> bool:
        """Return whether the requested stream/shank is already active."""
        return self.workspace_state_queries.is_loaded_stream_shank(
            stream_key,
            shank_idx,
        )

    def plan_load_data(
        self,
        stream_key: StreamKey | None,
        shank_idx: int,
    ) -> LoadDataPlan:
        """Return the stream-cache plan for one load-data request."""
        return self.workspace_state_queries.plan_load_data(stream_key, shank_idx)

    def stream_key_for_selection(
        self,
        recording_id: str,
        probe_name: str,
    ) -> StreamKey | None:
        """Resolve the ephys stream key for a recording/probe selection."""
        return self.workspace_state_queries.stream_key_for_selection(
            recording_id,
            probe_name,
        )

    def histology_data_loaded(self) -> bool:
        """Whether subject-level histology runtime data is already loaded."""
        return self.workspace_state_queries.histology_data_loaded()

    def active_mouse_root_path(self) -> Path | None:
        """Return the active mouse-root path, if one is loaded."""
        return self.workspace_state_queries.active_mouse_root_path()

    def mouse_root_loaded(self) -> bool:
        """Return whether an input mouse-root datapackage is loaded."""
        return self.workspace_state_queries.mouse_root_loaded()

    def active_output_root(self) -> Path | None:
        """Return the active output root, if one has been set."""
        return self.workspace_state_queries.active_output_root()

    def has_output_directory(self) -> bool:
        """Return whether the active probe output directory is available."""
        return self.workspace_state_queries.has_output_directory()

    def active_output_directory(self) -> Path | None:
        """Return the derived active output directory, if available."""
        return self.workspace_state_queries.active_output_directory()

    def active_plot_export_directory(self) -> Path | None:
        """Return the default plot-export directory for the active shank."""
        return self.workspace_state_queries.active_plot_export_directory()

    def depth_view_settings(self) -> Any:
        """Return feature-depth display settings."""
        return self.workspace_state_queries.depth_view_settings()

    def fit_depth_um(self) -> Any:
        """Return the depth grid used for fit-panel rendering."""
        return self.workspace_state_queries.fit_depth_um()

    def linear_fit_enabled(self) -> bool:
        """Return whether fit commands use linear fitting."""
        return self.workspace_state_queries.linear_fit_enabled()

    def active_brain_atlas(self) -> Any | None:
        """Return loaded brain-atlas runtime data for desktop rendering."""
        return self.workspace_state_queries.active_brain_atlas()

    def allen_structure_tree(self) -> Any | None:
        """Return Allen structure metadata for desktop rendering."""
        return self.workspace_state_queries.allen_structure_tree()

    def region_description(self, region_id: int) -> tuple[str, str] | None:
        """Return user-facing region description and lookup label."""
        return self.workspace_state_queries.region_description(region_id)

    def active_alignment_edit_screen_state(
        self,
    ) -> ActiveAlignmentEditScreenState:
        """Return edit-history status and previous reference-line render data."""
        return self.workspace_state_queries.active_alignment_edit_screen_state()

    def active_unit_filter(self) -> str:
        """Return the selected unit subset for active ephys plot data."""
        return self.ephys_plot_queries.active_unit_filter()

    def resolve_shank_preserve_plot_selection(
        self,
        preserve_plot_selection: bool | None,
    ) -> bool:
        """Return whether shank redraw should preserve current plot selections."""
        return self.workspace_state_queries.resolve_shank_preserve_plot_selection(
            preserve_plot_selection
        )

    def prepare_active_shank_plot_data_state(
        self,
        *,
        unit_filter: str | None = None,
    ) -> ActiveShankPlotDataState | None:
        """Materialize active shank PlotData and return frontend-safe bounds."""
        return self.ephys_plot_queries.prepare_active_shank_plot_data_state(
            unit_filter=unit_filter,
        )

    def active_shank_screen_state(
        self,
        *,
        preserve_plot_selection: bool,
        previous_ephys_plot_keys: Mapping[PlotMenu, str | None] | None = None,
        raw_image_payloads: Mapping[Any, Any] | None = None,
        previous_slice_selection: SliceSelection | None = None,
        offline: bool,
    ) -> ActiveShankScreenState:
        """Return the Qt-free screen state for the active shank."""
        return self.active_shank_screen_queries.active_shank_screen_state(
            preserve_plot_selection=preserve_plot_selection,
            previous_ephys_plot_keys=previous_ephys_plot_keys,
            raw_image_payloads=raw_image_payloads,
            previous_slice_selection=previous_slice_selection,
            offline=offline,
        )

    def prepare_active_shank_screen_state(
        self,
        *,
        histology_available: bool,
        preserve_plot_selection: bool,
        previous_ephys_plot_keys: Mapping[PlotMenu, str | None] | None = None,
        raw_image_payloads: Mapping[Any, Any] | None = None,
        previous_slice_selection: SliceSelection | None = None,
        offline: bool,
    ) -> PreparedActiveShankScreenState:
        """Materialize active shank runtime state and return its screen DTO."""
        return self.active_shank_screen_queries.prepare_active_shank_screen_state(
            histology_available=histology_available,
            preserve_plot_selection=preserve_plot_selection,
            previous_ephys_plot_keys=previous_ephys_plot_keys,
            raw_image_payloads=raw_image_payloads,
            previous_slice_selection=previous_slice_selection,
            offline=offline,
        )

    def active_plot_menu_state(
        self,
        previous_selected_keys: Mapping[PlotMenu, str | None] | None = None,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> PlotMenuState:
        """Return available plot menu entries for the active shank."""
        return self.ephys_plot_queries.active_plot_menu_state(
            previous_selected_keys=previous_selected_keys,
            raw_image_payloads=raw_image_payloads,
        )

    def active_plot_spec(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> PlotSpec | None:
        """Return an available plot spec for the active shank."""
        return self.ephys_plot_queries.active_plot_spec(
            spec_key,
            raw_image_payloads=raw_image_payloads,
        )

    def active_plot_payload(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> Any:
        """Resolve a plot payload for the active shank."""
        return self.ephys_plot_queries.active_plot_payload(
            spec_key,
            raw_image_payloads=raw_image_payloads,
        )

    def active_plot_bounds(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> Any:
        """Resolve optional plot bounds for the active shank."""
        return self.ephys_plot_queries.active_plot_bounds(
            spec_key,
            raw_image_payloads=raw_image_payloads,
        )

    def active_in_brain_depths_um(self) -> Any:
        """Return active PlotData in-brain depths, if available."""
        return self.ephys_plot_queries.active_in_brain_depths_um()

    def active_in_brain_depths_for_alignment(self) -> Any:
        """Return active channel depths whose aligned CCF annotation is not root."""
        return self.ephys_plot_queries.active_in_brain_depths_for_alignment()

    def prepare_active_slice_screen_data(self) -> ActiveSliceDataState | None:
        """Materialize active slice data when histology runtime is available."""
        return self.slice_queries.prepare_active_slice_screen_data()

    def active_cluster_detail(
        self,
        cluster_idx: int,
    ) -> ClusterDetailRenderState | None:
        """Return autocorrelogram/template detail for one active cluster."""
        return self.ephys_plot_queries.active_cluster_detail(cluster_idx)

    def active_session_notes(self) -> str:
        """Return notes for the active ephys stream, if any."""
        return self.ephys_plot_queries.active_session_notes()

    def active_histology_region_id(self, region_idx: int) -> int | None:
        """Return an active histology region id by plotted region index."""
        return self.alignment_render_queries.active_histology_region_id(region_idx)

    def active_alignment_render_state(self) -> ActiveAlignmentRenderState | None:
        """Return derived render data for the active alignment, if available."""
        return self.alignment_render_queries.active_alignment_render_state()

    def active_histology_panel_state(
        self,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> HistologyPanelRenderState | None:
        """Return histology-region render data for the active alignment."""
        return self.alignment_render_queries.active_histology_panel_state(
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )

    def probe_extent_render_state(
        self,
        active_alignment: ActiveAlignment,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> ProbeExtentRenderState | None:
        """Return probe-extent render data for an alignment."""
        return self.alignment_render_queries.probe_extent_render_state(
            active_alignment,
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )

    def active_scale_factor_state(
        self,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> ScaleFactorRenderState | None:
        """Return scale-factor render data for the active alignment."""
        return self.alignment_render_queries.active_scale_factor_state(
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )

    def active_nearby_boundary_state(
        self,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
        allen: Any,
        brain_atlas: Any,
        steps: int = 6,
    ) -> NearbyBoundaryRenderState | None:
        """Return nearby-boundary curves for the active alignment track."""
        return self.alignment_render_queries.active_nearby_boundary_state(
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
            allen=allen,
            brain_atlas=brain_atlas,
            steps=steps,
        )

    def active_fit_plot_state(
        self,
        *,
        depth_um: Any,
        lin_fit: bool,
    ) -> FitPlotRenderState | None:
        """Return feature/track fit curve render data for the active alignment."""
        return self.alignment_render_queries.active_fit_plot_state(
            depth_um=depth_um,
            lin_fit=lin_fit,
        )

    def ensure_active_slice_data_state(self) -> ActiveSliceDataState | None:
        """Build/cache and return coronal slice data for the active alignment."""
        return self.slice_queries.ensure_active_slice_data_state()

    def active_slice_data_state(self) -> ActiveSliceDataState | None:
        """Return currently active coronal slice data without building it."""
        return self.slice_queries.active_slice_data_state()

    def active_slice_data_by_attr(self) -> dict[str, Any]:
        """Return active slice data keyed by menu payload data-attr names."""
        return self.slice_queries.active_slice_data_by_attr()

    def active_slice_menu_state(
        self,
        *,
        offline: bool,
        previous_selection: SliceSelection | None = None,
    ) -> ActiveSliceMenuState | None:
        """Return menu and fallback-selection state for active slice data."""
        return self.slice_queries.active_slice_menu_state(
            offline=offline,
            previous_selection=previous_selection,
        )

    def active_slice_render_state(
        self,
        selection: SliceSelection,
    ) -> ActiveSliceRenderState | None:
        """Return a render payload for one active coronal slice selection."""
        return self.slice_queries.active_slice_render_state(selection)

    def active_perpendicular_slice_state(
        self,
        channel_name: str,
        *,
        extent_m: float = 500e-6,
        probe_margin_um: float = 100.0,
    ) -> PerpendicularSliceRenderState | None:
        """Build/cache and return a perpendicular slice render payload."""
        return self.slice_queries.active_perpendicular_slice_state(
            channel_name=channel_name,
            extent_m=extent_m,
            probe_margin_um=probe_margin_um,
        )

    def _active_shank_idx(self) -> int:
        return self.query_context.active_shank_idx()
