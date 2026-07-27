"""UI-facing application port for the alignment workspace."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentDerivedDataService,
    AlignmentHistologyData,
)
from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.alignment_events import (
    AlignmentEdited,
    AlignmentEditKind,
    ShankChanged,
)
from ephys_alignment_gui.alignment_read_models import (
    ActiveAlignmentRenderState,
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
    ProbeExtentRenderState,
    ScaleFactorRenderState,
)
from ephys_alignment_gui.controller import (
    AlignmentChoicesUpdated,
    AlignmentController,
    AlignmentEditApplied,
    AlignmentEditNoop,
    Failed,
    NoPreviousAlignments,
    PreviousAlignmentSelected,
    PreviousAlignmentsLoaded,
    ShankAlignmentRuntimeInitialized,
    ShankSelected,
)
from ephys_alignment_gui.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.ephys_stream_runtime import StreamKey
from ephys_alignment_gui.event_bus import EventBus
from ephys_alignment_gui.histology_data_workflow import (
    HistologyDataWorkflow,
    HistologyLoadResult,
)
from ephys_alignment_gui.plot_data_factory import PlotDataFactory
from ephys_alignment_gui.plot_menu_state import PlotMenuState, build_plot_menu_state
from ephys_alignment_gui.plot_registry import (
    PlotMenu,
    PlotSpec,
    resolve_plot_bounds,
    resolve_plot_payload,
)
from ephys_alignment_gui.probe_data_workflow import ProbeDataWorkflow
from ephys_alignment_gui.session_runtime import (
    LoadDataPlan,
    LoadDataTarget,
    SessionRuntime,
)
from ephys_alignment_gui.shank_runtime import ShankRuntime
from ephys_alignment_gui.slice_display_policy import SliceDisplayPolicy, SliceSelection
from ephys_alignment_gui.slice_runtime import SliceCacheEntry
from ephys_alignment_gui.workflow import Ok

logger = logging.getLogger(__name__)


class _ReferenceLinesNotProvided:
    pass


_REFERENCE_LINES_NOT_PROVIDED = _ReferenceLinesNotProvided()
ReferenceLineCapture = tuple[Any, Any] | None | _ReferenceLinesNotProvided


@dataclass(frozen=True)
class ShankSelectionState:
    """Read model for the active shank selection."""

    shank_idx: int
    shank_id: int
    alignment_key: AlignmentKey | None
    data_loaded: bool


@dataclass(frozen=True)
class FreshEphysDataLoaded:
    """Fresh ephys stream data was loaded and cached."""

    stream_runtime: Any
    shank_idx: int


@dataclass
class AlignmentCommands:
    """Command-side app port.

    Methods should be added here as UI call sites migrate. The controller
    remains the command implementation; this object is the UI boundary.
    """

    _controller: AlignmentController
    _events: EventBus
    _display_state: AlignmentDisplayState
    _runtime: SessionRuntime
    _probe_data_workflow: ProbeDataWorkflow
    _histology_data_workflow: HistologyDataWorkflow
    _plot_data_factory: PlotDataFactory

    def select_shank(
        self,
        shank_idx: int,
        *,
        outgoing_reference_lines: ReferenceLineCapture = _REFERENCE_LINES_NOT_PROVIDED,
        source: str = "command",
        preserve_plot_selection: bool | None = None,
    ) -> ShankSelected | Failed:
        """Select a shank as a complete app-level transaction."""
        if (
            self._controller.document.data_loaded
            and outgoing_reference_lines is not _REFERENCE_LINES_NOT_PROVIDED
        ):
            capture_result = self._capture_outgoing_reference_lines(
                outgoing_reference_lines
            )
            if isinstance(capture_result, Failed):
                return capture_result

        result = self._controller.select_shank(shank_idx)
        if isinstance(result, ShankSelected):
            self._events.emit(
                ShankChanged(
                    source=source,
                    previous_shank_idx=result.previous_shank_idx,
                    shank_idx=result.shank_idx,
                    previous_key=result.previous_key,
                    active_key=result.selected_key,
                    data_loaded=result.data_loaded,
                    preserve_plot_selection=preserve_plot_selection,
                )
            )
        return result

    def _capture_outgoing_reference_lines(
        self,
        outgoing_reference_lines: ReferenceLineCapture,
    ) -> Any:
        outgoing_shank_idx = self._controller.document.selected_shank
        if outgoing_reference_lines is None:
            return self._controller.clear_pending_reference_lines(outgoing_shank_idx)

        if outgoing_reference_lines is _REFERENCE_LINES_NOT_PROVIDED:
            return None

        feature_positions_um, track_positions_um = outgoing_reference_lines
        return self._controller.set_pending_reference_lines(
            feature_positions_um=feature_positions_um,
            track_positions_um=track_positions_um,
            shank_idx=outgoing_shank_idx,
        )

    def load_previous_alignments(
        self,
        *,
        folder: Path | None,
        use_docdb: bool,
        shank_idx: int | None = None,
    ) -> AlignmentChoicesUpdated | NoPreviousAlignments | Failed:
        """Load and store previous alignments for a document-selected shank."""
        target_shank = self._active_or_given_shank(shank_idx)
        loaded = self._controller.load_previous_alignments(
            folder=folder,
            shank_idx=target_shank,
            use_docdb=use_docdb,
        )
        if isinstance(loaded, Failed | NoPreviousAlignments):
            return loaded
        assert isinstance(loaded, PreviousAlignmentsLoaded)
        return self._controller.set_previous_alignments(
            loaded.alignments,
            shank_idx=target_shank,
        )

    def select_previous_alignment(
        self,
        idx: int,
        *,
        shank_idx: int | None = None,
    ) -> PreviousAlignmentSelected | Failed:
        """Select a previous/original alignment on a document-selected shank."""
        return self._controller.select_previous_alignment(
            idx,
            shank_idx=self._active_or_given_shank(shank_idx),
        )

    def initialize_shank_alignment_runtime(
        self,
        shank_runtime: Any,
        *,
        track_annotations_ras: Any,
        brain_atlas: Any,
    ) -> ShankAlignmentRuntimeInitialized | Failed:
        """Initialize runtime alignment data for the active shank."""
        return self._controller.initialize_shank_alignment_runtime(
            shank_runtime,
            track_annotations_ras=track_annotations_ras,
            brain_atlas=brain_atlas,
        )

    def _active_or_given_shank(self, shank_idx: int | None) -> int:
        if shank_idx is not None:
            return shank_idx
        return self._controller.document.selected_shank

    def can_load_previous_alignments(self) -> Ok | Failed:
        """Return whether previous alignments can be loaded."""
        return self._controller.can_load_previous_alignments()

    def load_fresh_ephys_data(
        self,
        shank_idx: int,
    ) -> FreshEphysDataLoaded | Failed:
        """Load selected-probe ephys data, cache runtime, and mark data loaded."""
        try:
            loaded = self._probe_data_workflow.load(shank_idx)
            if not loaded.stream.ephys_dir:
                return Failed("Failed to load ephys data")
            stream_runtime = self._runtime.cache_loaded_stream_data(
                loaded.stream,
                self._plot_data_factory,
                shank_idx=shank_idx,
            )
        except Exception as exc:
            return Failed(f"Failed to load ephys data: {exc}")

        self._controller.finish_load_data(shank_idx)
        return FreshEphysDataLoaded(
            stream_runtime=stream_runtime,
            shank_idx=shank_idx,
        )

    def load_histology_data(self) -> HistologyLoadResult:
        """Load subject-level histology runtime data if it is available."""
        return self._histology_data_workflow.load_if_needed()

    def set_unit_filter(self, unit_filter: str) -> Ok:
        """Select the unit subset used when preparing ephys plot data."""
        self._display_state.set_unit_filter(unit_filter)
        stream_runtime = self._runtime.active_stream_runtime
        if stream_runtime is not None:
            stream_runtime.filtered_plot_data_for_shank(
                self._active_or_given_shank(None),
                unit_filter=unit_filter,
            )
        return Ok()

    def offset_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        probe_tip_um: float,
        lin_fit: bool,
        track_shift_m: float = 0.0,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply an offset edit on a document-selected shank."""
        result = self._controller.offset_alignment_from_tip(
            tip_position_um=tip_position_um,
            probe_tip_um=probe_tip_um,
            lin_fit=lin_fit,
            track_shift_m=track_shift_m,
            shank_idx=self._active_or_given_shank(shank_idx),
        )
        self._emit_alignment_edited("offset", result)
        return result

    def fit_alignment_to_reference_lines(
        self,
        shank_runtime: Any,
        *,
        line_features_um: Any,
        line_tracks_um: Any,
        lin_fit: bool,
        extend_feature: int,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a reference-line fit for a document-selected shank runtime."""
        result = self._controller.fit_alignment_to_reference_lines(
            shank_runtime,
            line_features_um=line_features_um,
            line_tracks_um=line_tracks_um,
            lin_fit=lin_fit,
            extend_feature=extend_feature,
        )
        self._emit_alignment_edited("fit", result)
        return result

    def go_next_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor forward."""
        result = self._controller.go_next_alignment(
            self._active_or_given_shank(shank_idx)
        )
        self._emit_alignment_edited("next", result)
        return result

    def go_previous_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor backward."""
        result = self._controller.go_previous_alignment(
            self._active_or_given_shank(shank_idx)
        )
        self._emit_alignment_edited("previous", result)
        return result

    def reset_alignment_to_initial(
        self,
        shank_runtime: Any,
        *,
        lin_fit: bool,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Reset active alignment state to the loaded runtime's initial geometry."""
        result = self._controller.reset_alignment_to_initial(
            shank_runtime,
            lin_fit=lin_fit,
        )
        if isinstance(result, AlignmentEditApplied):
            clear_result = self._controller.clear_pending_reference_lines(
                shank_runtime.shank_idx
            )
            if isinstance(clear_result, Failed):
                logger.error(clear_result.message)
            self._emit_alignment_edited("reset", result)
        return result

    def _emit_alignment_edited(
        self,
        edit_kind: AlignmentEditKind,
        result: AlignmentEditApplied | AlignmentEditNoop | Failed,
    ) -> None:
        if not isinstance(result, AlignmentEditApplied):
            return
        active_key = self._controller.document.selected_alignment_key
        if active_key is None:
            logger.error("Cannot emit alignment edit event without an active key")
            return
        self._events.emit(
            AlignmentEdited(
                edit_kind=edit_kind,
                active_key=active_key,
                active_alignment=result.alignment,
                lin_fit=result.lin_fit,
            )
        )


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
    histology_context: Any | None = None
    slice_service: Any | None = None
    slice_display_policy: SliceDisplayPolicy = field(default_factory=SliceDisplayPolicy)

    def active_shank_selection(self) -> ShankSelectionState:
        """Return the current document-owned shank selection."""
        shank_idx = self._active_shank_idx()
        return ShankSelectionState(
            shank_idx=shank_idx,
            shank_id=shank_idx + 1,
            alignment_key=self.document.selected_alignment_key,
            data_loaded=self.document.data_loaded,
        )

    def is_loaded_stream_shank(
        self,
        stream_key: StreamKey | None,
        shank_idx: int,
    ) -> bool:
        """Return whether the requested stream/shank is already active."""
        if stream_key is None or not self.document.data_loaded:
            return False
        return (
            self.runtime.is_active_stream_shank(stream_key, shank_idx)
            and self._active_shank_idx() == shank_idx
        )

    def plan_load_data(
        self,
        stream_key: StreamKey | None,
        shank_idx: int,
    ) -> LoadDataPlan:
        """Return the stream-cache plan for one load-data request."""
        return self.runtime.plan_load_data(
            LoadDataTarget(stream_key=stream_key, shank_idx=shank_idx),
            data_loaded=self.document.data_loaded,
        )

    def stream_key_for_selection(
        self,
        recording_id: str,
        probe_name: str,
    ) -> StreamKey | None:
        """Resolve the ephys stream key for a recording/probe selection."""
        if self.data_context is None:
            return None
        try:
            return self.data_context.stream_key_for_selection(recording_id, probe_name)
        except Exception:
            logger.warning(
                "Could not resolve stream key for %s/%s",
                recording_id,
                probe_name,
                exc_info=True,
            )
            return None

    def histology_data_loaded(self) -> bool:
        """Whether subject-level histology runtime data is already loaded."""
        return (
            self.histology_context is not None
            and self.histology_context.brain_atlas is not None
        )

    def active_unit_filter(self) -> str:
        """Return the selected unit subset for active ephys plot data."""
        return self.display_state.unit_filter

    def resolve_shank_preserve_plot_selection(
        self,
        preserve_plot_selection: bool | None,
    ) -> bool:
        """Return whether shank redraw should preserve current plot selections."""
        if preserve_plot_selection is None:
            return self.document.data_loaded
        return preserve_plot_selection

    def prepare_active_shank_plot_data_state(
        self,
        *,
        unit_filter: str | None = None,
    ) -> ActiveShankPlotDataState | None:
        """Materialize active shank PlotData and return frontend-safe bounds."""
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return None
        shank_idx = self._active_shank_idx()
        unit_filter = self.active_unit_filter() if unit_filter is None else unit_filter
        plotdata = stream_runtime.filtered_plot_data_for_shank(
            shank_idx,
            unit_filter=unit_filter,
        )
        in_brain_depths_um = self.active_in_brain_depths_for_alignment()
        plotdata.in_brain_depths_um = in_brain_depths_um
        return ActiveShankPlotDataState(
            key=self.document.selected_alignment_key,
            shank_idx=shank_idx,
            unit_filter=unit_filter,
            channel_min_um=float(getattr(plotdata, "chn_min", 0.0)),
            channel_max_um=float(getattr(plotdata, "chn_max", 0.0)),
            in_brain_depths_um=in_brain_depths_um,
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
        selection = self.active_shank_selection()
        return ActiveShankScreenState(
            shank_idx=selection.shank_idx,
            shank_id=selection.shank_id,
            alignment_key=selection.alignment_key,
            data_loaded=selection.data_loaded,
            preserve_plot_selection=preserve_plot_selection,
            unit_filter=self.active_unit_filter(),
            plot_menu=self.active_plot_menu_state(
                previous_selected_keys=(
                    previous_ephys_plot_keys if preserve_plot_selection else None
                ),
                raw_image_payloads=raw_image_payloads,
            ),
            slice_menu=self.active_slice_menu_state(
                offline=offline,
                previous_selection=(
                    previous_slice_selection if preserve_plot_selection else None
                ),
            ),
        )

    def active_plot_menu_state(
        self,
        previous_selected_keys: Mapping[PlotMenu, str | None] | None = None,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> PlotMenuState:
        """Return available plot menu entries for the active shank."""
        plotdata = self._active_plotdata()
        return self._plot_menu_state_for_plotdata(
            plotdata,
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
        plotdata = self._active_plotdata()
        state = self._plot_menu_state_for_plotdata(
            plotdata,
            raw_image_payloads=raw_image_payloads,
        )
        return self._find_plot_spec(state, spec_key)

    def active_plot_payload(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> Any:
        """Resolve a plot payload for the active shank."""
        plotdata = self._active_plotdata()
        state = self._plot_menu_state_for_plotdata(
            plotdata,
            raw_image_payloads=raw_image_payloads,
        )
        spec = self._find_plot_spec(state, spec_key)
        if spec is None:
            return None
        return resolve_plot_payload(plotdata, spec)

    def active_plot_bounds(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> Any:
        """Resolve optional plot bounds for the active shank."""
        plotdata = self._active_plotdata()
        state = self._plot_menu_state_for_plotdata(
            plotdata,
            raw_image_payloads=raw_image_payloads,
        )
        spec = self._find_plot_spec(state, spec_key)
        if spec is None:
            return None
        return resolve_plot_bounds(plotdata, spec)

    def active_in_brain_depths_um(self) -> Any:
        """Return active PlotData in-brain depths, if available."""
        plotdata = self._active_plotdata()
        if plotdata is None:
            return None
        return getattr(plotdata, "in_brain_depths_um", None)

    def active_in_brain_depths_for_alignment(self) -> Any:
        """Return active channel depths whose aligned CCF annotation is not root."""
        context = self._active_alignment_context()
        if (
            context is None
            or self.histology_context is None
            or self.histology_context.brain_atlas is None
        ):
            return None
        _key, active_alignment, shank_runtime = context
        try:
            channel_locations_ras = (
                self.derived_data_service.compute_channel_locations(
                    ephysalign=shank_runtime.ephysalign,
                    feature=active_alignment.feature,
                    track=active_alignment.track,
                )
            )
            region_ids = self.histology_context.brain_atlas.get_labels(
                channel_locations_ras
            )
        except Exception:
            logger.warning(
                "Could not determine in-brain channels for probe cmap",
                exc_info=True,
            )
            return None
        in_brain = np.asarray(region_ids) != 0
        if not in_brain.any():
            return None
        return np.asarray(shank_runtime.chn_depths)[in_brain]

    def prepare_active_slice_screen_data(self) -> ActiveSliceDataState | None:
        """Materialize active slice data when histology runtime is available."""
        if (
            self.histology_context is None
            or self.histology_context.brain_atlas is None
        ):
            shank_runtime = self._active_shank_runtime()
            if shank_runtime is not None:
                shank_runtime.slice_data = {}
                shank_runtime.fp_slice_data = None
            return None
        return self.ensure_active_slice_data_state()

    def active_cluster_detail(
        self,
        cluster_idx: int,
    ) -> ClusterDetailRenderState | None:
        """Return autocorrelogram/template detail for one active cluster."""
        plotdata = self._active_plotdata()
        if plotdata is None:
            return None
        autocorr, cluster_no = plotdata.get_autocorr(cluster_idx)
        template_waveform = plotdata.get_template_wf(cluster_idx)
        return ClusterDetailRenderState(
            cluster_no=cluster_no,
            autocorr=np.asarray(autocorr),
            t_autocorr=np.asarray(plotdata.t_autocorr),
            template_waveform=np.asarray(template_waveform),
            t_template=np.asarray(plotdata.t_template),
        )

    def active_alignment_render_state(self) -> ActiveAlignmentRenderState | None:
        """Return derived render data for the active alignment, if available."""
        context = self._active_alignment_context()
        if context is None:
            return None
        key, active_alignment, shank_runtime = context
        return ActiveAlignmentRenderState(
            key=key,
            active_alignment=active_alignment,
            histology=self._compute_active_histology(active_alignment, shank_runtime),
            projection=self.derived_data_service.compute_channel_projection(
                ephysalign=shank_runtime.ephysalign,
                feature=active_alignment.feature,
                track=active_alignment.track,
            ),
        )

    def active_histology_panel_state(
        self,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> HistologyPanelRenderState | None:
        """Return histology-region render data for the active alignment."""
        context = self._active_alignment_context()
        if context is None:
            return None
        key, active_alignment, shank_runtime = context
        probe_extent = self._probe_extent_render_state(
            active_alignment,
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )
        if probe_extent is None:
            return None
        return HistologyPanelRenderState(
            key=key,
            histology=self._compute_active_histology(
                active_alignment,
                shank_runtime,
            ),
            probe_extent=probe_extent,
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
        return self._probe_extent_render_state(
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
        histology_state = self.active_histology_panel_state(
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )
        if histology_state is None:
            return None
        return ScaleFactorRenderState(
            key=histology_state.key,
            region=histology_state.histology.scale.region,
            scale=histology_state.histology.scale.scale,
            probe_extent=histology_state.probe_extent,
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
        context = self._active_alignment_context()
        if context is None:
            return None
        key, active_alignment, shank_runtime = context
        probe_extent = self._probe_extent_render_state(
            active_alignment,
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )
        if probe_extent is None:
            return None
        nearby_boundaries = shank_runtime.nearby_boundaries
        if nearby_boundaries is None:
            nearby_boundaries = self.derived_data_service.compute_nearby_boundaries(
                ephysalign=shank_runtime.ephysalign,
                allen=allen,
                brain_atlas=brain_atlas,
                steps=steps,
            )
            shank_runtime.nearby_boundaries = nearby_boundaries
        return NearbyBoundaryRenderState(
            key=key,
            x=nearby_boundaries.x,
            y=nearby_boundaries.y,
            colours=nearby_boundaries.colours,
            parent_x=nearby_boundaries.parent_x,
            parent_y=nearby_boundaries.parent_y,
            parent_colours=nearby_boundaries.parent_colours,
            probe_extent=probe_extent,
        )

    def active_fit_plot_state(
        self,
        *,
        depth_um: Any,
        lin_fit: bool,
    ) -> FitPlotRenderState | None:
        """Return feature/track fit curve render data for the active alignment."""
        context = self._active_alignment_context()
        if context is None:
            return None
        key, active_alignment, shank_runtime = context
        feature = np.asarray(active_alignment.feature, dtype=float)
        track = np.asarray(active_alignment.track, dtype=float)
        feature_um = feature * 1e6
        track_um = track * 1e6
        linear_feature_um = None
        linear_track_um = None
        depth_um = np.asarray(depth_um, dtype=float)
        if lin_fit and feature.size >= 5 and depth_um.size > 0:
            depth_lin = shank_runtime.ephysalign.feature2track_lin(
                depth_um / 1e6,
                feature,
                track,
            )
            if np.any(depth_lin):
                linear_feature_um = depth_um
                linear_track_um = np.asarray(depth_lin, dtype=float) * 1e6
        return FitPlotRenderState(
            key=key,
            feature_um=feature_um,
            track_um=track_um,
            linear_feature_um=linear_feature_um,
            linear_track_um=linear_track_um,
        )

    def ensure_active_slice_data_state(self) -> ActiveSliceDataState | None:
        """Build/cache and return coronal slice data for the active alignment."""
        context = self._active_alignment_context()
        if context is None:
            return None
        key, _active_alignment, shank_runtime = context
        if self.histology_context is None or self.slice_service is None:
            return None
        brain_atlas = self.histology_context.brain_atlas
        if brain_atlas is None:
            return None
        track = shank_runtime.ephysalign.track_interpolation_ras

        def build_slice() -> SliceCacheEntry:
            return SliceCacheEntry(
                slice_data=self.slice_service.build_slice_set(
                    brain_atlas=brain_atlas,
                    histology_images=self.histology_context.histology_images,
                    lazy_channel_paths=self.histology_context.lazy_channel_paths,
                    track_interpolation_ras=track,
                ),
                fp_slice_data=None,
            )

        entry = shank_runtime.slice_runtime.get_or_build_coronal_slice(
            alignment_key=key,
            track_interpolation_ras=track,
            builder=build_slice,
        )
        return ActiveSliceDataState(
            key=key,
            slice_data=entry.slice_data,
            fp_slice_data=entry.fp_slice_data,
        )

    def active_slice_data_state(self) -> ActiveSliceDataState | None:
        """Return currently active coronal slice data without building it."""
        context = self._active_alignment_context()
        if context is None:
            return None
        key, _active_alignment, shank_runtime = context
        entry = shank_runtime.slice_runtime.cached_coronal_slice(
            alignment_key=key,
            track_interpolation_ras=shank_runtime.ephysalign.track_interpolation_ras,
        )
        if entry is None:
            return None
        return ActiveSliceDataState(
            key=key,
            slice_data=entry.slice_data,
            fp_slice_data=entry.fp_slice_data,
        )

    def active_slice_data_by_attr(self) -> dict[str, Any]:
        """Return active slice data keyed by menu payload data-attr names."""
        state = self.active_slice_data_state()
        if state is None:
            return {"slice_data": None, "fp_slice_data": None}
        return state.data_by_attr

    def active_slice_menu_state(
        self,
        *,
        offline: bool,
        previous_selection: SliceSelection | None = None,
    ) -> ActiveSliceMenuState | None:
        """Return menu and fallback-selection state for active slice data."""
        state = self.active_slice_data_state()
        if state is None:
            return None
        slice_data = state.slice_data or {}
        if not isinstance(slice_data, Mapping):
            return None
        fp_slice_data = (
            state.fp_slice_data if isinstance(state.fp_slice_data, Mapping) else None
        )
        items = self.slice_display_policy.menu_items(
            slice_data=slice_data,
            fp_slice_data=fp_slice_data,
            offline=offline,
        )
        default_selection = self.slice_display_policy.default_selection(slice_data)
        selection = self.slice_display_policy.choose_selection(
            previous=previous_selection,
            default=default_selection,
            data_by_attr=state.data_by_attr,
        )
        return ActiveSliceMenuState(
            key=state.key,
            items=tuple(items),
            default_selection=default_selection,
            selection=selection,
        )

    def active_slice_render_state(
        self,
        selection: SliceSelection,
    ) -> ActiveSliceRenderState | None:
        """Return a render payload for one active coronal slice selection."""
        slice_state = self.active_slice_data_state()
        context = self._active_alignment_context()
        if slice_state is None or context is None:
            return None
        _key, active_alignment, shank_runtime = context
        data = slice_state.data_by_attr.get(selection.data_attr)
        if not isinstance(data, Mapping) or selection.key not in data:
            return None
        image = data[selection.key]
        decision = self.slice_display_policy.render_decision(data, selection.key)
        base_slice_data = slice_state.slice_data
        if not isinstance(base_slice_data, Mapping):
            base_slice_data = {}
        scale = np.asarray(data.get("scale", base_slice_data.get("scale")))
        offset = np.asarray(data.get("offset", base_slice_data.get("offset")))
        if scale.size < 2 or offset.size < 2:
            logger.warning(
                "Cannot render slice %s: missing scale/offset metadata",
                selection,
            )
            return None
        track_annos_and_ends_ras = shank_runtime.track_annos_and_ends_ras
        if track_annos_and_ends_ras is None:
            return None
        projection = self.derived_data_service.compute_channel_projection(
            ephysalign=shank_runtime.ephysalign,
            feature=active_alignment.feature,
            track=active_alignment.track,
        )
        return ActiveSliceRenderState(
            key=slice_state.key,
            selection=selection,
            image=image,
            scale=scale,
            offset=offset,
            decision=decision,
            track_annos_and_ends_ras=track_annos_and_ends_ras,
            projection=projection,
        )

    def active_perpendicular_slice_state(
        self,
        channel_name: str,
        *,
        extent_m: float = 500e-6,
        probe_margin_um: float = 100.0,
    ) -> PerpendicularSliceRenderState | None:
        """Build/cache and return a perpendicular slice render payload."""
        context = self._active_alignment_context()
        if context is None:
            return None
        key, active_alignment, shank_runtime = context
        if self.histology_context is None or self.slice_service is None:
            return None
        brain_atlas = self.histology_context.brain_atlas
        if brain_atlas is None:
            return None

        histology = self._compute_active_histology(active_alignment, shank_runtime)
        grid = self._perpendicular_feature_grid_um(
            shank_runtime=shank_runtime,
            histology=histology,
            brain_atlas=brain_atlas,
            extent_m=extent_m,
            probe_margin_um=probe_margin_um,
        )
        if grid is None:
            return None
        feature_grid_um, feature_grid_m, n_perp_samples = grid

        cache_key = shank_runtime.slice_runtime.perpendicular_key(
            alignment_key=key,
            channel_name=channel_name,
            track_interpolation_ras=shank_runtime.ephysalign.track_interpolation_ras,
            ephys_depths_along_track=(
                shank_runtime.ephysalign.ephys_depths_along_track
            ),
            feature_ref=active_alignment.feature,
            track_ref=active_alignment.track,
            feature_grid_m=feature_grid_m,
            extent_m=extent_m,
            n_perp_samples=n_perp_samples,
        )

        def build_perpendicular_image() -> Any:
            return self.slice_service.build_perpendicular_slice_image(
                brain_atlas=brain_atlas,
                histology_images=self.histology_context.histology_images,
                lazy_channel_paths=self.histology_context.lazy_channel_paths,
                ephysalign=shank_runtime.ephysalign,
                feature_ref=active_alignment.feature,
                track_ref=active_alignment.track,
                feature_grid_m=feature_grid_m,
                channel_name=channel_name,
                extent_m=extent_m,
                n_perp_samples=n_perp_samples,
            )

        try:
            image = shank_runtime.slice_runtime.get_or_build_perpendicular_slice(
                key=cache_key,
                builder=build_perpendicular_image,
            )
        except Exception:
            logger.warning(
                "Could not build perpendicular slice for channel '%s'",
                channel_name,
                exc_info=True,
            )
            return None

        return PerpendicularSliceRenderState(
            key=key,
            channel_name=channel_name,
            image=image,
            extent_um=float(extent_m) * 1e6,
            feature_min_um=float(feature_grid_um[0]),
            feature_max_um=float(feature_grid_um[-1]),
            n_perp_samples=n_perp_samples,
            n_depths=len(feature_grid_um),
            channel_depths_um=np.asarray(shank_runtime.chn_depths, dtype=float),
        )

    def _plot_menu_state_for_plotdata(
        self,
        plotdata: Any,
        *,
        previous_selected_keys: Mapping[PlotMenu, str | None] | None = None,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> PlotMenuState:
        return build_plot_menu_state(
            plotdata,
            previous_selected_keys=previous_selected_keys,
            raw_image_payloads=raw_image_payloads,
        )

    def _find_plot_spec(
        self,
        state: PlotMenuState,
        spec_key: str,
    ) -> PlotSpec | None:
        for spec in state.specs:
            if spec.key == spec_key:
                return spec
        logger.warning("Ignoring unavailable plot spec %s", spec_key)
        return None

    def _active_plotdata(self) -> Any:
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return None
        return stream_runtime.plot_data_for_shank(self._active_shank_idx())

    def _active_shank_runtime(self) -> ShankRuntime | None:
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return None
        return stream_runtime.shank_runtime_by_idx.get(self._active_shank_idx())

    def _active_alignment_context(
        self,
    ) -> tuple[AlignmentKey, ActiveAlignment, ShankRuntime] | None:
        key = self.document.selected_alignment_key
        state = self.document.active_alignment_state
        if key is None or state is None:
            return None
        active_alignment = state.active_alignment
        if active_alignment is None:
            return None
        shank_runtime = self._active_shank_runtime()
        if shank_runtime is None or shank_runtime.ephysalign is None:
            return None
        return key, active_alignment, shank_runtime

    def _compute_active_histology(
        self,
        active_alignment: ActiveAlignment,
        shank_runtime: ShankRuntime,
    ) -> AlignmentHistologyData:
        return self.derived_data_service.compute_histology(
            ephysalign=shank_runtime.ephysalign,
            feature=active_alignment.feature,
            track=active_alignment.track,
            region_annotation_source=self.display_state.region_annotation_source,
            region_fp=shank_runtime.region_fp,
            region_label_fp=shank_runtime.region_label_fp,
            region_colour_fp=shank_runtime.region_colour_fp,
        )

    def _probe_extent_render_state(
        self,
        active_alignment: ActiveAlignment,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> ProbeExtentRenderState | None:
        feature = np.asarray(active_alignment.feature, dtype=float)
        if feature.size == 0:
            return None

        offset_um = 1.0
        feature_min_um = float(feature[0] * 1e6)
        feature_max_um = float(feature[-1] * 1e6)
        feature_top_um = feature_max_um - offset_um
        if probe_top_um > feature_top_um:
            fallback_bounds = (
                feature_min_um + offset_um,
                feature_max_um - offset_um,
            )
            tip_bounds_um = fallback_bounds
            top_bounds_um = fallback_bounds
        else:
            tip_bounds_um = (
                feature_min_um + offset_um,
                feature_max_um - (probe_top_um + offset_um),
            )
            top_bounds_um = (
                feature_min_um + (probe_top_um + offset_um),
                feature_max_um - offset_um,
            )

        return ProbeExtentRenderState(
            probe_tip_um=float(probe_tip_um),
            probe_top_um=float(probe_top_um),
            probe_extra_um=float(probe_extra_um),
            feature_min_um=feature_min_um,
            feature_max_um=feature_max_um,
            tip_bounds_um=tip_bounds_um,
            top_bounds_um=top_bounds_um,
        )

    def _perpendicular_feature_grid_um(
        self,
        *,
        shank_runtime: ShankRuntime,
        histology: AlignmentHistologyData,
        brain_atlas: Any,
        extent_m: float,
        probe_margin_um: float,
    ) -> tuple[Any, Any, int] | None:
        depths = shank_runtime.chn_depths
        if depths is None:
            return None
        channel_depths_um = np.asarray(depths, dtype=float)
        if channel_depths_um.size == 0:
            return None
        finite_depths_um = channel_depths_um[np.isfinite(channel_depths_um)]
        if finite_depths_um.size == 0:
            return None

        dv_voxel_m = abs(float(brain_atlas.bc.dxyz[2]))
        if dv_voxel_m <= 0:
            return None

        feat_min_um = min(0.0, float(finite_depths_um.min())) - probe_margin_um
        feat_max_um = float(finite_depths_um.max()) + probe_margin_um
        regions = histology.histology.region
        try:
            has_regions = regions is not None and len(regions) > 0
        except TypeError:
            has_regions = regions is not None
        if has_regions:
            try:
                reg = np.asarray(regions, dtype=float)
            except (TypeError, ValueError):
                logger.debug("Could not coerce histology regions for slice bounds")
            else:
                reg = reg[np.isfinite(reg)]
                if reg.size:
                    feat_min_um = min(feat_min_um, float(reg.min()))
                    feat_max_um = max(feat_max_um, float(reg.max()))

        n_depths = int(round((feat_max_um - feat_min_um) * 1e-6 / dv_voxel_m)) + 1
        if n_depths <= 1:
            n_depths = 2
        feature_grid_um = np.linspace(feat_min_um, feat_max_um, n_depths)
        feature_grid_m = feature_grid_um * 1e-6
        n_perp_samples = int(round(2 * float(extent_m) / dv_voxel_m)) + 1
        if n_perp_samples <= 1:
            n_perp_samples = 2
        return feature_grid_um, feature_grid_m, n_perp_samples

    def _active_shank_idx(self) -> int:
        key = self.document.selected_alignment_key
        if key is not None:
            return key.shank_idx
        return self.document.selected_shank


@dataclass
class AlignmentApp:
    """Small public app port for desktop and future web frontends."""

    commands: AlignmentCommands
    queries: AlignmentQueries
    events: EventBus
