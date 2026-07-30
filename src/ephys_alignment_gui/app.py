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
)
from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.alignment_events import (
    AlignmentEdited,
    AlignmentEditKind,
    ShankChanged,
)
from ephys_alignment_gui.alignment_query_context import AlignmentQueryContext
from ephys_alignment_gui.alignment_read_models import (
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
    ProbeExtentRenderState,
    ScaleFactorRenderState,
)
from ephys_alignment_gui.alignment_render_queries import AlignmentRenderQueries
from ephys_alignment_gui.app_results import (
    ActiveStreamDetached,
    CachedEphysDataActivated,
    FreshEphysDataLoaded,
    LoadDataAlreadyActiveResult,
    LoadDataBeginResult,
    LoadDataCachedActivated,
    LoadDataFreshCompleted,
    LoadDataFreshPrepared,
    LoadDataFreshRequiredResult,
    LoadedShankPrepared,
    ProbeSelectionCacheResult,
    ShankSelectionState,
    StreamCacheEvicted,
    VisitedAlignmentOutputsSaved,
)
from ephys_alignment_gui.controller import (
    AlignmentChoicesUpdated,
    AlignmentController,
    AlignmentEditApplied,
    AlignmentEditNoop,
    AlignmentOutputsSaved,
    Failed,
    LoadDataPrepared,
    MouseRootLoaded,
    NoPreviousAlignments,
    OutputDirectoryDerived,
    OutputRootSet,
    PendingReferenceLinesUpdated,
    PreviousAlignmentSelected,
    PreviousAlignmentsLoaded,
    ProbeSelected,
    RecordingSelected,
    ShankSelected,
)
from ephys_alignment_gui.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.ephys_plot_queries import EphysPlotQueries
from ephys_alignment_gui.ephys_stream_loader import LoadedEphysSelection
from ephys_alignment_gui.ephys_stream_runtime import StreamKey
from ephys_alignment_gui.event_bus import EventBus
from ephys_alignment_gui.histology_data_service import HistologyDataContext
from ephys_alignment_gui.load_data_job import LoadDataJob, LoadDataJobRequest
from ephys_alignment_gui.plot_data_factory import PlotDataFactory
from ephys_alignment_gui.plot_menu_state import PlotMenuState
from ephys_alignment_gui.plot_registry import PlotMenu, PlotSpec
from ephys_alignment_gui.probe_track_service import ProbeTrackService
from ephys_alignment_gui.session_runtime import (
    LoadDataAlreadyActive,
    LoadDataCachedStreamAvailable,
    LoadDataPlan,
    LoadDataTarget,
    SessionRuntime,
)
from ephys_alignment_gui.shank_runtime import ShankRuntime
from ephys_alignment_gui.slice_data_runtime_service import SliceDataRuntimeService
from ephys_alignment_gui.slice_display_policy import SliceDisplayPolicy, SliceSelection
from ephys_alignment_gui.slice_queries import SliceQueries
from ephys_alignment_gui.workflow import Blocked, Ok, PolicyResult

logger = logging.getLogger(__name__)


class _ReferenceLinesNotProvided:
    pass


_REFERENCE_LINES_NOT_PROVIDED = _ReferenceLinesNotProvided()
ReferenceLineCapture = tuple[Any, Any] | None | _ReferenceLinesNotProvided


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
    _load_data_job: LoadDataJob
    _histology_context: HistologyDataContext
    _probe_track_service: ProbeTrackService
    _plot_data_factory: PlotDataFactory
    _derived_data_service: AlignmentDerivedDataService

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

    def capture_active_reference_lines(
        self,
        reference_lines: tuple[Any, Any] | None,
    ) -> PendingReferenceLinesUpdated | Ok | Failed:
        """Capture active reference-line coordinates as document state."""
        if not self._controller.document.data_loaded:
            return Ok()
        result = self._capture_outgoing_reference_lines(reference_lines)
        if result is None:
            return Ok()
        return result

    def set_mouse_root(self, mouse_root: Path) -> MouseRootLoaded | Failed:
        """Load a mouse root and update document metadata."""
        return self._controller.set_mouse_root(mouse_root)

    def clear_histology_context(self) -> Ok:
        """Clear loaded histology runtime data after a mouse-root change."""
        self._histology_context.clear()
        return Ok()

    def set_output_root(self, output_root: Path) -> OutputRootSet | Failed:
        """Set the output root and derive the active probe output directory."""
        return self._controller.set_output_root(output_root)

    def derive_output_directory(self) -> OutputDirectoryDerived | Failed:
        """Derive the active per-probe output directory from document state."""
        return self._controller.derive_output_directory()

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

    def select_recording_metadata(
        self,
        recording_id: str,
    ) -> RecordingSelected | Failed:
        """Select a recording and return its probe choices."""
        return self._controller.select_recording(recording_id)

    def select_probe_metadata(
        self,
        recording_id: str,
        probe_name: str,
    ) -> ProbeSelected | Failed:
        """Select a probe and load lightweight channel metadata."""
        return self._controller.select_probe(recording_id, probe_name)

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

    def _active_or_given_shank(self, shank_idx: int | None) -> int:
        if shank_idx is not None:
            return shank_idx
        return self._controller.document.selected_shank

    def _stream_key_for_selection(
        self,
        recording_id: str,
        probe_name: str,
    ) -> StreamKey | None:
        if self._controller.data_context is None:
            return None
        try:
            return self._controller.data_context.stream_key_for_selection(
                recording_id,
                probe_name,
            )
        except Exception:
            logger.warning(
                "Could not resolve stream key for %s/%s",
                recording_id,
                probe_name,
                exc_info=True,
            )
            return None

    def _capture_active_reference_lines_if_provided(
        self,
        reference_lines: ReferenceLineCapture,
    ) -> Ok | PendingReferenceLinesUpdated | Failed:
        if reference_lines is _REFERENCE_LINES_NOT_PROVIDED:
            return Ok()
        return self.capture_active_reference_lines(reference_lines)

    def can_load_previous_alignments(self) -> Ok | Failed:
        """Return whether previous alignments can be loaded."""
        return self._controller.can_load_previous_alignments()

    def begin_load_data(
        self,
        *,
        recording_id: str,
        probe_name: str,
        target_shank: int,
        outgoing_reference_lines: ReferenceLineCapture = _REFERENCE_LINES_NOT_PROVIDED,
    ) -> LoadDataBeginResult | Failed:
        """Prepare or activate the selected stream/shank load transaction."""
        stream_key = self._stream_key_for_selection(recording_id, probe_name)
        load_plan = self._runtime.plan_load_data(
            LoadDataTarget(stream_key=stream_key, shank_idx=target_shank),
            data_loaded=self._controller.document.data_loaded,
        )

        if isinstance(load_plan, LoadDataAlreadyActive):
            logger.info(
                "Data already loaded for stream %s shank %s; skipping load",
                stream_key,
                target_shank,
            )
            return LoadDataAlreadyActiveResult(
                stream_key=stream_key,
                shank_idx=target_shank,
            )

        capture_result = self._capture_active_reference_lines_if_provided(
            outgoing_reference_lines
        )
        if isinstance(capture_result, Failed):
            return capture_result

        if isinstance(load_plan, LoadDataCachedStreamAvailable):
            if stream_key is None:
                return Failed("Cached stream activation requires a stream key.")
            self.detach_active_stream()
            result = self.activate_cached_ephys_data(
                recording_id=recording_id,
                probe_name=probe_name,
                stream_key=stream_key,
                shank_idx=load_plan.target.shank_idx,
            )
            if isinstance(result, Failed):
                return result
            return LoadDataCachedActivated(stream_key=stream_key, activated=result)

        prepared = self.prepare_fresh_ephys_load(stream_key)
        selected = self._controller.select_shank(target_shank)
        if isinstance(selected, Failed):
            return selected
        return LoadDataFreshPrepared(
            stream_key=stream_key,
            shank_idx=selected.shank_idx,
            preserve_plot_selection=prepared.preserve_plot_selection,
        )

    def activate_cached_probe_selection(
        self,
        *,
        recording_id: str,
        probe_name: str,
        target_shank: int,
    ) -> ProbeSelectionCacheResult | Failed:
        """Activate a cached probe selection or report that fresh loading is needed."""
        stream_key = self._stream_key_for_selection(recording_id, probe_name)
        load_plan = self._runtime.plan_load_data(
            LoadDataTarget(stream_key=stream_key, shank_idx=target_shank),
            data_loaded=self._controller.document.data_loaded,
        )

        if isinstance(load_plan, LoadDataAlreadyActive):
            return LoadDataAlreadyActiveResult(
                stream_key=stream_key,
                shank_idx=target_shank,
            )
        if not isinstance(load_plan, LoadDataCachedStreamAvailable):
            return LoadDataFreshRequiredResult(
                stream_key=stream_key,
                shank_idx=target_shank,
            )
        if stream_key is None:
            return Failed("Cached stream activation requires a stream key.")

        self.detach_active_stream()
        result = self.activate_cached_ephys_data(
            recording_id=recording_id,
            probe_name=probe_name,
            stream_key=stream_key,
            shank_idx=load_plan.cached_shank_idx,
        )
        if isinstance(result, Failed):
            return result
        return LoadDataCachedActivated(stream_key=stream_key, activated=result)

    def complete_fresh_load_data(
        self,
        prepared: LoadDataFreshPrepared,
    ) -> LoadDataFreshCompleted | Failed:
        """Run fresh ephys and histology load steps for a prepared transaction."""
        job_result = self._load_data_job.run(LoadDataJobRequest(prepared.shank_idx))
        if isinstance(job_result, Failed):
            return job_result
        ephys_result = self._cache_loaded_probe_data(
            job_result.ephys,
            shank_idx=prepared.shank_idx,
        )
        return LoadDataFreshCompleted(
            stream_key=prepared.stream_key,
            ephys=ephys_result,
            histology=job_result.histology,
            preserve_plot_selection=prepared.preserve_plot_selection,
        )

    def prepare_fresh_ephys_load(
        self,
        stream_key: StreamKey | None,
    ) -> LoadDataPrepared:
        """Mark data unloaded and discard stale active/cache state."""
        prepared = self._controller.prepare_load_data()
        self._runtime.prepare_fresh_load(stream_key)
        self._display_state.reset_for_active_stream()
        return prepared

    def detach_active_stream(self) -> ActiveStreamDetached:
        """Detach the active stream while preserving cached runtimes."""
        self._runtime.clear_active_stream()
        self._display_state.reset_for_active_stream()
        return ActiveStreamDetached(
            cached_stream_count=len(self._runtime.stream_cache),
        )

    def evict_stream_cache(self) -> StreamCacheEvicted:
        """Evict cached stream runtimes for a recording/session transition."""
        evicted_stream_count = len(self._runtime.stream_cache)
        self._runtime.clear_stream_cache()
        self._display_state.reset_for_active_stream()
        return StreamCacheEvicted(evicted_stream_count=evicted_stream_count)

    def _cache_loaded_probe_data(
        self,
        loaded: LoadedEphysSelection,
        *,
        shank_idx: int,
    ) -> FreshEphysDataLoaded:
        """Insert loaded ephys data into runtime cache and mark document loaded."""
        stream_runtime = self._runtime.cache_loaded_stream_data(
            loaded.stream,
            self._plot_data_factory,
            shank_idx=shank_idx,
        )
        self._controller.finish_load_data(shank_idx)
        return FreshEphysDataLoaded(
            stream_runtime=stream_runtime,
            shank_idx=shank_idx,
        )

    def activate_cached_ephys_data(
        self,
        *,
        recording_id: str,
        probe_name: str,
        stream_key: StreamKey,
        shank_idx: int,
    ) -> CachedEphysDataActivated | Failed:
        """Activate cached ephys runtime data for one explicit shank."""
        cached_runtime = self._runtime.cached_stream(stream_key)
        if cached_runtime is None:
            return Failed(f"Cached stream not found: {stream_key}")

        probe = self._controller.select_probe(
            recording_id,
            probe_name,
            ephys_stream=cached_runtime.stream,
        )
        if isinstance(probe, Failed):
            return probe

        try:
            stream_runtime = self._runtime.activate_cached_stream_for_shank(
                stream_key,
                shank_idx=shank_idx,
            )
        except Exception as exc:
            return Failed(f"Failed to restore cached stream runtime: {exc}")

        self._controller.finish_load_data(shank_idx)
        return CachedEphysDataActivated(
            stream_runtime=stream_runtime,
            shank_idx=shank_idx,
            probe=probe,
        )

    def prepare_loaded_shank(
        self,
        shank_idx: int,
        *,
        select_default_alignment_if_empty: bool = True,
    ) -> LoadedShankPrepared | Failed:
        """Prepare Qt-free runtime state for a loaded active shank."""
        stream_runtime = self._runtime.active_stream_runtime
        if stream_runtime is None:
            return Failed("No active stream runtime for shank preparation")

        try:
            shank_runtime = stream_runtime.shank_runtime_for(shank_idx)
        except Exception as exc:
            return Failed(f"Failed to prepare shank runtime: {exc}")

        n_channels = len(shank_runtime.collection.depths)
        brain_atlas = self._histology_context.brain_atlas
        if brain_atlas is None:
            return LoadedShankPrepared(
                shank_idx=shank_idx,
                n_channels=n_channels,
                histology_available=False,
            )

        probe = self._controller.data_context.probe_info
        if probe is None:
            return Failed("No probe selected. Please select a probe first.")

        try:
            track_annotations_ras = shank_runtime.track_annotations_ras
            if track_annotations_ras is None:
                track_annotations_ras = (
                    self._probe_track_service.load_track_annotations(
                        probe=probe,
                        shank_idx=shank_idx,
                        brain_atlas=brain_atlas,
                    )
                )
        except Exception as exc:
            return Failed(f"Failed to load shank track annotations: {exc}")

        choices = self._controller.active_alignment_choices(shank_idx)
        if isinstance(choices, Failed):
            return choices

        active_state = self._controller.document.active_alignment_state
        if (
            select_default_alignment_if_empty
            and active_state is not None
            and active_state.active_alignment is None
        ):
            selected = self._controller.select_previous_alignment(
                0,
                shank_idx=shank_idx,
            )
            if isinstance(selected, Failed):
                return selected

        initialized = self._controller.initialize_shank_runtime(
            shank_runtime,
            track_annotations_ras=track_annotations_ras,
            brain_atlas=brain_atlas,
        )
        if isinstance(initialized, Failed):
            return initialized

        return LoadedShankPrepared(
            shank_idx=shank_idx,
            n_channels=n_channels,
            histology_available=True,
            alignment_choices=choices.choices,
        )

    def can_load_data(self) -> PolicyResult:
        """Return whether the selected stream can be loaded."""
        return self._controller.can_load_data()

    def can_save_alignment_output(self) -> Ok | Blocked:
        """Return whether visited alignment outputs can be saved."""
        return self._controller.can_save_alignment_output()

    def save_visited_alignment_outputs(
        self,
        *,
        use_docdb: bool,
    ) -> VisitedAlignmentOutputsSaved | Blocked | Failed:
        """Persist outputs for every visited alignment in the active stream."""
        ready = self._controller.can_save_alignment_output()
        if isinstance(ready, Blocked):
            return ready

        output_inputs, states_by_key = self._visited_alignment_output_inputs()
        if not output_inputs:
            return Failed("No visited alignments are ready to save")

        outputs = self._controller.build_alignment_outputs(output_inputs)
        if isinstance(outputs, Failed):
            return outputs

        for key, state in states_by_key.items():
            alignment = state.active_alignment
            if alignment is not None:
                state.add_alignment(alignment.feature, alignment.track)

        logger.info("Saving output files to results folder...")
        saved_outputs: dict[AlignmentKey, AlignmentOutputsSaved] = {}
        for key, output in outputs.items():
            state = states_by_key[key]
            saved = self._controller.save_alignment_output(
                output,
                state.alignments,
                key.shank_idx,
                use_docdb,
            )
            if isinstance(saved, Failed):
                return saved
            saved_outputs[key] = saved

        active_choices: list[str] | None = None
        choices = self._controller.active_alignment_choices(
            self._active_or_given_shank(None)
        )
        if isinstance(choices, AlignmentChoicesUpdated):
            active_choices = choices.choices
        elif isinstance(choices, Failed):
            logger.error(choices.message)

        return VisitedAlignmentOutputsSaved(
            saved_count=len(saved_outputs),
            saved_outputs=saved_outputs,
            active_choices=active_choices,
        )

    def _visited_alignment_output_inputs(
        self,
    ) -> tuple[
        dict[AlignmentKey, tuple[Any, Any]],
        dict[AlignmentKey, Any],
    ]:
        """Collect channel-location save inputs for visited shanks."""
        stream_runtime = self._runtime.active_stream_runtime
        if stream_runtime is None:
            return {}, {}
        probe = self._controller.data_context.probe_info
        if probe is None:
            return {}, {}

        states_for_probe = (
            self._controller.document.alignment_states_for_current_probe()
        )
        output_inputs: dict[AlignmentKey, tuple[Any, Any]] = {}
        states_by_key: dict[AlignmentKey, Any] = {}
        for shank_idx, shank_runtime in stream_runtime.visited_shank_runtimes().items():
            key = AlignmentKey(
                recording_id=probe.recording_id,
                ephys_collection=probe.ephys_collection,
                shank_idx=shank_idx,
            )
            state = states_for_probe.get(key)
            if state is None or state.active_alignment is None:
                continue
            if shank_runtime.ephysalign is None or shank_runtime.chn_coords is None:
                logger.info(
                    "Skipping shank %d during save because it has not been rendered",
                    shank_idx + 1,
                )
                continue
            alignment = state.active_alignment
            channel_locations_ras = (
                self._derived_data_service.compute_channel_locations(
                    ephysalign=shank_runtime.ephysalign,
                    feature=alignment.feature,
                    track=alignment.track,
                )
            )
            output_inputs[key] = (channel_locations_ras, shank_runtime.chn_coords)
            states_by_key[key] = state
        return output_inputs, states_by_key

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

    def offset_active_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        track_shift_m: float = 0.0,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a tip-offset edit using app-owned display settings."""
        return self.offset_alignment_from_tip(
            tip_position_um=tip_position_um,
            probe_tip_um=self._display_state.depth_view.probe_tip_um,
            lin_fit=self._display_state.edit_settings.lin_fit,
            track_shift_m=track_shift_m,
        )

    def nudge_active_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        track_shift_m: float,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a bounded tip-offset nudge for the active alignment."""
        if not self._active_alignment_can_shift(track_shift_m):
            return AlignmentEditNoop()
        return self.offset_active_alignment_from_tip(
            tip_position_um=tip_position_um,
            track_shift_m=track_shift_m,
        )

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

    def fit_active_alignment_from_pending_reference_lines(
        self,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a fit edit from document-owned pending reference lines."""
        shank_runtime = self._active_shank_runtime()
        if shank_runtime is None:
            return Failed("Cannot fit alignment: active shank runtime is not loaded")

        pending_lines = self._controller.active_pending_reference_lines(
            shank_runtime.shank_idx
        )
        if isinstance(pending_lines, Failed):
            return pending_lines
        if pending_lines is None:
            line_features_um = np.array([], dtype=float)
            line_tracks_um = np.array([], dtype=float)
        else:
            line_features_um = pending_lines.feature_positions_um
            line_tracks_um = pending_lines.track_positions_um

        return self.fit_alignment_to_reference_lines(
            shank_runtime,
            line_features_um=line_features_um,
            line_tracks_um=line_tracks_um,
            lin_fit=self._display_state.edit_settings.lin_fit,
            extend_feature=self._display_state.edit_settings.extend_feature,
        )

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

    def reset_active_alignment_to_initial(
        self,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Reset active alignment using the active runtime and display settings."""
        shank_runtime = self._active_shank_runtime()
        if shank_runtime is None:
            return Failed("Cannot reset alignment: active shank runtime is not loaded")
        return self.reset_alignment_to_initial(
            shank_runtime,
            lin_fit=self._display_state.edit_settings.lin_fit,
        )

    def _active_alignment_can_shift(self, track_shift_m: float) -> bool:
        """Return whether a bounded nudge keeps the alignment inside channel depths."""
        if track_shift_m == 0:
            return True
        state = self._controller.document.active_alignment_state
        alignment = None if state is None else state.active_alignment
        shank_runtime = self._active_shank_runtime()
        if (
            alignment is None
            or shank_runtime is None
            or shank_runtime.chn_depths is None
        ):
            return False

        channel_depths_m = np.asarray(shank_runtime.chn_depths, dtype=float) / 1e6
        if channel_depths_m.size == 0:
            return False
        if track_shift_m < 0:
            return alignment.track[-1] + track_shift_m >= float(
                np.max(channel_depths_m)
            )
        return alignment.track[0] + track_shift_m <= float(np.min(channel_depths_m))

    def _active_shank_runtime(self) -> ShankRuntime | None:
        """Return active shank runtime data for command-side alignment edits."""
        stream_runtime = self._runtime.active_stream_runtime
        if stream_runtime is None:
            return None
        return stream_runtime.shank_runtime_by_idx.get(
            self._controller.document.selected_shank
        )

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
    slice_data_runtime_service: SliceDataRuntimeService = field(
        default_factory=SliceDataRuntimeService
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

    def active_reference_line_state(
        self,
        shank_idx: int | None = None,
    ) -> ActiveReferenceLineRenderState | None:
        """Return pending or previous-alignment reference lines for rendering."""
        state = self.document.active_alignment_state
        key = self.document.selected_alignment_key
        if state is None or key is None:
            return None
        if shank_idx is not None and key.shank_idx != shank_idx:
            return None

        pending = state.pending_reference_lines
        if pending is not None:
            return ActiveReferenceLineRenderState(
                feature_positions_um=pending.feature_positions_um,
                track_positions_um=pending.track_positions_um,
            )

        feature_prev = state.feature_prev
        if feature_prev is None or not np.any(feature_prev):
            return None
        return ActiveReferenceLineRenderState(
            feature_positions_um=np.asarray(feature_prev)[1:-1] * 1e6,
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

    def active_mouse_root_path(self) -> Path | None:
        """Return the active mouse-root path, if one is loaded."""
        if self.data_context is None or self.data_context.mouse_root is None:
            return None
        return self.data_context.mouse_root.root

    def mouse_root_loaded(self) -> bool:
        """Return whether an input mouse-root datapackage is loaded."""
        return self.active_mouse_root_path() is not None

    def active_output_root(self) -> Path | None:
        """Return the active output root, if one has been set."""
        return self.document.output_root

    def has_output_directory(self) -> bool:
        """Return whether the active probe output directory is available."""
        return self.document.output_directory is not None

    def active_unit_filter(self) -> str:
        """Return the selected unit subset for active ephys plot data."""
        return self._ephys_plot_queries().active_unit_filter()

    def resolve_shank_preserve_plot_selection(
        self,
        preserve_plot_selection: bool | None,
    ) -> bool:
        """Return whether shank redraw should preserve current plot selections."""
        if preserve_plot_selection is None:
            return self.document.data_loaded
        return preserve_plot_selection

    def _query_context(self) -> AlignmentQueryContext:
        return AlignmentQueryContext(
            document=self.document,
            runtime=self.runtime,
        )

    def _ephys_plot_queries(self) -> EphysPlotQueries:
        return EphysPlotQueries(
            context=self._query_context(),
            display_state=self.display_state,
            derived_data_service=self.derived_data_service,
            histology_context=self.histology_context,
        )

    def _alignment_render_queries(self) -> AlignmentRenderQueries:
        return AlignmentRenderQueries(
            context=self._query_context(),
            display_state=self.display_state,
            derived_data_service=self.derived_data_service,
        )

    def _slice_queries(self) -> SliceQueries:
        return SliceQueries(
            context=self._query_context(),
            render_queries=self._alignment_render_queries(),
            derived_data_service=self.derived_data_service,
            slice_data_runtime_service=self.slice_data_runtime_service,
            histology_context=self.histology_context,
            slice_service=self.slice_service,
            slice_display_policy=self.slice_display_policy,
        )

    def prepare_active_shank_plot_data_state(
        self,
        *,
        unit_filter: str | None = None,
    ) -> ActiveShankPlotDataState | None:
        """Materialize active shank PlotData and return frontend-safe bounds."""
        return self._ephys_plot_queries().prepare_active_shank_plot_data_state(
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
        return self._ephys_plot_queries().active_plot_menu_state(
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
        return self._ephys_plot_queries().active_plot_spec(
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
        return self._ephys_plot_queries().active_plot_payload(
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
        return self._ephys_plot_queries().active_plot_bounds(
            spec_key,
            raw_image_payloads=raw_image_payloads,
        )

    def active_in_brain_depths_um(self) -> Any:
        """Return active PlotData in-brain depths, if available."""
        return self._ephys_plot_queries().active_in_brain_depths_um()

    def active_in_brain_depths_for_alignment(self) -> Any:
        """Return active channel depths whose aligned CCF annotation is not root."""
        return self._ephys_plot_queries().active_in_brain_depths_for_alignment()

    def prepare_active_slice_screen_data(self) -> ActiveSliceDataState | None:
        """Materialize active slice data when histology runtime is available."""
        return self._slice_queries().prepare_active_slice_screen_data()

    def active_cluster_detail(
        self,
        cluster_idx: int,
    ) -> ClusterDetailRenderState | None:
        """Return autocorrelogram/template detail for one active cluster."""
        return self._ephys_plot_queries().active_cluster_detail(cluster_idx)

    def active_session_notes(self) -> str:
        """Return notes for the active ephys stream, if any."""
        return self._ephys_plot_queries().active_session_notes()

    def active_histology_region_id(self, region_idx: int) -> int | None:
        """Return an active histology region id by plotted region index."""
        return self._alignment_render_queries().active_histology_region_id(region_idx)

    def active_alignment_render_state(self) -> ActiveAlignmentRenderState | None:
        """Return derived render data for the active alignment, if available."""
        return self._alignment_render_queries().active_alignment_render_state()

    def active_histology_panel_state(
        self,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> HistologyPanelRenderState | None:
        """Return histology-region render data for the active alignment."""
        return self._alignment_render_queries().active_histology_panel_state(
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
        return self._alignment_render_queries().probe_extent_render_state(
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
        return self._alignment_render_queries().active_scale_factor_state(
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
        return self._alignment_render_queries().active_nearby_boundary_state(
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
        return self._alignment_render_queries().active_fit_plot_state(
            depth_um=depth_um,
            lin_fit=lin_fit,
        )

    def ensure_active_slice_data_state(self) -> ActiveSliceDataState | None:
        """Build/cache and return coronal slice data for the active alignment."""
        return self._slice_queries().ensure_active_slice_data_state()

    def active_slice_data_state(self) -> ActiveSliceDataState | None:
        """Return currently active coronal slice data without building it."""
        return self._slice_queries().active_slice_data_state()

    def active_slice_data_by_attr(self) -> dict[str, Any]:
        """Return active slice data keyed by menu payload data-attr names."""
        return self._slice_queries().active_slice_data_by_attr()

    def active_slice_menu_state(
        self,
        *,
        offline: bool,
        previous_selection: SliceSelection | None = None,
    ) -> ActiveSliceMenuState | None:
        """Return menu and fallback-selection state for active slice data."""
        return self._slice_queries().active_slice_menu_state(
            offline=offline,
            previous_selection=previous_selection,
        )

    def active_slice_render_state(
        self,
        selection: SliceSelection,
    ) -> ActiveSliceRenderState | None:
        """Return a render payload for one active coronal slice selection."""
        return self._slice_queries().active_slice_render_state(selection)

    def active_perpendicular_slice_state(
        self,
        channel_name: str,
        *,
        extent_m: float = 500e-6,
        probe_margin_um: float = 100.0,
    ) -> PerpendicularSliceRenderState | None:
        """Build/cache and return a perpendicular slice render payload."""
        return self._slice_queries().active_perpendicular_slice_state(
            channel_name=channel_name,
            extent_m=extent_m,
            probe_margin_um=probe_margin_um,
        )

    def _active_shank_idx(self) -> int:
        return self._query_context().active_shank_idx()


@dataclass
class AlignmentApp:
    """Small public app port for desktop and future web frontends."""

    commands: AlignmentCommands
    queries: AlignmentQueries
    events: EventBus
