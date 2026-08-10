"""App-level load/cache command handlers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState
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
    LoadDataPrepared,
    ProbeSelectionCacheResult,
    StreamCacheEvicted,
)
from ephys_alignment_gui.controller import AlignmentController
from ephys_alignment_gui.ephys_stream_loader import LoadedEphysSelection
from ephys_alignment_gui.ephys_stream_runtime import StreamKey
from ephys_alignment_gui.load_data_job import LoadDataJob, LoadDataJobRequest
from ephys_alignment_gui.metadata_selection_commands import (
    MetadataSelectionCommandHandler,
)
from ephys_alignment_gui.plotting.payload_cache_factory import (
    EphysPlotPayloadCacheFactory,
)
from ephys_alignment_gui.reference_line_capture import (
    REFERENCE_LINES_NOT_PROVIDED,
    ReferenceLineCapture,
    capture_active_reference_lines_if_provided,
)
from ephys_alignment_gui.session_runtime import (
    LoadDataAlreadyActive,
    LoadDataCachedStreamAvailable,
    LoadDataTarget,
    SessionRuntime,
)
from ephys_alignment_gui.workflow import Failed, PolicyResult

logger = logging.getLogger(__name__)


@dataclass
class LoadDataCommandHandler:
    """Coordinate stream-cache, fresh-load, and loaded-shank transactions."""

    controller: AlignmentController
    data_context: AlignmentDataContext
    display_state: AlignmentDisplayState
    runtime: SessionRuntime
    load_data_job: LoadDataJob
    plot_payload_cache_factory: EphysPlotPayloadCacheFactory
    metadata_commands: MetadataSelectionCommandHandler

    def can_load_data(self) -> PolicyResult:
        """Return whether the selected stream can be loaded."""
        return self.controller.can_load_data()

    def begin_load_data(
        self,
        *,
        recording_id: str,
        probe_name: str,
        target_shank: int,
        outgoing_reference_lines: ReferenceLineCapture = REFERENCE_LINES_NOT_PROVIDED,
    ) -> LoadDataBeginResult | Failed:
        """Prepare or activate the selected stream/shank load transaction."""
        stream_key = self._stream_key_for_selection(recording_id, probe_name)
        load_plan = self.runtime.plan_load_data(
            LoadDataTarget(stream_key=stream_key, shank_idx=target_shank),
            data_loaded=self.controller.document.data_loaded,
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

        capture_result = capture_active_reference_lines_if_provided(
            self.controller,
            outgoing_reference_lines,
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
        selected = self.controller.select_shank(target_shank)
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
        load_plan = self.runtime.plan_load_data(
            LoadDataTarget(stream_key=stream_key, shank_idx=target_shank),
            data_loaded=self.controller.document.data_loaded,
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
        job_result = self.load_data_job.run(LoadDataJobRequest(prepared.shank_idx))
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
        prepared = self.controller.prepare_load_data()
        self.runtime.prepare_fresh_load(stream_key)
        self.display_state.reset_for_active_stream()
        return prepared

    def detach_active_stream(self) -> ActiveStreamDetached:
        """Detach the active stream while preserving cached runtimes."""
        self.runtime.clear_active_stream()
        self.display_state.reset_for_active_stream()
        return ActiveStreamDetached(
            cached_stream_count=len(self.runtime.stream_cache),
        )

    def evict_stream_cache(self) -> StreamCacheEvicted:
        """Evict cached stream runtimes for a recording/session transition."""
        evicted_stream_count = len(self.runtime.stream_cache)
        self.runtime.clear_stream_cache()
        self.display_state.reset_for_active_stream()
        return StreamCacheEvicted(evicted_stream_count=evicted_stream_count)

    def activate_cached_ephys_data(
        self,
        *,
        recording_id: str,
        probe_name: str,
        stream_key: StreamKey,
        shank_idx: int,
    ) -> CachedEphysDataActivated | Failed:
        """Activate cached ephys runtime data for one explicit shank."""
        cached_runtime = self.runtime.cached_stream(stream_key)
        if cached_runtime is None:
            return Failed(f"Cached stream not found: {stream_key}")

        probe = self.metadata_commands.select_probe_metadata(
            recording_id,
            probe_name,
            ephys_stream=cached_runtime.stream,
        )
        if isinstance(probe, Failed):
            return probe

        try:
            stream_runtime = self.runtime.activate_cached_stream_for_shank(
                stream_key,
                shank_idx=shank_idx,
            )
        except Exception as exc:
            return Failed(f"Failed to restore cached stream runtime: {exc}")

        self.controller.finish_load_data(shank_idx)
        return CachedEphysDataActivated(
            stream_runtime=stream_runtime,
            shank_idx=shank_idx,
            probe=probe,
        )

    def _cache_loaded_probe_data(
        self,
        loaded: LoadedEphysSelection,
        *,
        shank_idx: int,
    ) -> FreshEphysDataLoaded:
        """Insert loaded ephys data into runtime cache and mark document loaded."""
        stream_runtime = self.runtime.cache_loaded_stream_data(
            loaded.stream,
            self.plot_payload_cache_factory,
            shank_idx=shank_idx,
        )
        self.controller.finish_load_data(shank_idx)
        return FreshEphysDataLoaded(
            stream_runtime=stream_runtime,
            shank_idx=shank_idx,
        )

    def _stream_key_for_selection(
        self,
        recording_id: str,
        probe_name: str,
    ) -> StreamKey | None:
        try:
            return self.data_context.stream_key_for_selection(
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
