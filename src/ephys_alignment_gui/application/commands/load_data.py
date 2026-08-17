"""App-level load/cache command handlers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from ephys_alignment_gui.application.commands.load_data_lifecycle import (
    CancelledFreshLoadExecution,
    LoadDataExecutionLifecycle,
)
from ephys_alignment_gui.application.commands.metadata_selection import (
    MetadataSelectionCommandHandler,
)
from ephys_alignment_gui.application.results import (
    ActiveStreamDetached,
    CachedEphysDataActivated,
    FreshEphysDataLoaded,
    FreshLoadExecution,
    FreshLoadJobInvocation,
    LoadDataAlreadyActiveResult,
    LoadDataBeginResult,
    LoadDataCachedActivated,
    LoadDataFreshCompleted,
    LoadDataFreshPrepared,
    LoadDataFreshRequiredResult,
    LoadDataPreloadSkipped,
    LoadDataPrepared,
    LoadDataStaleResultIgnored,
    ProbeSelectionCacheResult,
)
from ephys_alignment_gui.application.results import (
    StreamCacheEvicted as StreamCacheEvictedResult,
)
from ephys_alignment_gui.application.save_runtime_dependencies import (
    describe_save_runtime_dependencies,
    plan_save_runtime_dependencies,
)
from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.core.alignment_events import (
    FreshLoadCompleted,
    HistologyLoadReported,
    LoadDataCancelled,
    LoadDataFailed,
    LoadDataProgressed,
    StreamActivated,
    StreamCacheEvicted,
    StreamDetached,
)
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.reference_line_capture import (
    REFERENCE_LINES_NOT_PROVIDED,
    ReferenceLineCapture,
    capture_active_reference_lines_if_provided,
)
from ephys_alignment_gui.core.workflow import Failed, Ok, PolicyResult
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.io.ephys_stream_loader import LoadedEphysSelection
from ephys_alignment_gui.io.load_data_job import (
    LoadDataCancelToken,
    LoadDataJob,
    LoadDataJobCancelled,
    LoadDataJobCompleted,
    LoadDataJobProgress,
    LoadDataJobRequest,
    LoadDataProgressCallback,
)
from ephys_alignment_gui.io.load_data_target import LoadDataJobTarget
from ephys_alignment_gui.plotting.payload_cache_factory import (
    EphysPlotPayloadCacheFactory,
)
from ephys_alignment_gui.plotting.payload_warmup import (
    PlotPayloadCacheWarmed,
    PlotPayloadWarmupJob,
    PlotPayloadWarmupRequest,
)
from ephys_alignment_gui.runtime.ephys_stream import StreamKey
from ephys_alignment_gui.runtime.histology_loader import (
    HistologyDataAlreadyLoaded,
    HistologyDataLoaded,
    HistologyDataUnavailable,
    HistologyLoadResult,
    HistologyRuntimeLoader,
)
from ephys_alignment_gui.runtime.session import (
    LoadDataAlreadyActive,
    LoadDataCachedStreamAvailable,
    LoadDataTarget,
    SessionRuntime,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadDataCommandHandler:
    """Coordinate stream-cache, fresh-load, and loaded-shank transactions."""

    controller: AlignmentController
    data_context: AlignmentDataContext
    display_state: AlignmentDisplayState
    runtime: SessionRuntime
    load_data_job: LoadDataJob
    load_lifecycle: LoadDataExecutionLifecycle
    preload_lifecycle: LoadDataExecutionLifecycle
    histology_runtime_loader: HistologyRuntimeLoader
    plot_payload_cache_factory: EphysPlotPayloadCacheFactory
    metadata_commands: MetadataSelectionCommandHandler
    events: EventBus

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
        preserve_plot_selection: bool | None = None,
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

        target = self._fresh_load_target_for_selection(
            recording_id=recording_id,
            probe_name=probe_name,
            stream_key=stream_key,
            shank_idx=target_shank,
        )
        if isinstance(target, Failed):
            return target

        prepared = self.prepare_fresh_ephys_load(
            stream_key,
            preserve_plot_selection=preserve_plot_selection,
        )
        selected = self.controller.select_shank(target_shank)
        if isinstance(selected, Failed):
            return selected
        target = replace(target, shank_idx=selected.shank_idx)
        return LoadDataFreshPrepared(
            stream_key=stream_key,
            shank_idx=selected.shank_idx,
            preserve_plot_selection=prepared.preserve_plot_selection,
            target=target,
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
        *,
        progress: LoadDataProgressCallback | None = None,
        cancel_token: LoadDataCancelToken | None = None,
    ) -> LoadDataFreshCompleted | LoadDataJobCancelled | Failed:
        """Run and activate a fresh load transaction synchronously."""
        execution = self.start_fresh_load_data(prepared, cancel_token=cancel_token)
        job_result = self.run_started_fresh_load_data(execution, progress=progress)
        if isinstance(job_result, Failed | LoadDataJobCancelled):
            return job_result
        activated = self.activate_started_fresh_load_data(execution, job_result)
        if isinstance(activated, LoadDataStaleResultIgnored):
            return LoadDataJobCancelled(
                target=prepared.target,
                reason=activated.reason,
            )
        return activated

    def start_fresh_load_data(
        self,
        prepared: LoadDataFreshPrepared,
        *,
        cancel_token: LoadDataCancelToken | None = None,
    ) -> FreshLoadExecution:
        """Start a tracked foreground fresh-load execution."""
        execution, cancelled = self.load_lifecycle.start(
            prepared,
            cancel_token=cancel_token,
        )
        if cancelled is not None:
            self._emit_cancelled_execution(cancelled)
        return execution

    def cancel_active_fresh_load(
        self,
        reason: str,
    ) -> LoadDataJobCancelled | None:
        """Cancel the active foreground fresh-load execution, if present."""
        cancelled = self.load_lifecycle.cancel_active(reason)
        if cancelled is None:
            return None
        self._emit_cancelled_execution(cancelled)
        return LoadDataJobCancelled(
            target=cancelled.execution.prepared.target,
            reason=reason,
        )

    def run_fresh_load_data(
        self,
        prepared: LoadDataFreshPrepared,
        *,
        progress: LoadDataProgressCallback | None = None,
        cancel_token: LoadDataCancelToken | None = None,
    ) -> LoadDataJobCompleted | LoadDataJobCancelled | Failed:
        """Run the fresh load job without activating the loaded stream."""
        execution = self.start_fresh_load_data(prepared, cancel_token=cancel_token)
        job_result = self.run_started_fresh_load_data(execution, progress=progress)
        if isinstance(job_result, LoadDataJobCompleted):
            self.load_lifecycle.finish(execution)
        return job_result

    def run_started_fresh_load_data(
        self,
        execution: FreshLoadExecution,
        *,
        progress: LoadDataProgressCallback | None = None,
    ) -> LoadDataJobCompleted | LoadDataJobCancelled | Failed:
        """Run a tracked fresh-load job without activating its result."""
        invocation = self.fresh_load_job_invocation(execution)
        if isinstance(invocation, LoadDataJobCancelled):
            return invocation

        def _progress(event: LoadDataJobProgress) -> None:
            self.publish_fresh_load_progress(execution, event)
            if progress is not None:
                progress(event)

        job_result = self.run_fresh_load_job(invocation, progress=_progress)
        return self.publish_started_fresh_load_job_result(execution, job_result)

    def fresh_load_job_invocation(
        self,
        execution: FreshLoadExecution,
    ) -> FreshLoadJobInvocation | LoadDataJobCancelled:
        """Return a runnable job invocation for an active fresh-load execution."""
        prepared = execution.prepared
        cancel_token = self.load_lifecycle.cancel_token_for(execution)
        if cancel_token is None:
            return LoadDataJobCancelled(
                target=prepared.target,
                reason="Fresh load request is no longer active.",
            )
        return FreshLoadJobInvocation(
            execution=execution,
            request=LoadDataJobRequest(prepared.target, load_id=execution.load_id),
            cancel_token=cancel_token,
        )

    def run_fresh_load_job(
        self,
        invocation: FreshLoadJobInvocation,
        *,
        progress: LoadDataProgressCallback | None = None,
    ) -> LoadDataJobCompleted | LoadDataJobCancelled | Failed:
        """Run fresh-load IO without publishing app events."""
        return self.load_data_job.run(
            invocation.request,
            progress=progress,
            cancel_token=invocation.cancel_token,
        )

    def publish_fresh_load_progress(
        self,
        execution: FreshLoadExecution,
        event: LoadDataJobProgress,
    ) -> None:
        """Publish one fresh-load progress event on the app event bus."""
        if not self.load_lifecycle.is_active(execution):
            return
        prepared = execution.prepared
        self.events.emit(
            LoadDataProgressed(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                phase=event.phase,
                status=event.status,
                message=event.message,
                load_id=event.load_id
                if event.load_id is not None
                else execution.load_id,
            )
        )

    def publish_started_fresh_load_job_result(
        self,
        execution: FreshLoadExecution,
        job_result: LoadDataJobCompleted | LoadDataJobCancelled | Failed,
    ) -> LoadDataJobCompleted | LoadDataJobCancelled | Failed:
        """Publish result events for a tracked fresh-load job."""
        prepared = execution.prepared
        if isinstance(job_result, Failed):
            self.load_lifecycle.finish(execution)
            self.events.emit(
                LoadDataFailed(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    message=job_result.message,
                    load_id=execution.load_id,
                )
            )
        elif isinstance(job_result, LoadDataJobCancelled):
            self.load_lifecycle.finish(execution)
            self.events.emit(
                LoadDataCancelled(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    reason=job_result.reason,
                    load_id=execution.load_id,
                )
            )
        else:
            if not self.load_lifecycle.is_active(execution):
                return LoadDataJobCancelled(
                    target=prepared.target,
                    reason="Fresh load result is stale and was ignored.",
                )
            self.events.emit(
                FreshLoadCompleted(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    warning_messages=tuple(
                        warning.message for warning in job_result.warnings
                    ),
                    load_id=execution.load_id,
                )
            )
        return job_result

    def publish_promoted_preload_job_result(
        self,
        preload_execution: FreshLoadExecution,
        foreground_execution: FreshLoadExecution,
        job_result: LoadDataJobCompleted | LoadDataJobCancelled | Failed,
    ) -> (
        LoadDataJobCompleted
        | LoadDataJobCancelled
        | LoadDataStaleResultIgnored
        | Failed
    ):
        """Publish a completed preload as the foreground load result."""
        try:
            if isinstance(job_result, Failed | LoadDataJobCancelled):
                return self.publish_started_fresh_load_job_result(
                    foreground_execution,
                    job_result,
                )

            if not self.preload_lifecycle.is_active(preload_execution):
                cancelled = LoadDataJobCancelled(
                    target=foreground_execution.prepared.target,
                    reason="Promoted preload request is no longer active.",
                )
                return self.publish_started_fresh_load_job_result(
                    foreground_execution,
                    cancelled,
                )

            stale = self._stale_promoted_preload_result_reason(
                preload_execution,
                foreground_execution,
                job_result,
            )
            if stale is not None:
                self.load_lifecycle.finish(foreground_execution)
                self.events.emit(
                    LoadDataFailed(
                        stream_key=foreground_execution.prepared.stream_key,
                        shank_idx=foreground_execution.prepared.shank_idx,
                        message=stale,
                        load_id=foreground_execution.load_id,
                    )
                )
                return LoadDataStaleResultIgnored(
                    load_id=foreground_execution.load_id,
                    stream_key=foreground_execution.prepared.stream_key,
                    shank_idx=foreground_execution.prepared.shank_idx,
                    reason=stale,
                )

            return self.publish_started_fresh_load_job_result(
                foreground_execution,
                job_result,
            )
        finally:
            self.preload_lifecycle.finish(preload_execution)

    def activate_started_fresh_load_data(
        self,
        execution: FreshLoadExecution,
        job_result: LoadDataJobCompleted,
    ) -> LoadDataFreshCompleted | LoadDataStaleResultIgnored | Failed:
        """Activate a completed load only if its request is still current."""
        if not self.load_lifecycle.is_active(execution):
            return LoadDataStaleResultIgnored(
                load_id=execution.load_id,
                stream_key=execution.prepared.stream_key,
                shank_idx=execution.prepared.shank_idx,
                reason="Fresh load request is no longer active.",
            )

        try:
            return self.activate_completed_fresh_load_data(
                execution.prepared,
                job_result,
                load_id=execution.load_id,
            )
        finally:
            self.load_lifecycle.finish(execution)

    def activate_completed_fresh_load_data(
        self,
        prepared: LoadDataFreshPrepared,
        job_result: LoadDataJobCompleted,
        *,
        load_id: int | None = None,
    ) -> LoadDataFreshCompleted | Failed:
        """Cache/activate completed fresh-load data if its target is still current."""
        stale = self._stale_fresh_load_reason(prepared, job_result)
        if stale is not None:
            self.events.emit(
                LoadDataFailed(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    message=stale,
                    load_id=load_id,
                )
            )
            return Failed(stale)

        self.histology_runtime_loader.activate_result(
            job_result.histology,
            mouse_root=prepared.target.mouse_root,
        )
        self._emit_histology_report(prepared, job_result.histology, load_id=load_id)
        ephys_result = self._cache_loaded_probe_data(
            job_result.ephys,
            shank_idx=prepared.shank_idx,
            activate=True,
        )
        self.events.emit(
            StreamActivated(
                source="fresh",
                stream_key=prepared.target.stream_key,
                shank_idx=prepared.shank_idx,
                active_key=self.controller.document.selected_alignment_key,
                preserve_plot_selection=prepared.preserve_plot_selection,
                load_id=load_id,
            )
        )
        return LoadDataFreshCompleted(
            stream_key=prepared.stream_key,
            target=prepared.target,
            ephys=ephys_result,
            histology=job_result.histology,
            preserve_plot_selection=prepared.preserve_plot_selection,
        )

    def cache_completed_fresh_load_data(
        self,
        job_result: LoadDataJobCompleted,
    ) -> FreshEphysDataLoaded | Failed:
        """Cache a completed fresh-load result without activating it."""
        if job_result.ephys.stream.stream_key != job_result.target.stream_key:
            return Failed(
                "Loaded stream does not match completed load target: "
                f"{job_result.ephys.stream.stream_key!r} != "
                f"{job_result.target.stream_key!r}"
            )
        return self._cache_loaded_probe_data(
            job_result.ephys,
            shank_idx=job_result.target.shank_idx,
            activate=False,
        )

    def prepare_fresh_ephys_load(
        self,
        stream_key: StreamKey | None,
        *,
        preserve_plot_selection: bool | None = None,
    ) -> LoadDataPrepared:
        """Mark data unloaded and discard stale active/cache state."""
        prepared = self.controller.prepare_load_data()
        if preserve_plot_selection is not None:
            prepared = replace(
                prepared,
                preserve_plot_selection=preserve_plot_selection,
            )
        self.runtime.prepare_fresh_load(stream_key)
        self.display_state.reset_for_active_stream()
        return prepared

    def detach_active_stream(self) -> ActiveStreamDetached:
        """Detach the active stream while preserving cached runtimes."""
        self.runtime.clear_active_stream()
        self.display_state.reset_for_active_stream()
        result = ActiveStreamDetached(
            cached_stream_count=len(self.runtime.stream_cache),
        )
        self.events.emit(StreamDetached(cached_stream_count=result.cached_stream_count))
        return result

    def evict_stream_cache(self) -> StreamCacheEvictedResult | Failed:
        """Evict cached stream runtimes for a recording/session transition."""
        dependency_plan = plan_save_runtime_dependencies(
            document=self.controller.document,
            data_context=self.data_context,
            runtime=self.runtime,
        )
        protected = dependency_plan.eviction_protected
        if protected:
            return Failed(
                "Cannot evict loaded stream runtimes while edited alignments "
                "depend on them. Save or discard edits first: "
                f"{describe_save_runtime_dependencies(protected)}."
            )

        evicted_stream_count = len(self.runtime.stream_cache)
        self.runtime.clear_stream_cache()
        self.display_state.reset_for_active_stream()
        result = StreamCacheEvictedResult(evicted_stream_count=evicted_stream_count)
        self.events.emit(
            StreamCacheEvicted(
                evicted_stream_count=result.evicted_stream_count,
            )
        )
        return result

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
        self.events.emit(
            StreamActivated(
                source="cached",
                stream_key=stream_key,
                shank_idx=shank_idx,
                active_key=self.controller.document.selected_alignment_key,
                preserve_plot_selection=True,
            )
        )
        return CachedEphysDataActivated(
            stream_runtime=stream_runtime,
            shank_idx=shank_idx,
            probe=probe,
        )

    def begin_preload_data(
        self,
        *,
        recording_id: str,
        probe_name: str,
        target_shank: int = 0,
    ) -> LoadDataFreshPrepared | LoadDataPreloadSkipped | Failed:
        """Prepare an inactive background preload for one probe selection."""
        stream_key = self._stream_key_for_selection(recording_id, probe_name)
        if stream_key is None:
            return Failed(
                f"Preload target could not resolve stream for {recording_id}/{probe_name}"
            )
        if self.runtime.cached_stream(stream_key) is not None:
            return LoadDataPreloadSkipped(
                stream_key=stream_key,
                shank_idx=target_shank,
                reason="target stream is already cached",
            )

        target = self._fresh_load_target_for_probe(
            recording_id=recording_id,
            probe_name=probe_name,
            stream_key=stream_key,
            shank_idx=target_shank,
        )
        if isinstance(target, Failed):
            return target

        return LoadDataFreshPrepared(
            stream_key=stream_key,
            shank_idx=target.shank_idx,
            preserve_plot_selection=True,
            target=target,
        )

    def start_preload_data(
        self,
        prepared: LoadDataFreshPrepared,
        *,
        cancel_token: LoadDataCancelToken | None = None,
    ) -> FreshLoadExecution:
        """Start a tracked background preload execution."""
        execution, _cancelled = self.preload_lifecycle.start(
            prepared,
            cancel_token=cancel_token,
        )
        return execution

    def cancel_active_preload(
        self,
        reason: str,
    ) -> LoadDataJobCancelled | None:
        """Cancel the active background preload execution, if any."""
        cancelled = self.preload_lifecycle.cancel_active(reason)
        if cancelled is None:
            return None
        return LoadDataJobCancelled(
            target=cancelled.execution.prepared.target,
            reason=reason,
        )

    def preload_job_invocation(
        self,
        execution: FreshLoadExecution,
    ) -> FreshLoadJobInvocation | LoadDataJobCancelled:
        """Return a runnable job invocation for an active preload execution."""
        prepared = execution.prepared
        cancel_token = self.preload_lifecycle.cancel_token_for(execution)
        if cancel_token is None:
            return LoadDataJobCancelled(
                target=prepared.target,
                reason="Preload request is no longer active.",
            )
        return FreshLoadJobInvocation(
            execution=execution,
            request=LoadDataJobRequest(prepared.target, load_id=execution.load_id),
            cancel_token=cancel_token,
        )

    def cache_started_preload_data(
        self,
        execution: FreshLoadExecution,
        job_result: LoadDataJobCompleted | LoadDataJobCancelled | Failed,
    ) -> (
        FreshEphysDataLoaded
        | LoadDataStaleResultIgnored
        | LoadDataJobCancelled
        | Failed
    ):
        """Cache a completed preload only if its request is still useful."""
        if isinstance(job_result, Failed | LoadDataJobCancelled):
            self.preload_lifecycle.finish(execution)
            return job_result

        try:
            if not self.preload_lifecycle.is_active(execution):
                return LoadDataStaleResultIgnored(
                    load_id=execution.load_id,
                    stream_key=execution.prepared.stream_key,
                    shank_idx=execution.prepared.shank_idx,
                    reason="Preload request is no longer active.",
                )

            stale = self._stale_preload_result_reason(execution.prepared, job_result)
            if stale is not None:
                return LoadDataStaleResultIgnored(
                    load_id=execution.load_id,
                    stream_key=execution.prepared.stream_key,
                    shank_idx=execution.prepared.shank_idx,
                    reason=stale,
                )

            return self.cache_completed_fresh_load_data(job_result)
        finally:
            self.preload_lifecycle.finish(execution)

    def attach_warmed_plot_payload_cache(
        self,
        warmed: PlotPayloadCacheWarmed,
    ) -> Ok | LoadDataStaleResultIgnored | Failed:
        """Attach a warmed plot cache to an inactive cached stream, if still useful."""
        stream_runtime = self.runtime.cached_stream(warmed.stream_key)
        if stream_runtime is None:
            return LoadDataStaleResultIgnored(
                load_id=None,
                stream_key=warmed.stream_key,
                shank_idx=warmed.shank_idx,
                reason="Warmed stream is no longer cached.",
            )
        if stream_runtime.stream is not warmed.stream:
            return LoadDataStaleResultIgnored(
                load_id=None,
                stream_key=warmed.stream_key,
                shank_idx=warmed.shank_idx,
                reason="Warmed stream does not match cached stream.",
            )
        if self.runtime.active_stream_runtime is stream_runtime:
            return LoadDataStaleResultIgnored(
                load_id=None,
                stream_key=warmed.stream_key,
                shank_idx=warmed.shank_idx,
                reason="Warmed stream is active; active view owns its plot cache.",
            )

        shank_runtime = stream_runtime.shank_runtime_by_idx.get(warmed.shank_idx)
        if shank_runtime is None:
            return Failed(
                f"Warmed shank runtime not found: {warmed.stream_key} "
                f"shank {warmed.shank_idx}"
            )
        if shank_runtime.plot_payload_cache is not None:
            return LoadDataStaleResultIgnored(
                load_id=None,
                stream_key=warmed.stream_key,
                shank_idx=warmed.shank_idx,
                reason="Warmed shank already has a plot cache.",
            )

        shank_runtime.plot_payload_cache = warmed.payload_cache
        return Ok()

    def run_plot_payload_warmup(
        self,
        request: PlotPayloadWarmupRequest,
    ) -> PlotPayloadCacheWarmed | Failed:
        """Run one Qt-free plot payload warmup job."""
        return PlotPayloadWarmupJob(self.plot_payload_cache_factory).run(request)

    def _cache_loaded_probe_data(
        self,
        loaded: LoadedEphysSelection,
        *,
        shank_idx: int,
        activate: bool,
    ) -> FreshEphysDataLoaded:
        """Insert loaded ephys data into runtime cache and mark document loaded."""
        stream_runtime = self.runtime.cache_loaded_stream_data(
            loaded.stream,
            self.plot_payload_cache_factory,
            shank_idx=shank_idx,
            activate=activate,
        )
        if activate:
            self.controller.finish_load_data(shank_idx)
        return FreshEphysDataLoaded(
            stream_runtime=stream_runtime,
            shank_idx=shank_idx,
        )

    def _fresh_load_target_for_selection(
        self,
        *,
        recording_id: str,
        probe_name: str,
        stream_key: StreamKey | None,
        shank_idx: int,
    ) -> LoadDataJobTarget | Failed:
        mouse_root = self.data_context.mouse_root
        if mouse_root is None:
            return Failed("Fresh load requires a loaded mouse root.")

        probe = self.data_context.probe_info
        if probe is None:
            return Failed("Fresh load requires selected probe metadata.")
        if probe.recording_id != recording_id or probe.probe_name != probe_name:
            return Failed(
                "Fresh load target does not match selected probe metadata: "
                f"{recording_id}/{probe_name}"
            )

        channel_table = self.data_context.channel_table
        if channel_table is None:
            return Failed("Fresh load requires selected channel metadata.")

        resolved_stream_key = stream_key or (probe.recording_id, probe.ephys_collection)
        return LoadDataJobTarget(
            recording_id=recording_id,
            probe_name=probe_name,
            stream_key=resolved_stream_key,
            shank_idx=shank_idx,
            mouse_root=mouse_root,
            probe_info=probe,
            channel_table=channel_table,
        )

    def _fresh_load_target_for_probe(
        self,
        *,
        recording_id: str,
        probe_name: str,
        stream_key: StreamKey | None,
        shank_idx: int,
    ) -> LoadDataJobTarget | Failed:
        mouse_root = self.data_context.mouse_root
        if mouse_root is None:
            return Failed("Preload requires a loaded mouse root.")

        try:
            probe = mouse_root.get_probe(recording_id, probe_name)
            channel_table = (
                self.metadata_commands.ephys_data_service.load_channel_table(probe)
            )
        except Exception as exc:
            return Failed(f"Failed to prepare preload target {probe_name}: {exc}")

        resolved_stream_key = stream_key or (probe.recording_id, probe.ephys_collection)
        return LoadDataJobTarget(
            recording_id=recording_id,
            probe_name=probe_name,
            stream_key=resolved_stream_key,
            shank_idx=shank_idx,
            mouse_root=mouse_root,
            probe_info=probe,
            channel_table=channel_table,
        )

    def _stale_fresh_load_reason(
        self,
        prepared: LoadDataFreshPrepared,
        job_result: LoadDataJobCompleted,
    ) -> str | None:
        if not prepared.target.same_product_identity(job_result.target):
            return "Loaded data target does not match the prepared load target."

        current_stream_key = self._stream_key_for_selection(
            prepared.target.recording_id,
            prepared.target.probe_name,
        )
        if current_stream_key != prepared.target.stream_key:
            return "Loaded data target is stale; selected stream changed."

        current_probe = self.data_context.probe_info
        if current_probe is None:
            return "Loaded data target is stale; no probe is currently selected."
        if (
            current_probe.recording_id != prepared.target.recording_id
            or current_probe.probe_name != prepared.target.probe_name
            or current_probe.ephys_collection != prepared.target.stream_key[1]
        ):
            return "Loaded data target is stale; selected probe changed."

        current_mouse_root = self.data_context.mouse_root
        if (
            current_mouse_root is None
            or current_mouse_root.root != prepared.target.mouse_root.root
        ):
            return "Loaded data target is stale; selected mouse root changed."

        return None

    def _stale_promoted_preload_result_reason(
        self,
        preload_execution: FreshLoadExecution,
        foreground_execution: FreshLoadExecution,
        job_result: LoadDataJobCompleted,
    ) -> str | None:
        preload_target = preload_execution.prepared.target
        foreground_target = foreground_execution.prepared.target
        if not preload_target.same_product_identity(foreground_target):
            return "Promoted preload target does not match the foreground load target."
        if not foreground_target.same_product_identity(job_result.target):
            return "Promoted preload result does not match the foreground load target."
        return None

    def _stale_preload_result_reason(
        self,
        prepared: LoadDataFreshPrepared,
        job_result: LoadDataJobCompleted,
    ) -> str | None:
        if not prepared.target.same_identity(job_result.target):
            return "Loaded preload target does not match the prepared preload target."

        current_mouse_root = self.data_context.mouse_root
        if (
            current_mouse_root is None
            or current_mouse_root.root != prepared.target.mouse_root.root
        ):
            return "Loaded preload target is stale; selected mouse root changed."

        if self.runtime.cached_stream(prepared.target.stream_key) is not None:
            return "Loaded preload target is already cached or active."

        return None

    def _emit_histology_report(
        self,
        prepared: LoadDataFreshPrepared,
        histology: HistologyLoadResult,
        *,
        load_id: int | None = None,
    ) -> None:
        """Emit a semantic event for non-fatal histology load availability."""
        if isinstance(histology, HistologyDataAlreadyLoaded):
            event = HistologyLoadReported(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                status="already_loaded",
                load_id=load_id,
            )
        elif isinstance(histology, HistologyDataLoaded):
            event = HistologyLoadReported(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                status="loaded",
                load_id=load_id,
            )
        elif isinstance(histology, HistologyDataUnavailable):
            event = HistologyLoadReported(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                status="unavailable",
                message=histology.message,
                load_id=load_id,
            )
        else:
            return
        self.events.emit(event)

    def _emit_cancelled_execution(
        self,
        cancelled: CancelledFreshLoadExecution,
    ) -> None:
        execution = cancelled.execution
        self.events.emit(
            LoadDataCancelled(
                stream_key=execution.prepared.stream_key,
                shank_idx=execution.prepared.shank_idx,
                reason=cancelled.reason,
                load_id=execution.load_id,
            )
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
