"""App-level load/cache command handlers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from ephys_alignment_gui.application.commands.metadata_selection import (
    MetadataSelectionCommandHandler,
)
from ephys_alignment_gui.application.results import (
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
from ephys_alignment_gui.application.workflow import Failed, PolicyResult
from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.core.alignment_events import (
    FreshLoadCompleted,
    HistologyLoadReported,
    LoadDataCancelled,
    LoadDataFailed,
    LoadDataProgressed,
    StreamActivated,
)
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.reference_line_capture import (
    REFERENCE_LINES_NOT_PROVIDED,
    ReferenceLineCapture,
    capture_active_reference_lines_if_provided,
)
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.io.ephys_stream_loader import LoadedEphysSelection
from ephys_alignment_gui.io.load_data_job import (
    LoadDataCancelToken,
    LoadDataJob,
    LoadDataJobCancelled,
    LoadDataJobCompleted,
    LoadDataJobRequest,
    LoadDataProgressCallback,
)
from ephys_alignment_gui.io.load_data_target import LoadDataJobTarget
from ephys_alignment_gui.plotting.payload_cache_factory import (
    EphysPlotPayloadCacheFactory,
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

        prepared = self.prepare_fresh_ephys_load(stream_key)
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
        job_result = self.run_fresh_load_data(
            prepared,
            progress=progress,
            cancel_token=cancel_token,
        )
        if isinstance(job_result, Failed | LoadDataJobCancelled):
            return job_result
        return self.activate_completed_fresh_load_data(prepared, job_result)

    def run_fresh_load_data(
        self,
        prepared: LoadDataFreshPrepared,
        *,
        progress: LoadDataProgressCallback | None = None,
        cancel_token: LoadDataCancelToken | None = None,
    ) -> LoadDataJobCompleted | LoadDataJobCancelled | Failed:
        """Run the fresh load job without activating the loaded stream."""
        job_result = self.load_data_job.run(
            LoadDataJobRequest(prepared.target),
            progress=self._fresh_load_progress_callback(prepared, progress),
            cancel_token=cancel_token,
        )
        if isinstance(job_result, Failed):
            self.events.emit(
                LoadDataFailed(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    message=job_result.message,
                )
            )
        elif isinstance(job_result, LoadDataJobCancelled):
            self.events.emit(
                LoadDataCancelled(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    reason=job_result.reason,
                )
            )
        else:
            self.events.emit(
                FreshLoadCompleted(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    warning_messages=tuple(
                        warning.message for warning in job_result.warnings
                    ),
                )
            )
        return job_result

    def activate_completed_fresh_load_data(
        self,
        prepared: LoadDataFreshPrepared,
        job_result: LoadDataJobCompleted,
    ) -> LoadDataFreshCompleted | Failed:
        """Cache/activate completed fresh-load data if its target is still current."""
        stale = self._stale_fresh_load_reason(prepared, job_result)
        if stale is not None:
            self.events.emit(
                LoadDataFailed(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    message=stale,
                )
            )
            return Failed(stale)

        self.histology_runtime_loader.activate_result(
            job_result.histology,
            mouse_root=prepared.target.mouse_root,
        )
        self._emit_histology_report(prepared, job_result.histology)
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

    def _stale_fresh_load_reason(
        self,
        prepared: LoadDataFreshPrepared,
        job_result: LoadDataJobCompleted,
    ) -> str | None:
        if not prepared.target.same_identity(job_result.target):
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

    def _fresh_load_progress_callback(
        self,
        prepared: LoadDataFreshPrepared,
        progress: LoadDataProgressCallback | None,
    ) -> LoadDataProgressCallback:
        """Return a progress callback that also emits app-level load events."""

        def _emit_progress(event) -> None:
            self.events.emit(
                LoadDataProgressed(
                    stream_key=prepared.stream_key,
                    shank_idx=prepared.shank_idx,
                    phase=event.phase,
                    status=event.status,
                    message=event.message,
                )
            )
            if progress is not None:
                progress(event)

        return _emit_progress

    def _emit_histology_report(
        self,
        prepared: LoadDataFreshPrepared,
        histology: HistologyLoadResult,
    ) -> None:
        """Emit a semantic event for non-fatal histology load availability."""
        if isinstance(histology, HistologyDataAlreadyLoaded):
            event = HistologyLoadReported(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                status="already_loaded",
            )
        elif isinstance(histology, HistologyDataLoaded):
            event = HistologyLoadReported(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                status="loaded",
            )
        elif isinstance(histology, HistologyDataUnavailable):
            event = HistologyLoadReported(
                stream_key=prepared.stream_key,
                shank_idx=prepared.shank_idx,
                status="unavailable",
                message=histology.message,
            )
        else:
            return
        self.events.emit(event)

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
