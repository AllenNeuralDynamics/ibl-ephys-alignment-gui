"""Desktop coordination shell for loading ephys/histology data."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

from ephys_alignment_gui.application.results import (
    FreshEphysDataLoaded,
    FreshLoadExecution,
    LoadDataAlreadyActiveResult,
    LoadDataCachedActivated,
    LoadDataFreshCompleted,
    LoadDataFreshPrepared,
    LoadDataFreshRequiredResult,
    LoadDataPreloadSkipped,
    LoadDataStaleResultIgnored,
)
from ephys_alignment_gui.core.alignment_events import (
    HistologyLoadReported,
    LoadDataCancelled,
    LoadDataFailed,
    LoadDataProgressed,
    StreamActivated,
)
from ephys_alignment_gui.core.event_bus import EventSubscription
from ephys_alignment_gui.core.timing import TimingSession, start_timing
from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.desktop.workers.load_data_runner import (
    FreshLoadJobResult,
    FreshLoadRunner,
    QtFreshLoadRunner,
)
from ephys_alignment_gui.desktop.workers.plot_payload_warmup_runner import (
    PlotPayloadWarmupResult,
    PlotPayloadWarmupRunner,
    QtPlotPayloadWarmupRunner,
)
from ephys_alignment_gui.io.load_data_job import (
    LoadDataJobCancelled,
    LoadDataJobCompleted,
    LoadDataJobProgress,
)
from ephys_alignment_gui.plotting.payload_warmup import (
    PlotPayloadCacheWarmed,
    PlotPayloadWarmupRequest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopLoadDataCallbacks:
    """Desktop callbacks used by the load-data coordinator."""

    reference_line_positions: Callable[[], Any]
    prepare_for_fresh_stream_load: Callable[[], None]
    render_loaded_shank: Callable[[int, bool | None], None]
    image_raster_request: Callable[[], Any]
    clear_empty_state: Callable[[], None]
    busy_context: Callable[..., AbstractContextManager[Any]]


@dataclass(frozen=True)
class _PromotedPreload:
    preload_execution: FreshLoadExecution
    foreground_execution: FreshLoadExecution
    session_name: str
    probe_name: str


@dataclass
class DesktopLoadDataCoordinator:
    """Coordinate desktop behavior for cached and fresh data loads."""

    app: Any
    selection_view: Any
    callbacks: DesktopLoadDataCallbacks
    load_runner: FreshLoadRunner = field(default_factory=QtFreshLoadRunner)
    preload_runner: FreshLoadRunner = field(default_factory=QtFreshLoadRunner)
    plot_warmup_runner: PlotPayloadWarmupRunner = field(
        default_factory=QtPlotPayloadWarmupRunner
    )
    _active_load_context: Any | None = field(default=None, init=False, repr=False)
    _active_load_context_manager: Any | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _active_load_id: int | None = field(default=None, init=False, repr=False)
    _active_preload_id: int | None = field(default=None, init=False, repr=False)
    _active_preload_execution: FreshLoadExecution | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _promoted_preload: _PromotedPreload | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _active_switch_timer: TimingSession | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _active_preload_timer: TimingSession | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _active_plot_warmup_timer: TimingSession | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def connect_load_events(self) -> list[EventSubscription]:
        """Subscribe desktop load coordination to semantic load events."""
        return [
            self.app.events.subscribe(LoadDataProgressed, self.on_load_data_progressed),
            self.app.events.subscribe(LoadDataFailed, self.on_load_data_failed),
            self.app.events.subscribe(LoadDataCancelled, self.on_load_data_cancelled),
            self.app.events.subscribe(
                HistologyLoadReported,
                self.on_histology_load_reported,
            ),
            self.app.events.subscribe(StreamActivated, self.on_stream_activated),
        ]

    def connect_stream_events(self) -> list[EventSubscription]:
        """Backward-compatible alias for older workbench tests/callers."""
        return self.connect_load_events()

    def on_load_data_progressed(self, event: LoadDataProgressed) -> None:
        """Update desktop progress coordination for an active load context."""
        if not self._event_matches_active_load(event.load_id):
            return
        if self._active_load_context is not None:
            self._active_load_context.update_message(event.message)

    def on_load_data_failed(self, event: LoadDataFailed) -> None:
        """Log fresh-load failures reported by the app layer."""
        if not self._event_matches_active_load(event.load_id):
            return
        logger.error(event.message)

    def on_load_data_cancelled(self, event: LoadDataCancelled) -> None:
        """Log fresh-load cancellation reported by the app layer."""
        if not self._event_matches_active_load(event.load_id):
            return
        logger.info("Load cancelled: %s", event.reason)

    def on_histology_load_reported(self, event: HistologyLoadReported) -> None:
        """Log non-fatal histology availability for a fresh load."""
        if not self._event_matches_active_load(event.load_id):
            return
        if event.status in {"already_loaded", "loaded"}:
            logger.info("Atlas and histology loaded successfully")
        elif event.status == "unavailable" and event.message is not None:
            logger.error(event.message)

    def on_stream_activated(self, event: StreamActivated) -> None:
        """Render desktop state for an activated stream/shank."""
        if not self._event_matches_active_load(event.load_id):
            return
        self._present_activated_stream(event)

    def load_heavy_data(
        self,
        *,
        preserve_plot_selection: bool | None = None,
    ) -> bool:
        """Load or activate the selected stream/shank for desktop display."""
        callbacks = self.callbacks
        if self._foreground_load_is_running():
            logger.info("Load request ignored because a foreground load is active")
            self.cancel_active_load("superseded by a newer load request")
            return False

        target_shank = self.app.queries.workspace.active_shank_selection().shank_idx
        session_name = self.selection_view.current_session()
        probe_name = self.selection_view.current_probe()
        timer = start_timing(
            "load_data_switch",
            recording_id=session_name,
            probe_name=probe_name,
            shank_idx=target_shank,
        )
        self._active_switch_timer = timer
        with timer.activate():
            with timer.step("begin_load_data"):
                begin_kwargs = {
                    "recording_id": session_name,
                    "probe_name": probe_name,
                    "target_shank": target_shank,
                    "outgoing_reference_lines": callbacks.reference_line_positions(),
                }
                if preserve_plot_selection is not None:
                    begin_kwargs["preserve_plot_selection"] = preserve_plot_selection
                begin_result = self.app.commands.load.begin_load_data(
                    **begin_kwargs,
                )
            if isinstance(begin_result, Failed):
                logger.error(begin_result.message)
                self._finish_switch_timer("failed", message=begin_result.message)
                return False
            if isinstance(begin_result, LoadDataAlreadyActiveResult):
                self._finish_switch_timer("already_active")
                self.schedule_next_probe_preload(session_name, probe_name)
                return True
            if isinstance(begin_result, LoadDataCachedActivated):
                self.cancel_active_preload("foreground load requested")
                logger.info("Activated cached stream %s", begin_result.stream_key)
                self._finish_switch_timer("cached")
                self.schedule_next_probe_preload(session_name, probe_name)
                return True
            assert isinstance(begin_result, LoadDataFreshPrepared)
            promoted = self._promote_matching_active_preload(
                begin_result,
                session_name=session_name,
                probe_name=probe_name,
                timer=timer,
            )
            if promoted is not None:
                return True

            self.cancel_active_preload("foreground load requested")
            with timer.step("start_fresh_load"):
                execution = self.app.commands.load.start_fresh_load_data(begin_result)
                invocation = self.app.commands.load.fresh_load_job_invocation(execution)
            if isinstance(invocation, LoadDataJobCancelled):
                logger.info("Load cancelled: %s", invocation.reason)
                self._finish_switch_timer("cancelled", reason=invocation.reason)
                return False

            with timer.step("open_load_context"):
                self._open_load_context(
                    "Loading heavy data...",
                    "Data loaded successfully",
                    disable_widgets=self.selection_view.selection_widgets(),
                )
            logger.info("=== Starting heavy data load ===")
            with timer.step("prepare_for_fresh_stream_load"):
                callbacks.prepare_for_fresh_stream_load()
            self._active_load_id = execution.load_id
            logger.info(
                "Loading probe data, active shank index %s",
                begin_result.shank_idx,
            )
            try:
                with timer.step("start_background_load_worker"):
                    self.load_runner.start(
                        execution=execution,
                        invocation=invocation,
                        run_job=self.app.commands.load.run_fresh_load_job,
                        on_progress=self._on_load_worker_progress,
                        on_finished=self._on_load_worker_finished,
                    )
            except Exception as exc:
                self.app.commands.load.cancel_active_fresh_load(
                    "failed to start background load"
                )
                self._close_load_context(exc)
                self._finish_switch_timer("failed", message=str(exc))
                logger.exception("Failed to start background load")
                return False
        return True

    def cancel_active_load(self, reason: str) -> bool:
        """Request cancellation for any active foreground load."""
        if (
            self._active_load_id is None
            and not self.load_runner.is_running
            and not self._promoted_preload_is_running()
        ):
            return False

        self.app.commands.load.cancel_active_fresh_load(reason)
        if self._promoted_preload_is_running():
            self.app.commands.load.cancel_active_preload(reason)
            self.preload_runner.cancel(reason)
        elif self.load_runner.is_running:
            self.load_runner.cancel(reason)
        return True

    def shutdown_active_load(
        self,
        reason: str = "application closing",
        *,
        timeout_ms: int = 5000,
    ) -> bool:
        """Cancel and settle any active foreground load before desktop teardown."""
        if not self.shutdown_active_preload(reason, timeout_ms=timeout_ms):
            return False

        if self._active_load_id is None and not self.load_runner.is_running:
            return True

        self.app.commands.load.cancel_active_fresh_load(reason)
        stopped = self.load_runner.shutdown(reason, timeout_ms=timeout_ms)
        if stopped:
            self._close_load_context(RuntimeError(f"Load cancelled: {reason}"))
            self._finish_switch_timer("cancelled", reason=reason)
        return stopped

    def schedule_next_probe_preload(
        self,
        recording_id: str,
        probe_name: str,
        *,
        allow_foreground_runner_settling: bool = False,
    ) -> bool:
        """Preload the next unloaded probe in the same recording, if one exists."""
        next_probe = self.app.queries.workspace.next_unloaded_probe_in_recording(
            recording_id,
            probe_name,
        )
        if not next_probe:
            return False
        return self.start_probe_preload(
            recording_id=recording_id,
            probe_name=next_probe,
            allow_foreground_runner_settling=allow_foreground_runner_settling,
        )

    def start_probe_preload(
        self,
        *,
        recording_id: str,
        probe_name: str,
        allow_foreground_runner_settling: bool = False,
    ) -> bool:
        """Start a background preload that only populates the inactive cache."""
        if self.load_runner.is_running and not allow_foreground_runner_settling:
            return False
        if self.preload_runner.is_running:
            logger.debug(
                "Skipping preload for %s/%s because another preload is still running",
                recording_id,
                probe_name,
            )
            return False

        timer = start_timing(
            "preload_probe",
            recording_id=recording_id,
            probe_name=probe_name,
            shank_idx=0,
        )
        with timer.activate():
            with timer.step("begin_preload_data"):
                prepared = self.app.commands.load.begin_preload_data(
                    recording_id=recording_id,
                    probe_name=probe_name,
                    target_shank=0,
                )
            if isinstance(prepared, Failed):
                logger.warning(prepared.message)
                self._finish_preload_timer(timer, "failed", message=prepared.message)
                return False
            if isinstance(prepared, LoadDataPreloadSkipped):
                logger.debug("Skipping preload for %s: %s", probe_name, prepared.reason)
                self._finish_preload_timer(timer, "skipped", reason=prepared.reason)
                return False

            assert isinstance(prepared, LoadDataFreshPrepared)
            with timer.step("start_preload"):
                execution = self.app.commands.load.start_preload_data(prepared)
                invocation = self.app.commands.load.preload_job_invocation(execution)
            if isinstance(invocation, LoadDataJobCancelled):
                logger.info("Preload cancelled: %s", invocation.reason)
                self._finish_preload_timer(
                    timer,
                    "cancelled",
                    reason=invocation.reason,
                )
                return False

            self._active_preload_id = execution.load_id
            self._active_preload_execution = execution
            self._active_preload_timer = timer
            logger.info("Preloading next probe %s/%s", recording_id, probe_name)
            try:
                with timer.step("start_background_preload_worker"):
                    self.preload_runner.start(
                        execution=execution,
                        invocation=invocation,
                        run_job=self.app.commands.load.run_fresh_load_job,
                        on_progress=self._on_preload_worker_progress,
                        on_finished=self._on_preload_worker_finished,
                    )
            except Exception as exc:
                self.app.commands.load.cancel_active_preload(
                    "failed to start background preload"
                )
                self._active_preload_id = None
                self._active_preload_execution = None
                self._finish_preload_timer(timer, "failed", message=str(exc))
                logger.exception("Failed to start background preload")
                return False
        return True

    def cancel_active_preload(self, reason: str) -> bool:
        """Request cancellation for any active background preload."""
        if self._active_preload_id is None and not self.preload_runner.is_running:
            return False
        self.app.commands.load.cancel_active_preload(reason)
        if self.preload_runner.is_running:
            self.preload_runner.cancel(reason)
        self._finish_preload_timer(
            self._active_preload_timer,
            "cancelled",
            reason=reason,
        )
        self._active_preload_id = None
        self._active_preload_execution = None
        if self._promoted_preload is not None:
            self._promoted_preload = None
        return True

    def shutdown_active_preload(
        self,
        reason: str = "application closing",
        *,
        timeout_ms: int = 5000,
    ) -> bool:
        """Cancel and settle any active background preload."""
        if not self.shutdown_plot_payload_warmup(reason, timeout_ms=timeout_ms):
            return False
        if self._active_preload_id is None and not self.preload_runner.is_running:
            return True
        self.app.commands.load.cancel_active_preload(reason)
        stopped = self.preload_runner.shutdown(reason, timeout_ms=timeout_ms)
        if stopped:
            self._finish_preload_timer(
                self._active_preload_timer,
                "cancelled",
                reason=reason,
            )
            self._active_preload_id = None
            self._active_preload_execution = None
            self._promoted_preload = None
        return stopped

    def shutdown_plot_payload_warmup(
        self,
        reason: str = "application closing",
        *,
        timeout_ms: int = 5000,
    ) -> bool:
        """Settle any active plot-cache warmup before desktop teardown."""
        if not self.plot_warmup_runner.is_running:
            return True
        stopped = self.plot_warmup_runner.shutdown(reason, timeout_ms=timeout_ms)
        if stopped:
            self._finish_plot_warmup_timer("cancelled", reason=reason)
        return stopped

    def _on_load_worker_progress(
        self,
        execution: FreshLoadExecution,
        event: LoadDataJobProgress,
    ) -> None:
        """Publish worker progress from the GUI thread."""
        if self._active_switch_timer is not None:
            self._active_switch_timer.mark(
                "load_worker_progress",
                load_id=execution.load_id,
                phase=event.phase,
                status=event.status,
                message=event.message,
            )
        self.app.commands.load.publish_fresh_load_progress(execution, event)

    def _on_load_worker_finished(
        self,
        execution: FreshLoadExecution,
        job_result: FreshLoadJobResult,
    ) -> None:
        """Publish worker completion and activate successful fresh-load data."""
        timer = self._active_switch_timer
        context = timer.activate() if timer is not None else nullcontext()
        with context:
            step = (
                timer.step("publish_worker_result")
                if timer is not None
                else nullcontext()
            )
            with step:
                published = (
                    self.app.commands.load.publish_started_fresh_load_job_result(
                        execution,
                        job_result,
                    )
                )
        self._activate_published_fresh_load_result(
            execution,
            published,
        )

    def _activate_published_fresh_load_result(
        self,
        execution: FreshLoadExecution,
        published: Any,
    ) -> None:
        """Activate a published fresh-load result or close the load context."""
        timer = self._active_switch_timer
        if isinstance(published, Failed):
            logger.error(published.message)
            self._close_load_context(RuntimeError(published.message))
            self._finish_switch_timer("failed", message=published.message)
            return
        if isinstance(published, LoadDataJobCancelled):
            logger.info("Load cancelled: %s", published.reason)
            self._close_load_context(
                RuntimeError(f"Load cancelled: {published.reason}")
            )
            self._finish_switch_timer("cancelled", reason=published.reason)
            return
        if isinstance(published, LoadDataStaleResultIgnored):
            logger.info("Load result ignored: %s", published.reason)
            self._close_load_context(RuntimeError(published.reason))
            self._finish_switch_timer("stale", reason=published.reason)
            return
        if not isinstance(published, LoadDataJobCompleted):
            self._close_load_context(RuntimeError("Fresh load returned no result"))
            self._finish_switch_timer(
                "failed",
                message="Fresh load returned no result",
            )
            return

        context = timer.activate() if timer is not None else nullcontext()
        with context:
            if self._active_load_context is not None:
                self._active_load_context.update_message("Setting up visualization...")
            step = (
                timer.step("activate_and_render")
                if timer is not None
                else nullcontext()
            )
            with step:
                completed = self.app.commands.load.activate_started_fresh_load_data(
                    execution,
                    published,
                )
        if isinstance(completed, Failed):
            logger.error(completed.message)
            self._close_load_context(RuntimeError(completed.message))
            self._finish_switch_timer("failed", message=completed.message)
            return
        if isinstance(completed, LoadDataStaleResultIgnored):
            logger.info("Load result ignored: %s", completed.reason)
            self._close_load_context(RuntimeError(completed.reason))
            self._finish_switch_timer("stale", reason=completed.reason)
            return
        assert isinstance(completed, LoadDataFreshCompleted)
        stream_runtime = completed.ephys.stream_runtime
        logger.info("Loaded ephys data from %s", stream_runtime.stream.ephys_dir)
        logger.info("=== Heavy data load complete ===")
        self._close_load_context()
        self._finish_switch_timer("fresh")
        self.schedule_next_probe_preload(
            completed.target.recording_id,
            completed.target.probe_name,
            allow_foreground_runner_settling=True,
        )

    def _on_preload_worker_progress(
        self,
        execution: FreshLoadExecution,
        event: LoadDataJobProgress,
    ) -> None:
        """Log preload progress without touching active desktop rendering."""
        if execution.load_id != self._active_preload_id:
            return
        promoted = self._promoted_preload
        if (
            promoted is not None
            and promoted.preload_execution.load_id == execution.load_id
        ):
            self._on_load_worker_progress(
                promoted.foreground_execution,
                replace(event, load_id=promoted.foreground_execution.load_id),
            )
            return
        if self._active_preload_timer is not None:
            self._active_preload_timer.mark(
                "preload_worker_progress",
                load_id=execution.load_id,
                phase=event.phase,
                status=event.status,
                message=event.message,
            )
        logger.debug("Preload progress: %s", event.message)

    def _on_preload_worker_finished(
        self,
        execution: FreshLoadExecution,
        job_result: FreshLoadJobResult,
    ) -> None:
        """Cache successful preload results without activating the stream."""
        if execution.load_id != self._active_preload_id:
            return
        promoted = self._promoted_preload
        if (
            promoted is not None
            and promoted.preload_execution.load_id == execution.load_id
        ):
            self._on_promoted_preload_worker_finished(
                promoted,
                execution,
                job_result,
            )
            return

        timer = self._active_preload_timer
        context = timer.activate() if timer is not None else nullcontext()
        try:
            with context:
                step = (
                    timer.step("cache_preload_result")
                    if timer is not None
                    else nullcontext()
                )
                with step:
                    cached = self.app.commands.load.cache_started_preload_data(
                        execution,
                        job_result,
                    )
            if isinstance(cached, Failed):
                logger.warning(cached.message)
                self._finish_preload_timer(timer, "failed", message=cached.message)
                return
            if isinstance(cached, LoadDataJobCancelled):
                logger.info("Preload cancelled: %s", cached.reason)
                self._finish_preload_timer(timer, "cancelled", reason=cached.reason)
                return
            if isinstance(cached, LoadDataStaleResultIgnored):
                logger.info("Preload result ignored: %s", cached.reason)
                self._finish_preload_timer(timer, "stale", reason=cached.reason)
                return
            assert isinstance(cached, FreshEphysDataLoaded)
            logger.info("Preloaded stream %s", cached.stream_runtime.stream_key)
            self.start_plot_payload_warmup(cached)
            self._finish_preload_timer(
                timer,
                "completed",
                stream_key=cached.stream_runtime.stream_key,
            )
        finally:
            if execution.load_id == self._active_preload_id:
                self._active_preload_id = None
                self._active_preload_execution = None
                self._active_preload_timer = None

    def _on_promoted_preload_worker_finished(
        self,
        promoted: _PromotedPreload,
        execution: FreshLoadExecution,
        job_result: FreshLoadJobResult,
    ) -> None:
        """Activate a preload result that was promoted to foreground ownership."""
        timer = self._active_switch_timer
        context = timer.activate() if timer is not None else nullcontext()
        try:
            with context:
                step = (
                    timer.step("publish_promoted_preload_result")
                    if timer is not None
                    else nullcontext()
                )
                with step:
                    published = (
                        self.app.commands.load.publish_promoted_preload_job_result(
                            execution,
                            promoted.foreground_execution,
                            job_result,
                        )
                    )
            self._activate_published_fresh_load_result(
                promoted.foreground_execution,
                published,
            )
        finally:
            if execution.load_id == self._active_preload_id:
                self._active_preload_id = None
                self._active_preload_execution = None
                self._active_preload_timer = None
            if self._promoted_preload is promoted:
                self._promoted_preload = None

    def start_plot_payload_warmup(self, cached: FreshEphysDataLoaded) -> bool:
        """Warm an inactive preloaded stream's plot payload cache."""
        if self.plot_warmup_runner.is_running:
            logger.debug(
                "Skipping plot payload warmup for %s because another warmup is running",
                cached.stream_runtime.stream_key,
            )
            return False

        request = PlotPayloadWarmupRequest(
            stream_key=cached.stream_runtime.stream_key,
            stream=cached.stream_runtime.stream,
            shank_idx=cached.shank_idx,
            unit_filter=self.app.queries.ephys.active_unit_filter(),
            raster_request=self.callbacks.image_raster_request(),
        )
        timer = start_timing(
            "plot_payload_warmup",
            recording_id=request.stream_key[0],
            probe_name=request.stream_key[1],
            shank_idx=request.shank_idx,
            unit_filter=request.unit_filter,
        )
        self._active_plot_warmup_timer = timer
        with timer.activate():
            try:
                with timer.step("start_plot_payload_warmup_worker"):
                    self.plot_warmup_runner.start(
                        request=request,
                        run_job=self.app.commands.load.run_plot_payload_warmup,
                        on_finished=self._on_plot_payload_warmup_finished,
                    )
            except Exception as exc:
                self._finish_plot_warmup_timer("failed", message=str(exc))
                logger.warning("Failed to start plot payload warmup", exc_info=True)
                return False
        return True

    def _on_plot_payload_warmup_finished(
        self,
        request: PlotPayloadWarmupRequest,
        result: PlotPayloadWarmupResult,
    ) -> None:
        timer = self._active_plot_warmup_timer
        context = timer.activate() if timer is not None else nullcontext()
        with context:
            if isinstance(result, Failed):
                logger.debug(result.message)
                self._finish_plot_warmup_timer("failed", message=result.message)
                return
            assert isinstance(result, PlotPayloadCacheWarmed)
            step = (
                timer.step("attach_warmed_plot_payload_cache")
                if timer is not None
                else nullcontext()
            )
            with step:
                attached = self.app.commands.load.attach_warmed_plot_payload_cache(
                    result,
                )
            if isinstance(attached, Failed):
                logger.warning(attached.message)
                self._finish_plot_warmup_timer("failed", message=attached.message)
                return
            if isinstance(attached, LoadDataStaleResultIgnored):
                logger.debug("Ignoring warmed plot cache: %s", attached.reason)
                self._finish_plot_warmup_timer("stale", reason=attached.reason)
                return
            logger.info(
                "Warmed plot payload cache for %s shank %s (%s)",
                request.stream_key,
                request.shank_idx,
                ", ".join(result.warmed_spec_keys) or "no payloads",
            )
            self._finish_plot_warmup_timer(
                "completed",
                warmed_spec_keys=result.warmed_spec_keys,
            )

    def _open_load_context(self, *args: Any, **kwargs: Any) -> None:
        """Enter and hold the desktop busy context for an async load."""
        manager = self.callbacks.busy_context(*args, **kwargs)
        self._active_load_context_manager = manager
        self._active_load_context = manager.__enter__()

    def _close_load_context(self, exc: BaseException | None = None) -> None:
        """Exit the active desktop busy context, if one is open."""
        manager = self._active_load_context_manager
        try:
            if manager is None:
                return
            if exc is None:
                manager.__exit__(None, None, None)
            else:
                manager.__exit__(type(exc), exc, exc.__traceback__)
        finally:
            self._active_load_context = None
            self._active_load_context_manager = None
            self._active_load_id = None

    def present_cached_probe_selection(
        self,
        *,
        session_name: str,
        probe_name: str,
        target_shank: int,
    ) -> bool:
        """Present a cached probe-selection change after caller teardown."""
        timer = start_timing(
            "cached_probe_switch",
            recording_id=session_name,
            probe_name=probe_name,
            shank_idx=target_shank,
        )
        self._active_switch_timer = timer
        with timer.activate():
            with timer.step("activate_cached_probe_selection"):
                result = self.app.commands.load.activate_cached_probe_selection(
                    recording_id=session_name,
                    probe_name=probe_name,
                    target_shank=target_shank,
                )
        return self._finish_cached_probe_selection(
            result,
            session_name=session_name,
            probe_name=probe_name,
        )

    def _finish_cached_probe_selection(
        self,
        result: Any,
        *,
        session_name: str,
        probe_name: str,
    ) -> bool:
        if isinstance(result, Failed):
            logger.error(result.message)
            self._finish_switch_timer("failed", message=result.message)
            return False
        if isinstance(result, LoadDataAlreadyActiveResult):
            self._finish_switch_timer("already_active")
            self.schedule_next_probe_preload(session_name, probe_name)
            return True
        if isinstance(result, LoadDataFreshRequiredResult):
            self._finish_switch_timer("fresh_required")
            return False

        assert isinstance(result, LoadDataCachedActivated)
        logger.info("Activated cached stream %s", result.stream_key)
        self._finish_switch_timer("cached")
        self.schedule_next_probe_preload(session_name, probe_name)
        return True

    def _present_activated_stream(self, event: StreamActivated) -> None:
        """Display active stream state after an app-level activation event."""
        callbacks = self.callbacks
        timer = self._active_switch_timer

        with _timed_step(timer, "clear_empty_state"):
            callbacks.clear_empty_state()

        with _timed_step(timer, "active_probe_selection_state"):
            selection_state = self.app.queries.workspace.active_probe_selection_state()
        if selection_state is not None and selection_state.shanks:
            with _timed_step(timer, "populate_loaded_shanks"):
                self.selection_view.populate_loaded_shanks(
                    selection_state.shanks,
                    event.shank_idx,
                )
        with _timed_step(timer, "render_loaded_shank"):
            callbacks.render_loaded_shank(
                event.shank_idx,
                event.preserve_plot_selection,
            )

    def _event_matches_active_load(self, load_id: int | None) -> bool:
        """Return whether a load event should affect the current desktop state."""
        return load_id is None or load_id == self._active_load_id

    def _finish_switch_timer(self, status: str, **fields: Any) -> None:
        timer = self._active_switch_timer
        if timer is not None:
            timer.finish(status, **fields)
            self._active_switch_timer = None

    def _finish_preload_timer(
        self,
        timer: TimingSession | None,
        status: str,
        **fields: Any,
    ) -> None:
        if timer is not None:
            timer.finish(status, **fields)
        if timer is self._active_preload_timer:
            self._active_preload_timer = None

    def _finish_plot_warmup_timer(self, status: str, **fields: Any) -> None:
        timer = self._active_plot_warmup_timer
        if timer is not None:
            timer.finish(status, **fields)
            self._active_plot_warmup_timer = None

    def _promote_matching_active_preload(
        self,
        prepared: LoadDataFreshPrepared,
        *,
        session_name: str,
        probe_name: str,
        timer: TimingSession,
    ) -> _PromotedPreload | None:
        preload_execution = self._active_preload_execution
        if preload_execution is None or not self.preload_runner.is_running:
            return None
        if not _same_load_product(preload_execution.prepared, prepared):
            return None

        with timer.step("promote_active_preload"):
            foreground_execution = self.app.commands.load.start_fresh_load_data(
                prepared,
            )
            promoted = _PromotedPreload(
                preload_execution=preload_execution,
                foreground_execution=foreground_execution,
                session_name=session_name,
                probe_name=probe_name,
            )
            self._promoted_preload = promoted
            self._active_load_id = foreground_execution.load_id
            self._finish_preload_timer(
                self._active_preload_timer,
                "promoted",
                foreground_load_id=foreground_execution.load_id,
            )

        with timer.step("open_load_context"):
            self._open_load_context(
                "Loading heavy data...",
                "Data loaded successfully",
                disable_widgets=self.selection_view.selection_widgets(),
            )
        logger.info(
            "Promoting active preload for %s/%s to foreground load",
            session_name,
            probe_name,
        )
        with timer.step("prepare_for_fresh_stream_load"):
            self.callbacks.prepare_for_fresh_stream_load()
        return promoted

    def _foreground_load_is_running(self) -> bool:
        return self.load_runner.is_running or self._promoted_preload_is_running()

    def _promoted_preload_is_running(self) -> bool:
        return self._promoted_preload is not None and self.preload_runner.is_running


def _timed_step(timer: TimingSession | None, name: str):
    return timer.step(name) if timer is not None else nullcontext()


def _same_load_product(
    left: LoadDataFreshPrepared,
    right: LoadDataFreshPrepared,
) -> bool:
    left_target = left.target
    right_target = right.target
    if hasattr(left_target, "same_product_identity"):
        return bool(left_target.same_product_identity(right_target))

    return (
        getattr(left_target, "recording_id", None)
        == getattr(right_target, "recording_id", None)
        and getattr(left_target, "probe_name", None)
        == getattr(right_target, "probe_name", None)
        and left.stream_key == right.stream_key
        and _mouse_root_path(left_target) == _mouse_root_path(right_target)
    )


def _mouse_root_path(target: Any) -> Any:
    if hasattr(target, "mouse_root_path"):
        return target.mouse_root_path
    mouse_root = getattr(target, "mouse_root", None)
    if mouse_root is None:
        return None
    return getattr(mouse_root, "root", mouse_root)
