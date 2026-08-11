"""Desktop coordination shell for loading ephys/histology data."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from ephys_alignment_gui.application.results import (
    FreshLoadExecution,
    LoadDataAlreadyActiveResult,
    LoadDataCachedActivated,
    LoadDataFreshCompleted,
    LoadDataFreshPrepared,
    LoadDataFreshRequiredResult,
    LoadDataStaleResultIgnored,
)
from ephys_alignment_gui.application.workflow import Failed
from ephys_alignment_gui.core.alignment_events import (
    HistologyLoadReported,
    LoadDataCancelled,
    LoadDataFailed,
    LoadDataProgressed,
    StreamActivated,
)
from ephys_alignment_gui.core.event_bus import EventSubscription
from ephys_alignment_gui.desktop.workers.load_data_runner import (
    FreshLoadJobResult,
    FreshLoadRunner,
    QtFreshLoadRunner,
)
from ephys_alignment_gui.io.load_data_job import (
    LoadDataJobCancelled,
    LoadDataJobCompleted,
    LoadDataJobProgress,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopLoadDataCallbacks:
    """Desktop callbacks used by the load-data coordinator."""

    reference_line_positions: Callable[[], Any]
    prepare_for_fresh_stream_load: Callable[[], None]
    render_loaded_shank: Callable[[int, bool | None], None]
    clear_empty_state: Callable[[], None]
    busy_context: Callable[..., AbstractContextManager[Any]]


@dataclass
class DesktopLoadDataCoordinator:
    """Coordinate desktop behavior for cached and fresh data loads."""

    app: Any
    selection_view: Any
    callbacks: DesktopLoadDataCallbacks
    load_runner: FreshLoadRunner = field(default_factory=QtFreshLoadRunner)
    _active_load_context: Any | None = field(default=None, init=False, repr=False)
    _active_load_context_manager: Any | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _active_load_id: int | None = field(default=None, init=False, repr=False)

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

    def load_heavy_data(self) -> bool:
        """Load or activate the selected stream/shank for desktop display."""
        callbacks = self.callbacks
        if self.load_runner.is_running:
            logger.info("Load request ignored because a foreground load is active")
            self.cancel_active_load("superseded by a newer load request")
            return False

        target_shank = self.app.queries.workspace.active_shank_selection().shank_idx
        session_name = self.selection_view.current_session()
        probe_name = self.selection_view.current_probe()
        begin_result = self.app.commands.load.begin_load_data(
            recording_id=session_name,
            probe_name=probe_name,
            target_shank=target_shank,
            outgoing_reference_lines=callbacks.reference_line_positions(),
        )
        if isinstance(begin_result, Failed):
            logger.error(begin_result.message)
            return False
        if isinstance(begin_result, LoadDataAlreadyActiveResult):
            return True
        if isinstance(begin_result, LoadDataCachedActivated):
            logger.info("Activated cached stream %s", begin_result.stream_key)
            return True
        assert isinstance(begin_result, LoadDataFreshPrepared)
        execution = self.app.commands.load.start_fresh_load_data(begin_result)
        invocation = self.app.commands.load.fresh_load_job_invocation(execution)
        if isinstance(invocation, LoadDataJobCancelled):
            logger.info("Load cancelled: %s", invocation.reason)
            return False

        self._open_load_context(
            "Loading heavy data...",
            "Data loaded successfully",
            disable_widgets=self.selection_view.load_data_widget(),
        )
        logger.info("=== Starting heavy data load ===")
        callbacks.prepare_for_fresh_stream_load()
        self._active_load_id = execution.load_id
        logger.info(
            "Loading probe data, active shank index %s",
            begin_result.shank_idx,
        )
        try:
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
            logger.exception("Failed to start background load")
            return False
        return True

    def cancel_active_load(self, reason: str) -> bool:
        """Request cancellation for any active foreground load."""
        if self._active_load_id is None and not self.load_runner.is_running:
            return False

        self.app.commands.load.cancel_active_fresh_load(reason)
        if self.load_runner.is_running:
            self.load_runner.cancel(reason)
        return True

    def shutdown_active_load(
        self,
        reason: str = "application closing",
        *,
        timeout_ms: int = 5000,
    ) -> bool:
        """Cancel and settle any active foreground load before desktop teardown."""
        if self._active_load_id is None and not self.load_runner.is_running:
            return True

        self.app.commands.load.cancel_active_fresh_load(reason)
        stopped = self.load_runner.shutdown(reason, timeout_ms=timeout_ms)
        if stopped:
            self._close_load_context(RuntimeError(f"Load cancelled: {reason}"))
        return stopped

    def _on_load_worker_progress(
        self,
        execution: FreshLoadExecution,
        event: LoadDataJobProgress,
    ) -> None:
        """Publish worker progress from the GUI thread."""
        self.app.commands.load.publish_fresh_load_progress(execution, event)

    def _on_load_worker_finished(
        self,
        execution: FreshLoadExecution,
        job_result: FreshLoadJobResult,
    ) -> None:
        """Publish worker completion and activate successful fresh-load data."""
        published = self.app.commands.load.publish_started_fresh_load_job_result(
            execution,
            job_result,
        )
        if isinstance(published, Failed):
            logger.error(published.message)
            self._close_load_context(RuntimeError(published.message))
            return
        if isinstance(published, LoadDataJobCancelled):
            logger.info("Load cancelled: %s", published.reason)
            self._close_load_context(
                RuntimeError(f"Load cancelled: {published.reason}")
            )
            return
        if not isinstance(published, LoadDataJobCompleted):
            self._close_load_context(RuntimeError("Fresh load returned no result"))
            return

        if self._active_load_context is not None:
            self._active_load_context.update_message("Setting up visualization...")
        completed = self.app.commands.load.activate_started_fresh_load_data(
            execution,
            published,
        )
        if isinstance(completed, Failed):
            logger.error(completed.message)
            self._close_load_context(RuntimeError(completed.message))
            return
        if isinstance(completed, LoadDataStaleResultIgnored):
            logger.info("Load result ignored: %s", completed.reason)
            self._close_load_context(RuntimeError(completed.reason))
            return
        assert isinstance(completed, LoadDataFreshCompleted)
        stream_runtime = completed.ephys.stream_runtime
        logger.info("Loaded ephys data from %s", stream_runtime.stream.ephys_dir)
        logger.info("=== Heavy data load complete ===")
        self._close_load_context()

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
        result = self.app.commands.load.activate_cached_probe_selection(
            recording_id=session_name,
            probe_name=probe_name,
            target_shank=target_shank,
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        if isinstance(result, LoadDataAlreadyActiveResult):
            self.selection_view.set_load_data_enabled(True)
            return True
        if isinstance(result, LoadDataFreshRequiredResult):
            return False

        assert isinstance(result, LoadDataCachedActivated)
        logger.info("Activated cached stream %s", result.stream_key)
        return True

    def _present_activated_stream(self, event: StreamActivated) -> None:
        """Display active stream state after an app-level activation event."""
        callbacks = self.callbacks

        callbacks.clear_empty_state()

        selection_state = self.app.queries.workspace.active_probe_selection_state()
        if selection_state is not None and selection_state.shanks:
            self.selection_view.populate_loaded_shanks(
                selection_state.shanks,
                event.shank_idx,
            )
        callbacks.render_loaded_shank(
            event.shank_idx,
            event.preserve_plot_selection,
        )
        self.selection_view.set_load_data_enabled(True)

    def _event_matches_active_load(self, load_id: int | None) -> bool:
        """Return whether a load event should affect the current desktop state."""
        return load_id is None or load_id == self._active_load_id
