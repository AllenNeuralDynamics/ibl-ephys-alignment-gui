"""Desktop presentation shell for loading ephys/histology data."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results import (
    LoadDataAlreadyActiveResult,
    LoadDataCachedActivated,
    LoadDataFreshCompleted,
    LoadDataFreshPrepared,
    LoadDataFreshRequiredResult,
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
from ephys_alignment_gui.io.load_data_job import LoadDataJobCancelled

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopLoadDataCallbacks:
    """Desktop callbacks used by the load-data presenter."""

    reference_line_positions: Callable[[], Any]
    prepare_for_fresh_stream_load: Callable[[], None]
    display_output_directory: Callable[[Path | None], None]
    render_loaded_shank: Callable[[int, bool | None], None]
    clear_empty_state: Callable[[], None]
    busy_context: Callable[..., AbstractContextManager[Any]]


@dataclass
class DesktopLoadDataPresenter:
    """Coordinate desktop behavior for cached and fresh data loads."""

    app: Any
    selection_view: Any
    callbacks: DesktopLoadDataCallbacks
    _active_load_context: Any | None = field(default=None, init=False, repr=False)

    def connect_load_events(self) -> list[EventSubscription]:
        """Subscribe desktop load presentation to semantic load events."""
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
        """Update desktop progress presentation for an active load context."""
        if self._active_load_context is not None:
            self._active_load_context.update_message(event.message)

    def on_load_data_failed(self, event: LoadDataFailed) -> None:
        """Log fresh-load failures reported by the app layer."""
        logger.error(event.message)

    def on_load_data_cancelled(self, event: LoadDataCancelled) -> None:
        """Log fresh-load cancellation reported by the app layer."""
        logger.info("Load cancelled: %s", event.reason)

    def on_histology_load_reported(self, event: HistologyLoadReported) -> None:
        """Log non-fatal histology availability for a fresh load."""
        if event.status in {"already_loaded", "loaded"}:
            logger.info("Atlas and histology loaded successfully")
        elif event.status == "unavailable" and event.message is not None:
            logger.error(event.message)

    def on_stream_activated(self, event: StreamActivated) -> None:
        """Render desktop state for an activated stream/shank."""
        self._present_activated_stream(event)

    def load_heavy_data(self) -> bool:
        """Load or activate the selected stream/shank for desktop display."""
        callbacks = self.callbacks
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

        with callbacks.busy_context(
            "Loading heavy data...",
            "Data loaded successfully",
            disable_widgets=self.selection_view.load_data_widget(),
        ) as ctx:
            logger.info("=== Starting heavy data load ===")
            callbacks.prepare_for_fresh_stream_load()
            self._active_load_context = ctx

            try:
                logger.info(
                    "Loading probe data, active shank index %s",
                    begin_result.shank_idx,
                )
                job_result = self.app.commands.load.run_fresh_load_data(begin_result)
                if isinstance(job_result, Failed | LoadDataJobCancelled):
                    return False

                ctx.update_message("Setting up visualization...")
                completed = self.app.commands.load.activate_completed_fresh_load_data(
                    begin_result,
                    job_result,
                )
                if isinstance(completed, Failed):
                    return False
                assert isinstance(completed, LoadDataFreshCompleted)
                stream_runtime = completed.ephys.stream_runtime

                logger.info(
                    "Loaded ephys data from %s",
                    stream_runtime.stream.ephys_dir,
                )
                logger.info("=== Heavy data load complete ===")
            finally:
                self._active_load_context = None
        return True

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
        if selection_state is not None:
            callbacks.display_output_directory(selection_state.output_directory)

        callbacks.render_loaded_shank(
            event.shank_idx,
            event.preserve_plot_selection,
        )
        self.selection_view.set_load_data_enabled(True)
