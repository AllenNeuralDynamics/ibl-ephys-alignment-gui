"""Desktop presentation shell for loading ephys/histology data."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
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
from ephys_alignment_gui.runtime.histology_loader import (
    HistologyDataLoaded,
    HistologyDataUnavailable,
)

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
            self._present_cached_stream(begin_result)
            self.selection_view.set_load_data_enabled(True)
            return True
        assert isinstance(begin_result, LoadDataFreshPrepared)

        with callbacks.busy_context(
            "Loading heavy data...",
            "Data loaded successfully",
            disable_widgets=self.selection_view.load_data_widget(),
        ) as ctx:
            logger.info("=== Starting heavy data load ===")
            callbacks.prepare_for_fresh_stream_load()

            ctx.update_message("Loading ephys and histology data...")
            logger.info(
                "Loading probe data, active shank index %s",
                begin_result.shank_idx,
            )
            completed = self.app.commands.load.complete_fresh_load_data(begin_result)
            if isinstance(completed, Failed):
                logger.error(completed.message)
                return False
            assert isinstance(completed, LoadDataFreshCompleted)
            stream_runtime = completed.ephys.stream_runtime
            target_shank = completed.ephys.shank_idx

            logger.info("Loaded ephys data from %s", stream_runtime.stream.ephys_dir)

            histology_result = completed.histology
            if isinstance(histology_result, HistologyDataLoaded):
                logger.info("Atlas and histology loaded successfully")
            elif isinstance(histology_result, HistologyDataUnavailable):
                logger.error(histology_result.message)

            ctx.update_message("Setting up visualization...")
            callbacks.render_loaded_shank(
                target_shank,
                completed.preserve_plot_selection,
            )

            callbacks.clear_empty_state()
            logger.info("=== Heavy data load complete ===")
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
        self._present_cached_stream(result)
        self.selection_view.set_load_data_enabled(True)
        return True

    def _present_cached_stream(self, result: LoadDataCachedActivated) -> None:
        """Display an already-loaded stream from the cache without heavy IO."""
        callbacks = self.callbacks
        activated = result.activated
        target_shank = activated.shank_idx

        callbacks.clear_empty_state()

        if activated.probe.shanks:
            self.selection_view.populate_loaded_shanks(
                activated.probe.shanks,
                target_shank,
            )
        callbacks.display_output_directory(activated.probe.output_directory)

        callbacks.render_loaded_shank(target_shank, True)
        logger.info("Activated cached stream %s", result.stream_key)
