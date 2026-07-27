"""Desktop presentation shell for loading ephys/histology data."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.app import CachedEphysDataActivated
from ephys_alignment_gui.histology_data_workflow import (
    HistologyDataLoaded,
    HistologyDataUnavailable,
)
from ephys_alignment_gui.session_runtime import (
    LoadDataAlreadyActive,
    LoadDataCachedStreamAvailable,
)
from ephys_alignment_gui.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopLoadDataCallbacks:
    """Desktop callbacks used by the load-data presenter."""

    session_name: Callable[[], str]
    probe_name: Callable[[], str]
    capture_pending_reference_lines: Callable[[], None]
    stash_and_detach_current: Callable[[], None]
    teardown_session: Callable[[], None]
    init_session_variables: Callable[[], None]
    select_shank_for_view: Callable[[int, str], int | None]
    populate_shanks: Callable[[list[str], int], None]
    display_output_directory: Callable[[Path | None], None]
    setup_session_view: Callable[[bool | None, int], None]
    clear_empty_state: Callable[[], None]
    set_load_data_enabled: Callable[[bool], None]
    set_histology_available: Callable[[bool], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    load_data_button: Callable[[], Any]


@dataclass
class DesktopLoadDataPresenter:
    """Coordinate desktop behavior for cached and fresh data loads."""

    app: Any
    callbacks: DesktopLoadDataCallbacks

    def load_heavy_data(self) -> bool:
        """Load or activate the selected stream/shank for desktop display."""
        callbacks = self.callbacks
        target_shank = self.app.queries.active_shank_selection().shank_idx
        session_name = callbacks.session_name()
        probe_name = callbacks.probe_name()
        stream_key = self.app.queries.stream_key_for_selection(
            session_name,
            probe_name,
        )
        load_plan = self.app.queries.plan_load_data(stream_key, target_shank)

        if isinstance(load_plan, LoadDataAlreadyActive):
            logger.info(
                "Data already loaded for stream %s shank %s; skipping load",
                stream_key,
                target_shank,
            )
            return True

        if isinstance(load_plan, LoadDataCachedStreamAvailable):
            callbacks.capture_pending_reference_lines()
            callbacks.stash_and_detach_current()
            cached_stream_key = load_plan.target.stream_key
            assert cached_stream_key is not None
            if self.present_cached_stream(
                session_name=session_name,
                probe_name=probe_name,
                stream_key=cached_stream_key,
                shank_idx=load_plan.target.shank_idx,
            ):
                callbacks.set_load_data_enabled(True)
                return True
            return False

        with callbacks.busy_context(
            "Loading heavy data...",
            "Data loaded successfully",
            disable_widgets=callbacks.load_data_button(),
        ) as ctx:
            logger.info("=== Starting heavy data load ===")
            callbacks.capture_pending_reference_lines()
            prepared = self.app.commands.prepare_fresh_ephys_load(stream_key)
            callbacks.teardown_session()
            callbacks.init_session_variables()

            selected_shank = callbacks.select_shank_for_view(
                target_shank,
                "load-data",
            )
            if selected_shank is None:
                return False
            target_shank = selected_shank
            logger.info("Loading probe data, active shank index %s", target_shank)

            ctx.update_message("Loading ephys data...")
            logger.info("Loading ephys data...")
            load_result = self.app.commands.load_fresh_ephys_data(target_shank)
            if isinstance(load_result, Failed):
                logger.error(load_result.message)
                return False
            stream_runtime = load_result.stream_runtime
            target_shank = load_result.shank_idx

            logger.info("Loaded ephys data from %s", stream_runtime.stream.ephys_dir)

            if not self.app.queries.histology_data_loaded():
                ctx.update_message("Loading atlas and histology...")
                logger.info("Loading atlas and histology...")
            histology_result = self.app.commands.load_histology_data()
            if isinstance(histology_result, HistologyDataLoaded):
                logger.info("Atlas and histology loaded successfully")
            elif isinstance(histology_result, HistologyDataUnavailable):
                logger.error(histology_result.message)
                callbacks.set_histology_available(False)

            ctx.update_message("Setting up visualization...")
            callbacks.setup_session_view(
                prepared.preserve_plot_selection,
                target_shank,
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
        stream_key = self.app.queries.stream_key_for_selection(session_name, probe_name)
        load_plan = self.app.queries.plan_load_data(stream_key, target_shank)
        if not isinstance(load_plan, LoadDataCachedStreamAvailable):
            return False

        cached_stream_key = load_plan.target.stream_key
        assert cached_stream_key is not None
        if self.present_cached_stream(
            session_name=session_name,
            probe_name=probe_name,
            stream_key=cached_stream_key,
            shank_idx=load_plan.cached_shank_idx,
        ):
            self.callbacks.set_load_data_enabled(True)
            return True
        return False

    def present_cached_stream(
        self,
        *,
        session_name: str,
        probe_name: str,
        stream_key: tuple[str, str],
        shank_idx: int,
    ) -> bool:
        """Display an already-loaded stream from the cache without heavy IO."""
        callbacks = self.callbacks
        callbacks.init_session_variables()
        result = self.app.commands.activate_cached_ephys_data(
            recording_id=session_name,
            probe_name=probe_name,
            stream_key=stream_key,
            shank_idx=shank_idx,
        )
        if isinstance(result, Failed):
            logger.error(result.message)
            return False
        assert isinstance(result, CachedEphysDataActivated)
        target_shank = result.shank_idx

        callbacks.clear_empty_state()

        if result.probe.shanks:
            callbacks.populate_shanks(result.probe.shanks, target_shank)
        callbacks.display_output_directory(result.probe.output_directory)

        callbacks.setup_session_view(True, target_shank)
        logger.info("Activated cached stream %s", stream_key)
        return True
