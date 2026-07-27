"""Desktop presentation shell for probe selection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.controller import ProbeSelected
from ephys_alignment_gui.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopProbeSelectionCallbacks:
    """Desktop callbacks used by the probe-selection presenter."""

    mouse_root_loaded: Callable[[], bool]
    active_shank_idx: Callable[[], int]
    capture_pending_reference_lines: Callable[[], None]
    stash_and_detach_current: Callable[[], None]
    present_cached_probe_selection: Callable[[str, str, int], bool]
    show_empty_state: Callable[[], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    init_session_variables: Callable[[], None]
    select_shank_for_view: Callable[[int, str], int | None]
    display_output_directory: Callable[[Path | None], None]


@dataclass
class DesktopProbeSelectionPresenter:
    """Coordinate desktop behavior for selecting a probe."""

    commands: Any
    selection_view: Any
    callbacks: DesktopProbeSelectionCallbacks

    def probe_selected(self) -> bool:
        """Select the current probe or present its cached stream."""
        callbacks = self.callbacks
        if not callbacks.mouse_root_loaded():
            return False

        session_name = self.selection_view.current_session()
        probe_name = self.selection_view.current_probe()
        if not session_name or not probe_name:
            return False

        callbacks.capture_pending_reference_lines()
        callbacks.stash_and_detach_current()

        if callbacks.present_cached_probe_selection(
            session_name,
            probe_name,
            callbacks.active_shank_idx(),
        ):
            return True

        return self._prepare_probe_for_fresh_load(session_name, probe_name)

    def _prepare_probe_for_fresh_load(
        self,
        session_name: str,
        probe_name: str,
    ) -> bool:
        """Load channel metadata and prepare the desktop for explicit Load."""
        callbacks = self.callbacks
        callbacks.show_empty_state()
        with callbacks.busy_context(
            "Loading channel info...",
            "Ready",
            disable_widgets=self.selection_view.selection_widgets(),
        ):
            result = self.commands.select_probe_metadata(session_name, probe_name)
            if isinstance(result, Failed):
                logger.error(result.message)
                self.selection_view.set_load_data_enabled(False)
                return False
            assert isinstance(result, ProbeSelected)

            if result.shanks:
                self.selection_view.populate_probe_shanks(result.shanks)
                logger.info("Found %s shanks in data.", result.n_shanks)

            callbacks.init_session_variables()
            if callbacks.select_shank_for_view(0, "probe-selected") is None:
                self.selection_view.set_load_data_enabled(False)
                return False

            callbacks.display_output_directory(result.output_directory)

        self.selection_view.set_load_data_enabled(True)
        return True
