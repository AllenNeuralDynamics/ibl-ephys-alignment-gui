"""Desktop workflow for loading previous alignments."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results.alignment_persistence import (
    NoPreviousAlignments,
    PreviousAlignmentPackageLoaded,
)
from ephys_alignment_gui.core.alignment_events import (
    PreviousAlignmentLoadFailed,
    PreviousAlignmentsLoaded,
    PreviousAlignmentsUnavailable,
)
from ephys_alignment_gui.core.event_bus import EventSubscription
from ephys_alignment_gui.core.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviousAlignmentLoadCallbacks:
    """Desktop side effects for loading previous alignments."""

    select_folder: Callable[[Path | None], Path | None]
    default_folder: Callable[[], Path | None]
    use_docdb: Callable[[], bool]
    set_reload_folder_text: Callable[[str], None]
    render_alignment_choices: Callable[[list[str]], None]
    select_alignment: Callable[[int], bool]
    busy_context: Callable[..., AbstractContextManager[Any]]
    reload_button: Callable[[], Any]


@dataclass
class DesktopPreviousAlignmentLoadCoordinator:
    """Own desktop shell behavior for the previous-alignment load command."""

    commands: Any
    events: Any
    callbacks: PreviousAlignmentLoadCallbacks
    _last_selection_result: bool = field(default=True, init=False, repr=False)

    def connect_previous_alignment_events(self) -> list[EventSubscription]:
        """Subscribe desktop coordination to previous-alignment load events."""
        return [
            self.events.subscribe(
                PreviousAlignmentsLoaded,
                self.on_previous_alignments_loaded,
            ),
            self.events.subscribe(
                PreviousAlignmentsUnavailable,
                self.on_previous_alignments_unavailable,
            ),
            self.events.subscribe(
                PreviousAlignmentLoadFailed,
                self.on_previous_alignment_load_failed,
            ),
        ]

    def on_previous_alignments_loaded(
        self,
        event: PreviousAlignmentsLoaded,
    ) -> None:
        """Render loaded previous-alignment choices."""
        choices = list(event.choices)
        self.callbacks.render_alignment_choices(choices)
        self._last_selection_result = True
        if event.auto_select:
            self._last_selection_result = self.callbacks.select_alignment(0)
        if self._last_selection_result:
            logger.info("Loaded %d previous alignments", len(choices))

    def on_previous_alignments_unavailable(
        self,
        _event: PreviousAlignmentsUnavailable,
    ) -> None:
        """Log that no previous alignments were found."""
        logger.info("No previous alignments found")

    def on_previous_alignment_load_failed(
        self,
        event: PreviousAlignmentLoadFailed,
    ) -> None:
        """Log previous-alignment load failure."""
        logger.error(event.message)

    def load_existing_alignments(self) -> bool:
        """Prompt for alignment history and render the command result."""
        ready = self.commands.can_load_previous_alignments()
        if isinstance(ready, Failed):
            logger.error(ready.message)
            return False

        selected = self.callbacks.select_folder(self.callbacks.default_folder())
        if selected is None:
            return False

        use_docdb = self.callbacks.use_docdb()
        folder_path = selected
        self.callbacks.set_reload_folder_text(str(folder_path))

        with self.callbacks.busy_context(
            "Loading alignments...",
            "Alignments loaded",
            disable_widgets=self.callbacks.reload_button(),
        ):
            self._last_selection_result = True
            logger.info(
                "Loading alignments from %s, use_docdb=%s",
                folder_path,
                use_docdb,
            )
            result = self.commands.load_previous_alignments(
                folder=folder_path,
                use_docdb=use_docdb,
            )
            if isinstance(result, Failed):
                return False
            if not self._last_selection_result:
                return False
            if isinstance(result, NoPreviousAlignments):
                return True
            if isinstance(result, PreviousAlignmentPackageLoaded):
                logger.info(
                    "Loaded previous alignments for %d stream/shank(s)",
                    result.loaded_count,
                )

        return True
