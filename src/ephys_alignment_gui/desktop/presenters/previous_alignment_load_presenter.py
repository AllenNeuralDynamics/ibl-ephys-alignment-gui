"""Desktop workflow for loading previous alignments."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results import AlignmentChoicesUpdated
from ephys_alignment_gui.application.results.alignment_persistence import (
    NoPreviousAlignments,
)
from ephys_alignment_gui.application.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviousAlignmentLoadCallbacks:
    """Desktop side effects for loading previous alignments."""

    select_folder: Callable[[], str]
    use_docdb: Callable[[], bool]
    set_reload_folder_text: Callable[[str], None]
    render_alignment_choices: Callable[[list[str]], None]
    select_alignment: Callable[[int], bool]
    busy_context: Callable[..., AbstractContextManager[Any]]
    reload_button: Callable[[], Any]


@dataclass
class DesktopPreviousAlignmentLoadPresenter:
    """Own desktop shell behavior for the previous-alignment load command."""

    commands: Any
    callbacks: PreviousAlignmentLoadCallbacks

    def load_existing_alignments(self) -> bool:
        """Prompt for alignment history and render the command result."""
        ready = self.commands.can_load_previous_alignments()
        if isinstance(ready, Failed):
            logger.error(ready.message)
            return False

        selected = self.callbacks.select_folder()
        use_docdb = self.callbacks.use_docdb()
        # Cancel returns "". Keep DocDB cancel semantics: without a local folder,
        # DocDB mode may still load using repository defaults.
        if not selected and not use_docdb:
            return False

        folder_path = Path(selected) if selected else None
        if folder_path is not None:
            self.callbacks.set_reload_folder_text(str(folder_path))

        with self.callbacks.busy_context(
            "Loading alignments...",
            "Alignments loaded",
            disable_widgets=self.callbacks.reload_button(),
        ):
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
                logger.error(result.message)
                return False
            if isinstance(result, AlignmentChoicesUpdated):
                self.callbacks.render_alignment_choices(result.choices)
                if not self.callbacks.select_alignment(0):
                    return False
                logger.info("Loaded %d previous alignments", len(result.choices))
            elif isinstance(result, NoPreviousAlignments):
                logger.info("No previous alignments found")

        return True
