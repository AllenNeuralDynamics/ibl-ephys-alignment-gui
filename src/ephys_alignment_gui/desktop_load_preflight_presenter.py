"""Desktop load preflight prompts and command gating."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.workflow import (
    CHOOSE_OUTPUT_FOLDER,
    Blocked,
    Ok,
    PolicyResult,
    Requirement,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputFolderPromptCallbacks:
    """Callbacks for output-folder prompt side effects."""

    derive_output_directory_from_save_root: Callable[[], bool]
    has_output_directory: Callable[[], bool]
    select_output_folder: Callable[[], bool]


@dataclass
class DesktopOutputFolderPrompt:
    """Render desktop output-folder requirements for load/save commands."""

    callbacks: OutputFolderPromptCallbacks
    parent: Any = None
    message_box_factory: Callable[[Any], Any] = QtWidgets.QMessageBox

    def ensure_for_load(self, requirement: Requirement | None = None) -> bool:
        """Require output before loading data, deriving from save root first."""
        if self.callbacks.derive_output_directory_from_save_root():
            return True
        return self._ensure_output_directory(
            requirement
            or Requirement(
                code="output_required",
                message="Choose an output folder before loading data.",
                action=CHOOSE_OUTPUT_FOLDER,
            ),
            informative_text=(
                "The GUI saves in-progress alignments when switching probes or "
                "sessions."
            ),
            cancel_log="Load data cancelled",
        )

    def ensure_for_save(self, requirement: Requirement | None = None) -> bool:
        """Require output before saving alignment outputs."""
        return self._ensure_output_directory(
            requirement
            or Requirement(
                code="output_required",
                message="Choose an output folder before saving.",
                action=CHOOSE_OUTPUT_FOLDER,
            ),
            informative_text=(
                "The GUI writes channel locations and alignment history to the "
                "output folder."
            ),
            cancel_log="Save cancelled",
        )

    def _ensure_output_directory(
        self,
        requirement: Requirement,
        *,
        informative_text: str,
        cancel_log: str,
    ) -> bool:
        if self.callbacks.has_output_directory():
            return True

        msg = self.message_box_factory(self.parent)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("Output Folder Required")
        msg.setText(requirement.message)
        msg.setInformativeText(informative_text)
        set_button = msg.addButton(
            "Set Output Folder...", QtWidgets.QMessageBox.AcceptRole
        )
        msg.addButton(QtWidgets.QMessageBox.Cancel)
        msg.setDefaultButton(set_button)
        msg.exec_()

        if msg.clickedButton() != set_button:
            logger.info("%s: output directory is not set.", cancel_log)
            return False

        if not self.callbacks.select_output_folder():
            logger.info("%s: no output folder selected.", cancel_log)
            return False

        if not self.callbacks.has_output_directory():
            logger.error(
                "Output folder selected but no probe output directory was derived."
            )
            return False
        return True


@dataclass
class DesktopLoadPreflightPresenter:
    """Own desktop preflight handling for the Load Data button."""

    can_load_data: Callable[[], PolicyResult]
    load_heavy_data: Callable[[], None]
    output_folder_prompt: DesktopOutputFolderPrompt

    def load_data_button_pressed(self) -> bool:
        """Run load-data policy checks, render blockers, and start heavy load."""
        result = self.can_load_data()
        if isinstance(result, Blocked):
            if not self._handle_blocked_load(result):
                return False
            result = self.can_load_data()

        if not isinstance(result, Ok):
            if isinstance(result, Blocked):
                self.log_requirement(result.first)
            return False

        logger.info("Load Data button pressed")
        self.load_heavy_data()
        return True

    def _handle_blocked_load(self, result: Blocked) -> bool:
        requirement = result.first
        if requirement.action == CHOOSE_OUTPUT_FOLDER:
            return self.output_folder_prompt.ensure_for_load(requirement)
        self.log_requirement(requirement)
        return False

    @staticmethod
    def log_requirement(requirement: Requirement) -> None:
        """Log a policy requirement that has no desktop prompt action."""
        logger.error(requirement.message)
