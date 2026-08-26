"""Desktop autosave recovery workflow."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.commands.autosave import (
    AUTOSAVE_DIRECTORY_NAME,
    AUTOSAVE_DOCUMENT_FILENAME,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointInspected,
    AutosaveCheckpointRecovered,
)
from ephys_alignment_gui.application.results.metadata import (
    ProbeSelected,
    RecordingSelected,
)
from ephys_alignment_gui.core.workflow import Failed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutosaveRecoveryCallbacks:
    """Desktop callbacks for autosave recovery."""

    select_folder: Callable[[Path | None], Path | None]
    default_folder: Callable[[], Path | None]
    confirm_recovery: Callable[[AutosaveCheckpointInspected], bool]
    set_mouse_root: Callable[[Path], bool]
    activate_selected_stream: Callable[..., bool]
    render_output_paths: Callable[[Path | None, Path | None], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    warning: Callable[[str, str], Any]


@dataclass
class DesktopAutosaveRecoveryCoordinator:
    """Coordinate recovering document checkpoints from the desktop UI."""

    app: Any
    selection_view: Any
    callbacks: AutosaveRecoveryCallbacks

    def recover_autosave(self) -> bool:
        """Prompt for and recover an autosave checkpoint."""
        selected = self.callbacks.select_folder(self.callbacks.default_folder())
        if selected is None:
            return False
        checkpoint_path = _checkpoint_path_from_selection(selected)
        if checkpoint_path is None:
            self.callbacks.warning(
                "Recover Autosave",
                "Selected folder does not contain an autosave checkpoint.",
            )
            return False

        inspected = self.app.commands.autosave.inspect_checkpoint(checkpoint_path)
        if isinstance(inspected, Failed):
            self.callbacks.warning("Recover Autosave", inspected.message)
            return False
        assert isinstance(inspected, AutosaveCheckpointInspected)

        if not self._ensure_mouse_root_loaded(inspected):
            return False

        inspected = self.app.commands.autosave.inspect_checkpoint(checkpoint_path)
        if isinstance(inspected, Failed):
            self.callbacks.warning("Recover Autosave", inspected.message)
            return False
        assert isinstance(inspected, AutosaveCheckpointInspected)
        if inspected.recoverable_alignment_count == 0:
            self.callbacks.warning(
                "Recover Autosave",
                "No checkpoint alignments match the loaded mouse root.",
            )
            return False
        if not self.callbacks.confirm_recovery(inspected):
            return False

        with self.callbacks.busy_context(
            "Recovering autosave...",
            "Autosave recovered",
            disable_widgets=self.selection_view.selection_widgets(),
        ):
            recovered = self.app.commands.autosave.recover_checkpoint(checkpoint_path)
            if isinstance(recovered, Failed):
                self.callbacks.warning("Recover Autosave", recovered.message)
                return False
            assert isinstance(recovered, AutosaveCheckpointRecovered)
            self._warn_recovery_details(recovered)
            if not self._restore_recovered_selection(recovered):
                return False

        if recovered.selected_alignment_key is None:
            return True
        return self.callbacks.activate_selected_stream(
            preserve_plot_selection=False,
        )

    def _ensure_mouse_root_loaded(
        self,
        inspected: AutosaveCheckpointInspected,
    ) -> bool:
        if self.app.queries.workspace.mouse_root_loaded():
            return True
        if inspected.mouse_root is None:
            self.callbacks.warning(
                "Recover Autosave",
                "Load the matching mouse root before recovering this checkpoint.",
            )
            return False
        if self.callbacks.set_mouse_root(inspected.mouse_root):
            return True
        self.callbacks.warning(
            "Recover Autosave",
            f"Failed to load checkpoint mouse root: {inspected.mouse_root}",
        )
        return False

    def _restore_recovered_selection(
        self,
        recovered: AutosaveCheckpointRecovered,
    ) -> bool:
        key = recovered.selected_alignment_key
        if key is None:
            return True

        active_probe = self.app.queries.workspace.active_probe_selection_state()
        if active_probe is None:
            self.callbacks.warning(
                "Recover Autosave",
                "Recovered autosave did not contain an active probe selection.",
            )
            return False

        session_idx = self.selection_view.select_session_text(key.recording_id)
        if session_idx is None:
            self.callbacks.warning(
                "Recover Autosave",
                f"Recovered session is not available: {key.recording_id}",
            )
            return False

        recording = self.app.commands.metadata.select_recording_metadata(
            key.recording_id,
        )
        if isinstance(recording, Failed):
            self.callbacks.warning("Recover Autosave", recording.message)
            return False
        assert isinstance(recording, RecordingSelected)
        self.selection_view.populate_probes(recording.probes)

        probe_idx = self.selection_view.select_probe_text(active_probe.probe_name)
        if probe_idx is None:
            self.callbacks.warning(
                "Recover Autosave",
                f"Recovered probe is not available: {active_probe.probe_name}",
            )
            return False

        probe = self.app.commands.metadata.select_probe_metadata(
            key.recording_id,
            active_probe.probe_name,
        )
        if isinstance(probe, Failed):
            self.callbacks.warning("Recover Autosave", probe.message)
            return False
        assert isinstance(probe, ProbeSelected)
        self.selection_view.populate_probe_shanks(probe.shanks)
        self.selection_view.select_shank_index(key.shank_idx)

        shank = self.app.commands.shanks.select_shank(
            key.shank_idx,
            source="autosave-recovered",
        )
        if isinstance(shank, Failed):
            self.callbacks.warning("Recover Autosave", shank.message)
            return False

        self.callbacks.render_output_paths(
            self.app.queries.workspace.active_output_root(),
            self.app.queries.workspace.active_output_directory(),
        )
        return True

    def _warn_recovery_details(self, recovered: AutosaveCheckpointRecovered) -> None:
        if recovered.skipped_keys:
            logger.warning(
                "Recovered autosave with %d invalid alignment state(s) skipped",
                len(recovered.skipped_keys),
            )
        for warning in recovered.warnings:
            logger.warning("Autosave recovery warning: %s", warning)


def _checkpoint_path_from_selection(selected: Path) -> Path | None:
    selected = Path(selected)
    if selected.is_file() and selected.name == AUTOSAVE_DOCUMENT_FILENAME:
        return selected
    candidates = (
        selected / AUTOSAVE_DOCUMENT_FILENAME,
        selected / AUTOSAVE_DIRECTORY_NAME / AUTOSAVE_DOCUMENT_FILENAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
