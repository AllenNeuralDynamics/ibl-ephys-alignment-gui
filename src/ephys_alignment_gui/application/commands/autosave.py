"""Application commands for cheap alignment document checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointCleared,
    AutosaveCheckpointRead,
    AutosaveCheckpointRestored,
    AutosaveCheckpointWritten,
)
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.core.document_snapshot import AlignmentDocumentSnapshot
from ephys_alignment_gui.core.workflow import Failed

AUTOSAVE_DIRECTORY_NAME = "autosave"
AUTOSAVE_DOCUMENT_FILENAME = "alignment_document.json"


@dataclass
class AutosaveCheckpointCommandHandler:
    """Write, read, restore, and clear document-only autosave checkpoints."""

    controller: AlignmentController

    def default_checkpoint_path(self) -> Path | Failed:
        """Return the package-local autosave checkpoint path."""
        package_directory = self.controller.document.output_package_directory
        if package_directory is None:
            return Failed(
                "No alignment output package is available for autosave checkpoint."
            )
        return (
            package_directory
            / AUTOSAVE_DIRECTORY_NAME
            / AUTOSAVE_DOCUMENT_FILENAME
        )

    def write_checkpoint(
        self,
        path: Path | None = None,
    ) -> AutosaveCheckpointWritten | Failed:
        """Write the current document-owned alignment state to one checkpoint."""
        path_or_failed = self._checkpoint_path(path)
        if isinstance(path_or_failed, Failed):
            return path_or_failed
        checkpoint_path = path_or_failed
        snapshot = AlignmentDocumentSnapshot.from_document(
            self.controller.document
        )
        try:
            snapshot.write_json(checkpoint_path)
        except OSError as exc:
            return Failed(
                f"Failed to write autosave checkpoint {checkpoint_path}: {exc}"
            )
        return AutosaveCheckpointWritten(
            path=checkpoint_path,
            alignment_state_count=len(snapshot.alignment_states),
        )

    def read_checkpoint(
        self,
        path: Path | None = None,
    ) -> AutosaveCheckpointRead | Failed:
        """Read a document checkpoint without mutating the live document."""
        path_or_failed = self._checkpoint_path(path)
        if isinstance(path_or_failed, Failed):
            return path_or_failed
        checkpoint_path = path_or_failed
        if not checkpoint_path.exists():
            return Failed(f"No autosave checkpoint found at {checkpoint_path}.")
        try:
            snapshot = AlignmentDocumentSnapshot.read_json(checkpoint_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return Failed(
                f"Failed to read autosave checkpoint {checkpoint_path}: {exc}"
            )
        return AutosaveCheckpointRead(path=checkpoint_path, snapshot=snapshot)

    def restore_checkpoint(
        self,
        path: Path | None = None,
    ) -> AutosaveCheckpointRestored | Failed:
        """Restore a checkpoint into the existing live document object."""
        read = self.read_checkpoint(path)
        if isinstance(read, Failed):
            return read
        read.snapshot.restore_into(self.controller.document)
        return AutosaveCheckpointRestored(
            path=read.path,
            alignment_state_count=len(read.snapshot.alignment_states),
            selected_alignment_key=self.controller.document.selected_alignment_key,
        )

    def clear_checkpoint(
        self,
        path: Path | None = None,
    ) -> AutosaveCheckpointCleared | Failed:
        """Remove the autosave checkpoint file if it exists."""
        path_or_failed = self._checkpoint_path(path)
        if isinstance(path_or_failed, Failed):
            return path_or_failed
        checkpoint_path = path_or_failed
        existed = checkpoint_path.exists()
        if existed:
            try:
                checkpoint_path.unlink()
            except OSError as exc:
                return Failed(
                    "Failed to clear autosave checkpoint "
                    f"{checkpoint_path}: {exc}"
                )
        return AutosaveCheckpointCleared(path=checkpoint_path, existed=existed)

    def _checkpoint_path(self, path: Path | None) -> Path | Failed:
        if path is not None:
            return Path(path)
        return self.default_checkpoint_path()
