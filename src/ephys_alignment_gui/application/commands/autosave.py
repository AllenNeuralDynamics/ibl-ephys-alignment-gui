"""Application commands for cheap alignment document checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointCleared,
    AutosaveCheckpointInspected,
    AutosaveCheckpointRead,
    AutosaveCheckpointRecovered,
    AutosaveCheckpointRestored,
    AutosaveCheckpointSkippedKey,
    AutosaveCheckpointWritten,
)
from ephys_alignment_gui.core.alignment_events import AutosaveRecovered
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.core.document_snapshot import AlignmentDocumentSnapshot
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.workflow import Failed, Ok

AUTOSAVE_DIRECTORY_NAME = "autosave"
AUTOSAVE_DOCUMENT_FILENAME = "alignment_document.json"


@dataclass
class AutosaveCheckpointCommandHandler:
    """Write, read, restore, and clear document-only autosave checkpoints."""

    controller: AlignmentController
    input_dataset_provider: Callable[[], Any | None] | None = None
    events: EventBus | None = None

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

    def write_checkpoint_if_available(self) -> AutosaveCheckpointWritten | Ok | Failed:
        """Write a checkpoint if the document already has an output package."""
        if self.controller.document.output_package_directory is None:
            return Ok()
        return self.write_checkpoint()

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

    def inspect_checkpoint(
        self,
        path: Path | None = None,
    ) -> AutosaveCheckpointInspected | Failed:
        """Read and summarize a checkpoint without mutating live state."""
        read = self.read_checkpoint(path)
        if isinstance(read, Failed):
            return read
        snapshot = read.snapshot
        skipped_keys = self._skipped_checkpoint_keys(snapshot)
        recoverable_count = len(snapshot.alignment_states) - len(skipped_keys)
        restored = snapshot.restore_document()
        return AutosaveCheckpointInspected(
            path=read.path,
            mouse_id=snapshot.mouse_id,
            mouse_root=_optional_path(snapshot.mouse_root),
            output_package_directory=_optional_path(
                snapshot.output_package_directory,
            ),
            selected_alignment_key=(
                snapshot.selected_alignment_key.to_key()
                if snapshot.selected_alignment_key is not None
                else None
            ),
            alignment_state_count=len(snapshot.alignment_states),
            saveable_alignment_count=len(restored.saveable_alignment_items()),
            dirty_alignment_count=len(restored.dirty_alignment_states()),
            recoverable_alignment_count=recoverable_count,
            skipped_keys=tuple(skipped_keys.values()),
        )

    def recover_checkpoint(
        self,
        path: Path | None = None,
        *,
        require_input_dataset: bool = True,
        write_backup: bool = True,
    ) -> AutosaveCheckpointRecovered | Failed:
        """Validate and restore a checkpoint as a GUI recovery transaction."""
        read = self.read_checkpoint(path)
        if isinstance(read, Failed):
            return read

        input_dataset = self._input_dataset()
        if require_input_dataset and input_dataset is None:
            return Failed("Load the matching mouse root before recovering autosave.")
        mismatch = self._mouse_id_mismatch(read.snapshot, input_dataset)
        if mismatch is not None:
            return mismatch

        skipped_by_key = self._skipped_checkpoint_keys(read.snapshot)
        recovered_snapshot = self._filtered_recovery_snapshot(
            read.snapshot,
            skipped_by_key,
            input_dataset,
        )
        if not recovered_snapshot.alignment_states:
            return Failed(
                "No autosave alignment states match the loaded input dataset."
            )

        backup_path, warnings = self._write_pre_restore_backup(
            read.path,
            write_backup=write_backup,
        )
        recovered_snapshot.restore_into(self.controller.document)
        self._normalize_recovered_document(input_dataset)

        result = AutosaveCheckpointRecovered(
            path=read.path,
            backup_path=backup_path,
            selected_alignment_key=self.controller.document.selected_alignment_key,
            restored_alignment_count=len(recovered_snapshot.alignment_states),
            skipped_keys=tuple(skipped_by_key.values()),
            warnings=warnings,
        )
        if self.events is not None:
            self.events.emit(
                AutosaveRecovered(
                    path=result.path,
                    selected_key=result.selected_alignment_key,
                    restored_alignment_count=result.restored_alignment_count,
                    skipped_alignment_count=len(result.skipped_keys),
                    backup_path=result.backup_path,
                    warnings=result.warnings,
                )
            )
        return result

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

    def _input_dataset(self) -> Any | None:
        if self.input_dataset_provider is None:
            return None
        return self.input_dataset_provider()

    def _mouse_id_mismatch(
        self,
        snapshot: AlignmentDocumentSnapshot,
        input_dataset: Any | None,
    ) -> Failed | None:
        if input_dataset is None or not snapshot.mouse_id:
            return None
        loaded_mouse_id = getattr(input_dataset, "mouse_id", None)
        if loaded_mouse_id and loaded_mouse_id != snapshot.mouse_id:
            return Failed(
                "Autosave checkpoint mouse "
                f"{snapshot.mouse_id!r} does not match loaded mouse "
                f"{loaded_mouse_id!r}."
            )
        return None

    def _skipped_checkpoint_keys(
        self,
        snapshot: AlignmentDocumentSnapshot,
    ) -> dict[AlignmentKey, AutosaveCheckpointSkippedKey]:
        input_dataset = self._input_dataset()
        if input_dataset is None:
            return {}

        skipped: dict[AlignmentKey, AutosaveCheckpointSkippedKey] = {}
        for state_snapshot in snapshot.alignment_states:
            key = state_snapshot.key.to_key()
            reason = _invalid_key_reason(input_dataset, key)
            if reason is None:
                continue
            skipped[key] = AutosaveCheckpointSkippedKey(key=key, reason=reason)
        return skipped

    def _filtered_recovery_snapshot(
        self,
        snapshot: AlignmentDocumentSnapshot,
        skipped_by_key: dict[AlignmentKey, AutosaveCheckpointSkippedKey],
        input_dataset: Any | None,
    ) -> AlignmentDocumentSnapshot:
        valid_states = tuple(
            state_snapshot
            for state_snapshot in snapshot.alignment_states
            if state_snapshot.key.to_key() not in skipped_by_key
        )
        valid_keys = {state_snapshot.key.to_key() for state_snapshot in valid_states}
        selected_key = (
            snapshot.selected_alignment_key.to_key()
            if snapshot.selected_alignment_key is not None
            else None
        )
        if selected_key not in valid_keys:
            selected_key = _first_recoverable_key(valid_states)

        selected_key_snapshot = None
        selected_recording = None
        selected_probe = None
        selected_shank = 0
        if selected_key is not None:
            selected_key_snapshot = type(snapshot.alignment_states[0].key).from_key(
                selected_key
            )
            selected_recording = selected_key.recording_id
            selected_probe = _probe_name_for_key(input_dataset, selected_key)
            selected_shank = selected_key.shank_idx

        return replace(
            snapshot,
            selected_recording=selected_recording,
            selected_probe=selected_probe,
            selected_shank=selected_shank,
            selected_alignment_key=selected_key_snapshot,
            channel_info_loaded=False,
            data_loaded=False,
            dirty=True,
            alignment_states=valid_states,
        )

    def _normalize_recovered_document(self, input_dataset: Any | None) -> None:
        document = self.controller.document
        document.channel_info_loaded = False
        document.data_loaded = False
        document.dirty = True
        key = document.selected_alignment_key
        if key is None:
            document.selected_recording = None
            document.selected_probe = None
            document.selected_shank = 0
            document.output_directory = None
            return
        document.selected_recording = key.recording_id
        document.selected_probe = _probe_name_for_key(input_dataset, key)
        document.selected_shank = key.shank_idx
        if document.output_package_directory is not None and document.selected_probe:
            document.output_directory = (
                document.output_package_directory
                / key.recording_id
                / document.selected_probe
            )

    def _write_pre_restore_backup(
        self,
        checkpoint_path: Path,
        *,
        write_backup: bool,
    ) -> tuple[Path | None, tuple[str, ...]]:
        if not write_backup or not _document_has_recovery_relevant_state(
            self.controller.document
        ):
            return None, ()

        backup_path = checkpoint_path.with_name("alignment_document.pre_restore.json")
        try:
            AlignmentDocumentSnapshot.from_document(
                self.controller.document,
            ).write_json(backup_path)
        except OSError as exc:
            return None, (f"Failed to write pre-restore backup: {exc}",)
        return backup_path, ()


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _invalid_key_reason(input_dataset: Any, key: AlignmentKey) -> str | None:
    try:
        probe = input_dataset.probe_for_stream_key(
            key.recording_id,
            key.ephys_collection,
        )
    except Exception as exc:
        return str(exc)
    num_shanks = int(getattr(probe, "num_shanks", 1))
    if key.shank_idx >= num_shanks:
        return f"shank {key.shank_idx + 1} exceeds probe shank count {num_shanks}"
    return None


def _first_recoverable_key(state_snapshots: tuple[Any, ...]) -> AlignmentKey | None:
    for state_snapshot in state_snapshots:
        if state_snapshot.active_alignment is not None:
            return state_snapshot.key.to_key()
    if not state_snapshots:
        return None
    return state_snapshots[0].key.to_key()


def _probe_name_for_key(input_dataset: Any | None, key: AlignmentKey) -> str:
    if input_dataset is None:
        return key.ephys_collection
    try:
        probe = input_dataset.probe_for_stream_key(
            key.recording_id,
            key.ephys_collection,
        )
    except Exception:
        return key.ephys_collection
    return str(getattr(probe, "probe_name", key.ephys_collection))


def _document_has_recovery_relevant_state(document: AlignmentDocument) -> bool:
    return bool(
        document.alignment_states
        or document.selected_alignment_key is not None
        or document.output_package_directory is not None
    )
