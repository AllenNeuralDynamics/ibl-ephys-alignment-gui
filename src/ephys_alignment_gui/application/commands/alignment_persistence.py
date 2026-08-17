"""App-level alignment history and output persistence commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.alignment_save_input_factory import (
    AlignmentSaveInput,
    AlignmentSaveInputFactory,
    AlignmentSaveInputFactoryError,
)
from ephys_alignment_gui.application.alignment_save_job import (
    AlignmentSaveCancelToken,
    AlignmentSaveJobCancelled,
    AlignmentSaveJobCompleted,
    PreparedAlignmentSave,
    PreparedAlignmentSaveTarget,
)
from ephys_alignment_gui.application.commands.autosave import (
    AutosaveCheckpointCommandHandler,
)
from ephys_alignment_gui.application.output_paths import (
    alignment_output_package_directory,
    probe_alignment_output_directory,
)
from ephys_alignment_gui.application.results import (
    AlignmentChoicesUpdated,
    EditedAlignmentOutputsSaved,
    PreviousAlignmentSelected,
)
from ephys_alignment_gui.application.results.alignment_persistence import (
    AlignmentOutputBuilt,
    AlignmentOutputsSaved,
    NoPreviousAlignments,
    PreviousAlignmentPackageLoaded,
)
from ephys_alignment_gui.core.alignment_events import (
    PreviousAlignmentLoadFailed,
    PreviousAlignmentsLoaded,
    PreviousAlignmentsUnavailable,
    SaveCancelled,
    SaveCompleted,
    SaveDocDbStatus,
    SaveFailed,
    SaveProgressPhase,
    SaveProgressStarted,
    SaveProgressStatus,
    SaveProgressUpdated,
)
from ephys_alignment_gui.core.alignment_output import (
    AlignmentOutputInput,
    AlignmentOutputMetadata,
)
from ephys_alignment_gui.core.alignment_state import (
    LEGACY_AUTO_ALIGNMENT_LABEL,
    AlignmentState,
)
from ephys_alignment_gui.core.controller import (
    AlignmentController,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.workflow import Blocked, Failed, Ok
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.services.alignment_derived_data import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.services.alignment_repository import (
    AlignmentHistory,
    AlignmentRepository,
    LoadedAlignmentPackage,
)

logger = logging.getLogger(__name__)


@dataclass
class AlignmentPersistenceCommandHandler:
    """Coordinate alignment history loading and visited-output persistence."""

    controller: AlignmentController
    data_context: AlignmentDataContext
    derived_data_service: AlignmentDerivedDataService
    alignment_repository: AlignmentRepository
    output_builder: Any
    events: EventBus
    autosave_checkpoints: AutosaveCheckpointCommandHandler | None = None
    save_input_factory: AlignmentSaveInputFactory | None = None

    def can_load_previous_alignments(self) -> Ok | Failed:
        """Return whether previous alignments can be loaded."""
        if self.data_context.mouse_root is None:
            return Failed("No mouse root loaded. Please select a mouse root first.")
        return Ok()

    def load_previous_alignments(
        self,
        *,
        folder: Path | None,
        use_docdb: bool,
        shank_idx: int | None = None,
    ) -> (
        AlignmentChoicesUpdated
        | NoPreviousAlignments
        | PreviousAlignmentPackageLoaded
        | Failed
    ):
        """Load and store previous alignments for a document-selected shank."""
        if folder is not None:
            package_result = self._load_previous_alignment_package(folder)
            if package_result is not None:
                return package_result

        target_shank = self._active_or_given_shank(shank_idx)
        ready = self.controller.can_load_previous_alignments()
        if isinstance(ready, Failed):
            self.events.emit(
                PreviousAlignmentLoadFailed(
                    shank_idx=target_shank,
                    message=ready.message,
                )
            )
            return ready
        probe = self.data_context.probe_info
        assert probe is not None

        try:
            loaded = self._alignment_repository().load_previous_alignments(
                folder=folder,
                recording_id=probe.recording_id,
                probe_name=probe.probe_name,
                shank_idx=target_shank,
                n_shanks=self.data_context.n_shanks,
                use_docdb=use_docdb,
            )
        except Exception as exc:
            return self._previous_alignment_load_failed(
                target_shank,
                f"Failed to load previous alignments: {exc}",
            )

        if loaded is None:
            self.events.emit(PreviousAlignmentsUnavailable(shank_idx=target_shank))
            return NoPreviousAlignments()
        result = self.controller.set_previous_alignments(
            loaded.alignments,
            shank_idx=target_shank,
        )
        if isinstance(result, AlignmentChoicesUpdated):
            self.events.emit(
                PreviousAlignmentsLoaded(
                    shank_idx=target_shank,
                    choices=tuple(result.choices),
                )
            )
            self._write_autosave_checkpoint("previous alignment load")
        elif isinstance(result, Failed):
            self.events.emit(
                PreviousAlignmentLoadFailed(
                    shank_idx=target_shank,
                    message=result.message,
                )
            )
        return result

    def _load_previous_alignment_package(
        self,
        folder: Path,
    ) -> PreviousAlignmentPackageLoaded | NoPreviousAlignments | Failed | None:
        try:
            package = self._alignment_repository().load_previous_alignment_package(
                folder=folder,
            )
        except Exception as exc:
            return self._previous_alignment_load_failed(
                None,
                f"Failed to load previous alignment package: {exc}",
            )

        if not package.histories:
            return None
        return self._import_previous_alignment_package(package)

    def _import_previous_alignment_package(
        self,
        package: LoadedAlignmentPackage,
    ) -> PreviousAlignmentPackageLoaded | NoPreviousAlignments | Failed:
        mouse_root = self.data_context.mouse_root
        if mouse_root is None:
            return self._previous_alignment_load_failed(
                None,
                "No mouse root loaded. Please select a mouse root first.",
            )

        loaded: dict[AlignmentKey, AlignmentChoicesUpdated] = {}
        for (recording_id, probe_name, shank_idx), history in package.histories.items():
            try:
                probe = mouse_root.get_probe(recording_id, probe_name)
            except Exception:
                logger.warning(
                    "Skipping previous alignment for unknown probe %s/%s",
                    recording_id,
                    probe_name,
                    exc_info=True,
                )
                continue

            key = AlignmentKey(
                recording_id=probe.recording_id,
                ephys_collection=probe.ephys_collection,
                shank_idx=shank_idx,
            )
            loaded[key] = self.controller.import_previous_alignments_for_key(
                key,
                history.alignments,
            )

        if not loaded:
            return self._previous_alignment_load_failed(
                None,
                "No previous alignments in the selected package matched the loaded mouse root.",
            )

        active_choices = self._emit_active_package_choices(loaded)
        self._write_autosave_checkpoint("previous alignment package import")
        return PreviousAlignmentPackageLoaded(
            loaded_count=len(loaded),
            loaded_keys=tuple(
                sorted(
                    loaded,
                    key=lambda key: (
                        key.recording_id,
                        key.ephys_collection,
                        key.shank_idx,
                    ),
                )
            ),
            active_choices=active_choices,
        )

    def _emit_active_package_choices(
        self,
        loaded: dict[AlignmentKey, AlignmentChoicesUpdated],
    ) -> list[str] | None:
        active_key = self.controller.document.selected_alignment_key
        if active_key is None or active_key not in loaded:
            return None

        choices = loaded[active_key].choices
        self.events.emit(
            PreviousAlignmentsLoaded(
                shank_idx=active_key.shank_idx,
                choices=tuple(choices),
                auto_select=False,
            )
        )
        return choices

    def _previous_alignment_load_failed(
        self,
        shank_idx: int | None,
        message: str,
    ) -> Failed:
        self.events.emit(
            PreviousAlignmentLoadFailed(
                shank_idx=shank_idx,
                message=message,
            )
        )
        return Failed(message)

    def select_previous_alignment(
        self,
        idx: int,
        *,
        shank_idx: int | None = None,
    ) -> PreviousAlignmentSelected | Failed:
        """Select a previous/original alignment on a document-selected shank."""
        result = self.controller.select_previous_alignment(
            idx,
            shank_idx=self._active_or_given_shank(shank_idx),
            mark_changed=True,
        )
        if isinstance(result, PreviousAlignmentSelected):
            self._write_autosave_checkpoint("previous alignment selection")
        return result

    def can_save_alignment_output(self) -> Ok | Blocked:
        """Return whether alignment outputs can be saved."""
        return self.controller.can_save_alignment_output()

    def save_edited_alignment_outputs(
        self,
        *,
        use_docdb: bool,
    ) -> EditedAlignmentOutputsSaved | Blocked | Failed:
        """Persist outputs for every saveable alignment state in the document."""
        prepared = self.prepare_edited_alignment_save(
            use_docdb=use_docdb,
        )
        if isinstance(prepared, Blocked | Failed):
            return prepared

        result = self.run_prepared_alignment_save(
            prepared,
            progress=self._emit_save_progress_event,
        )
        return self.publish_prepared_alignment_save_result(prepared, result)

    def prepare_edited_alignment_save(
        self,
        *,
        use_docdb: bool,
    ) -> PreparedAlignmentSave | Blocked | Failed:
        """Prepare immutable save-job inputs on the application thread."""
        ready = self.controller.can_save_alignment_output()
        if isinstance(ready, Blocked):
            return ready

        target_keys = self._saveable_alignment_keys()
        if not target_keys:
            return self._save_failed("No alignment outputs are ready to save")
        self._emit_save_progress_started(target_keys)

        save_inputs = self._saveable_alignment_output_inputs()
        if isinstance(save_inputs, Failed):
            return self._save_failed(save_inputs.message)
        if not save_inputs:
            return self._save_failed("No alignment outputs are ready to save")

        targets: list[PreparedAlignmentSaveTarget] = []
        for key, save_input in save_inputs.items():
            state = save_input.state
            alignments_to_save = state.alignment_history_for_save()
            targets.append(
                PreparedAlignmentSaveTarget(
                    key=key,
                    state=state,
                    output_input=save_input.output_input,
                    output_metadata=save_input.output_metadata,
                    output_directory=save_input.output_directory,
                    multi_shank=save_input.multi_shank,
                    alignments_to_save=alignments_to_save,
                )
            )

        return PreparedAlignmentSave(tuple(targets), use_docdb=use_docdb)

    def run_prepared_alignment_save(
        self,
        prepared: PreparedAlignmentSave,
        *,
        progress: Any | None = None,
        cancel_token: AlignmentSaveCancelToken | None = None,
    ) -> AlignmentSaveJobCompleted | AlignmentSaveJobCancelled | Failed:
        """Build CCF outputs and write files without touching document state."""
        cancel_token = cancel_token or AlignmentSaveCancelToken()
        output_inputs = {target.key: target.output_input for target in prepared.targets}
        cancelled = self._cancelled_save(
            cancel_token,
            progress,
            phase="building_outputs",
            completed=0,
            total=len(output_inputs),
        )
        if cancelled is not None:
            return cancelled
        self._emit_or_callback_save_progress(
            progress,
            SaveProgressUpdated(
                key=None,
                phase="building_outputs",
                status="started",
                completed=0,
                total=len(output_inputs),
                message=(
                    "Batching CCF transform points for "
                    f"{len(output_inputs)} alignment output(s)..."
                ),
            ),
        )
        outputs = self._build_alignment_outputs(
            output_inputs,
            multi_shank_by_key={
                target.key: target.multi_shank for target in prepared.targets
            },
        )
        if isinstance(outputs, Failed):
            return outputs
        cancelled = self._cancelled_save(
            cancel_token,
            progress,
            phase="building_outputs",
            completed=len(output_inputs),
            total=len(output_inputs),
        )
        if cancelled is not None:
            return cancelled
        self._emit_or_callback_save_progress(
            progress,
            SaveProgressUpdated(
                key=None,
                phase="building_outputs",
                status="completed",
                completed=len(output_inputs),
                total=len(output_inputs),
                message=(
                    "Built alignment output dictionaries for "
                    f"{len(output_inputs)} alignment output(s)."
                ),
            ),
        )
        missing_ccf_outputs = tuple(
            key for key, output in outputs.items() if not output.ccf_channel_results
        )
        if missing_ccf_outputs:
            logger.warning(
                "CCF channel coordinate output is unavailable for %d alignment "
                "output(s); saving anatomical outputs without CCF coordinates",
                len(missing_ccf_outputs),
            )
            self._emit_or_callback_save_progress(
                progress,
                SaveProgressUpdated(
                    key=None,
                    phase="building_outputs",
                    status="warning",
                    completed=len(output_inputs),
                    total=len(output_inputs),
                    message=(
                        "Warning: CCF channel coordinates could not be generated for "
                        f"{len(missing_ccf_outputs)} alignment output"
                        f"{'' if len(missing_ccf_outputs) == 1 else 's'}. "
                        "Anatomical channel locations and alignments will still be "
                        "saved."
                    ),
                ),
            )

        logger.info("Saving output files to results folder...")
        saved_outputs: dict[AlignmentKey, AlignmentOutputsSaved] = {}
        total_outputs = len(outputs)
        target_by_key = {target.key: target for target in prepared.targets}
        for output_index, (key, output) in enumerate(outputs.items(), start=1):
            cancelled = self._cancelled_save(
                cancel_token,
                progress,
                phase="writing_files",
                completed=output_index - 1,
                total=total_outputs,
            )
            if cancelled is not None:
                return cancelled
            target = target_by_key[key]
            self._emit_or_callback_save_progress(
                progress,
                SaveProgressUpdated(
                    key=key,
                    phase="writing_files",
                    status="started",
                    completed=output_index - 1,
                    total=total_outputs,
                    message=f"Writing output files for {self._describe_key(key)}...",
                ),
            )
            saved = self._save_alignment_output(
                output,
                target.alignments_to_save,
                key.shank_idx,
                prepared.use_docdb,
                output_directory=target.output_directory,
                output_metadata=target.output_metadata,
            )
            if isinstance(saved, Failed):
                return saved
            cancelled = self._cancelled_save(
                cancel_token,
                progress,
                phase="writing_files",
                completed=output_index,
                total=total_outputs,
            )
            if cancelled is not None:
                return cancelled
            saved_outputs[key] = saved
            self._emit_or_callback_save_progress(
                progress,
                SaveProgressUpdated(
                    key=key,
                    phase="writing_files",
                    status="completed",
                    completed=output_index,
                    total=total_outputs,
                    message=f"Wrote output files for {self._describe_key(key)}.",
                ),
            )
        return AlignmentSaveJobCompleted(saved_outputs=saved_outputs)

    def publish_prepared_alignment_save_result(
        self,
        prepared: PreparedAlignmentSave,
        job_result: AlignmentSaveJobCompleted | AlignmentSaveJobCancelled | Failed,
    ) -> EditedAlignmentOutputsSaved | AlignmentSaveJobCancelled | Failed:
        """Publish a save job result and update document save state."""
        if isinstance(job_result, Failed):
            return self._save_failed(job_result.message)
        if isinstance(job_result, AlignmentSaveJobCancelled):
            return self._save_cancelled(job_result.reason)

        for target in prepared.targets:
            if target.key not in job_result.saved_outputs:
                continue
            target.state.set_alignments(target.alignments_to_save)
            target.state.mark_saved()
        self.controller.document.dirty = self.controller.document.has_unsaved_alignments

        active_choices: list[str] | None = None
        choices = self.controller.active_alignment_choices(
            self._active_or_given_shank(None)
        )
        if isinstance(choices, AlignmentChoicesUpdated):
            active_choices = choices.choices
        elif isinstance(choices, Failed):
            logger.error(choices.message)

        saved_outputs = job_result.saved_outputs
        result = EditedAlignmentOutputsSaved(
            saved_count=len(saved_outputs),
            saved_outputs=saved_outputs,
            active_choices=active_choices,
        )
        self._clear_autosave_checkpoint_after_save()
        self.events.emit(
            SaveCompleted(
                saved_count=result.saved_count,
                active_choices=(
                    tuple(result.active_choices)
                    if result.active_choices is not None
                    else None
                ),
                docdb_statuses=self._docdb_statuses(result),
            )
        )
        return result

    def _save_failed(self, message: str) -> Failed:
        self.controller.document.dirty = self.controller.document.has_unsaved_alignments
        self._emit_save_failed(message)
        return Failed(message)

    def _save_cancelled(self, reason: str) -> AlignmentSaveJobCancelled:
        self.controller.document.dirty = self.controller.document.has_unsaved_alignments
        message = f"Save cancelled: {reason}"
        self.events.emit(SaveCancelled(reason=reason, message=message))
        return AlignmentSaveJobCancelled(reason=reason)

    def _emit_save_failed(self, message: str) -> None:
        self.events.emit(SaveFailed(message=message))

    def _write_autosave_checkpoint(self, action: str) -> None:
        if self.autosave_checkpoints is None:
            return
        result = self.autosave_checkpoints.write_checkpoint_if_available()
        if isinstance(result, Failed):
            logger.warning(
                "Autosave checkpoint failed after %s: %s",
                action,
                result.message,
            )

    def _clear_autosave_checkpoint_after_save(self) -> None:
        if self.autosave_checkpoints is None:
            return
        result = self.autosave_checkpoints.clear_checkpoint()
        if isinstance(result, Failed):
            logger.warning(
                "Failed to clear autosave checkpoint after full Save: %s",
                result.message,
            )

    @staticmethod
    def _docdb_statuses(
        result: EditedAlignmentOutputsSaved,
    ) -> tuple[SaveDocDbStatus, ...]:
        statuses: list[SaveDocDbStatus] = []
        for saved in result.saved_outputs.values():
            if saved.saved.docdb_probe_name is None:
                continue
            statuses.append(
                SaveDocDbStatus(
                    probe_name=saved.saved.docdb_probe_name,
                    error=saved.saved.docdb_error,
                )
            )
        return tuple(statuses)

    def _saveable_alignment_output_inputs(
        self,
    ) -> dict[AlignmentKey, AlignmentSaveInput] | Failed:
        """Collect save inputs for every saveable document alignment state."""
        saveable_items = self._saveable_alignment_items()
        if not saveable_items:
            return {}

        save_inputs: dict[AlignmentKey, AlignmentSaveInput] = {}
        total = len(saveable_items)
        for input_index, (key, state) in enumerate(saveable_items, start=1):
            self._emit_save_progress_updated(
                key=key,
                phase="preparing",
                status="started",
                completed=input_index - 1,
                total=total,
                message=f"Preparing channel locations for {self._describe_key(key)}...",
            )

            output_directory = self._output_directory_for_key(key)
            if isinstance(output_directory, Failed):
                return output_directory

            factory = self._save_input_factory()
            if factory is None:
                return Failed("No alignment save input factory is configured.")
            try:
                save_inputs[key] = factory.build(
                    key=key,
                    state=state,
                    output_directory=output_directory,
                )
            except AlignmentSaveInputFactoryError as exc:
                return Failed(f"Failed to prepare alignment save input: {exc}")
            self._emit_save_progress_updated(
                key=key,
                phase="preparing",
                status="completed",
                completed=input_index,
                total=total,
                message=f"Prepared channel locations for {self._describe_key(key)}.",
            )
        return save_inputs

    def _saveable_alignment_keys(self) -> tuple[AlignmentKey, ...]:
        """Return alignment keys that have active output data to save."""
        return tuple(key for key, _state in self._saveable_alignment_items())

    def _saveable_alignment_items(
        self,
    ) -> tuple[tuple[AlignmentKey, AlignmentState], ...]:
        """Return sorted alignment states with active output data."""
        return self.controller.document.saveable_alignment_items()

    def _build_alignment_outputs(
        self,
        alignments: dict[AlignmentKey, AlignmentOutputInput],
        *,
        multi_shank_by_key: dict[AlignmentKey, bool] | None = None,
    ) -> dict[AlignmentKey, AlignmentOutputBuilt] | Failed:
        output_builder = self._output_builder()
        if output_builder is None:
            return Failed("No alignment output builder is configured.")
        try:
            if hasattr(output_builder, "get_alignment_results_batch"):
                batch_results = output_builder.get_alignment_results_batch(alignments)
            else:
                batch_results = {
                    key: self._get_alignment_results(output_builder, output_input)
                    for key, output_input in alignments.items()
                }
        except Exception as exc:
            return Failed(f"Failed to build alignment outputs: {exc}")

        return {
            key: AlignmentOutputBuilt(
                channel_results=channel_results,
                ccf_channel_results=ccf_channel_results,
                multi_shank=(
                    multi_shank_by_key.get(key, multi_shank)
                    if multi_shank_by_key is not None
                    else multi_shank
                ),
            )
            for key, (
                channel_results,
                ccf_channel_results,
                multi_shank,
            ) in batch_results.items()
        }

    @staticmethod
    def _get_alignment_results(
        output_builder: Any,
        output_input: AlignmentOutputInput,
    ) -> Any:
        """Call a single-alignment output builder with channel identity."""
        return output_builder.get_alignment_results(
            output_input.channel_locations_ras,
            output_input.channel_coordinates,
            output_input.channel_identity,
        )

    def _save_alignment_output(
        self,
        output: AlignmentOutputBuilt,
        alignments: AlignmentHistory,
        shank_idx: int,
        use_docdb: bool,
        *,
        output_directory: Path | None = None,
        output_metadata: AlignmentOutputMetadata,
    ) -> AlignmentOutputsSaved | Failed:
        output_directory = output_directory or self.controller.document.output_directory
        if output_directory is None:
            return Failed("Choose an output folder before saving.")

        persistable_alignments = {
            key: value
            for key, value in alignments.items()
            if key != LEGACY_AUTO_ALIGNMENT_LABEL
        }
        try:
            saved = self._alignment_repository().save_alignment_outputs(
                output_directory=output_directory,
                shank_idx=shank_idx,
                multi_shank=output.multi_shank,
                channel_results=output.channel_results,
                previous_alignments=persistable_alignments,
                ccf_channel_results=output.ccf_channel_results,
                metadata=output_metadata,
                use_docdb=use_docdb,
                output_package_directory=(
                    self.controller.document.output_package_directory
                ),
                mouse_id=self._output_package_mouse_id(),
            )
        except Exception as exc:
            return Failed(f"Failed to save alignment output: {exc}")

        return AlignmentOutputsSaved(
            saved=saved,
            previous_alignments=persistable_alignments,
        )

    def _output_directory_for_key(self, key: AlignmentKey) -> Path | Failed:
        document = self.controller.document
        active_probe = self.data_context.probe_info
        if (
            active_probe is not None
            and active_probe.recording_id == key.recording_id
            and active_probe.ephys_collection == key.ephys_collection
            and document.output_directory is not None
        ):
            return document.output_directory

        output_root = document.output_root
        if output_root is None:
            return Failed(
                "Choose an output root before saving alignment outputs from "
                "non-active streams."
            )

        try:
            probe = self.data_context.probe_for_stream_key(
                key.recording_id,
                key.ephys_collection,
            )
        except Exception as exc:
            return Failed(
                f"Cannot resolve output directory for alignment output: {exc}"
            )

        output_package_directory = self._output_package_directory_for_save(output_root)
        if isinstance(output_package_directory, Failed):
            return output_package_directory

        output_directory = probe_alignment_output_directory(
            output_package_directory,
            probe.recording_id,
            probe.probe_name,
        )
        try:
            output_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return Failed(
                f"Failed to create probe output directory {output_directory}: {exc}"
            )
        return output_directory

    def _output_package_directory_for_save(self, output_root: Path) -> Path | Failed:
        document = self.controller.document
        if document.output_package_directory is not None:
            return document.output_package_directory

        mouse_id = document.mouse_id
        if mouse_id is None and self.data_context.mouse_root is not None:
            mouse_id = self.data_context.mouse_root.mouse_id
        if mouse_id is None or str(mouse_id).strip() == "":
            return Failed(
                "Mouse ID is not loaded; cannot derive annotation output package."
            )

        output_package_directory = alignment_output_package_directory(
            output_root,
            mouse_id,
            datetime.now(),
        )
        try:
            output_package_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return Failed(
                "Failed to create annotation output package "
                f"{output_package_directory}: {exc}"
            )
        self.controller.record_output_package_directory(output_package_directory)
        return output_package_directory

    def _output_package_mouse_id(self) -> str | None:
        mouse_id = self.controller.document.mouse_id
        if mouse_id is not None:
            return mouse_id
        if self.data_context.mouse_root is not None:
            return self.data_context.mouse_root.mouse_id
        return None

    def _active_or_given_shank(self, shank_idx: int | None) -> int:
        if shank_idx is not None:
            return shank_idx
        return self.controller.document.selected_shank

    def _alignment_repository(self) -> AlignmentRepository:
        return self.alignment_repository

    def _output_builder(self) -> Any | None:
        return self.output_builder

    def _save_input_factory(self) -> AlignmentSaveInputFactory | None:
        return self.save_input_factory

    def _emit_save_progress_started(
        self,
        target_keys: tuple[AlignmentKey, ...],
    ) -> None:
        self.events.emit(
            SaveProgressStarted(
                targets=target_keys,
                message=(
                    f"Saving {len(target_keys)} alignment output"
                    f"{'' if len(target_keys) == 1 else 's'}..."
                ),
            )
        )

    def _emit_save_progress_updated(
        self,
        *,
        key: AlignmentKey | None,
        phase: SaveProgressPhase,
        status: SaveProgressStatus,
        completed: int,
        total: int,
        message: str,
    ) -> None:
        self.events.emit(
            SaveProgressUpdated(
                key=key,
                phase=phase,
                status=status,
                completed=completed,
                total=total,
                message=message,
            )
        )

    def _emit_save_progress_event(self, event: SaveProgressUpdated) -> None:
        """Emit one prepared-save progress event."""
        self.events.emit(event)

    def _emit_or_callback_save_progress(
        self,
        progress: Any | None,
        event: SaveProgressUpdated,
    ) -> None:
        """Route save-job progress either to a callback or the app event bus."""
        if progress is None:
            self._emit_save_progress_event(event)
        else:
            progress(event)

    def _cancelled_save(
        self,
        cancel_token: AlignmentSaveCancelToken,
        progress: Any | None,
        *,
        phase: SaveProgressPhase,
        completed: int,
        total: int,
    ) -> AlignmentSaveJobCancelled | None:
        """Return a cancellation result and emit one terminal progress update."""
        if not cancel_token.cancelled:
            return None
        reason = cancel_token.reason or "cancelled"
        self._emit_or_callback_save_progress(
            progress,
            SaveProgressUpdated(
                key=None,
                phase=phase,
                status="cancelled",
                completed=completed,
                total=total,
                message=f"Save cancelled: {reason}",
            ),
        )
        return AlignmentSaveJobCancelled(reason=reason)

    @staticmethod
    def _describe_key(key: AlignmentKey) -> str:
        return f"{key.recording_id}/{key.ephys_collection} shank {key.shank_idx + 1}"
