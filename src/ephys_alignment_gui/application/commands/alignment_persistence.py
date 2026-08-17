"""App-level alignment history and output persistence commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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
from ephys_alignment_gui.application.save_runtime_dependencies import (
    plan_save_runtime_dependencies,
)
from ephys_alignment_gui.application.save_runtime_rehydration import (
    SaveRuntimeRehydrated,
    SaveRuntimeRehydrationCancelled,
    SaveRuntimeRehydrationPlan,
    SaveRuntimeRehydrator,
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
    ChannelOutputIdentity,
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
from ephys_alignment_gui.io.load_data_job import (
    LoadDataCancelToken,
    LoadDataProgressCallback,
)
from ephys_alignment_gui.runtime.session import SessionRuntime
from ephys_alignment_gui.services.alignment_derived_data import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.services.alignment_repository import (
    AlignmentHistory,
    AlignmentRepository,
    LoadedAlignmentPackage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentSaveInput:
    """Prepared runtime/document data needed to persist one alignment key."""

    state: AlignmentState
    output_input: AlignmentOutputInput
    output_metadata: AlignmentOutputMetadata
    output_directory: Path
    multi_shank: bool


@dataclass
class AlignmentPersistenceCommandHandler:
    """Coordinate alignment history loading and visited-output persistence."""

    controller: AlignmentController
    data_context: AlignmentDataContext
    runtime: SessionRuntime
    derived_data_service: AlignmentDerivedDataService
    alignment_repository: AlignmentRepository
    output_builder: Any
    events: EventBus
    save_runtime_rehydrator: SaveRuntimeRehydrator | None = None
    autosave_checkpoints: AutosaveCheckpointCommandHandler | None = None

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
        """Return whether edited alignment outputs can be saved."""
        return self.controller.can_save_alignment_output()

    def save_edited_alignment_outputs(
        self,
        *,
        use_docdb: bool,
        rehydrate_missing: bool = True,
    ) -> EditedAlignmentOutputsSaved | Blocked | Failed:
        """Persist outputs for every dirty alignment state in the document."""
        prepared = self.prepare_edited_alignment_save(
            use_docdb=use_docdb,
            rehydrate_missing=rehydrate_missing,
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
        rehydrate_missing: bool = True,
    ) -> PreparedAlignmentSave | Blocked | Failed:
        """Prepare immutable save-job inputs on the application thread."""
        ready = self.controller.can_save_alignment_output()
        if isinstance(ready, Blocked):
            return ready

        target_keys = self._dirty_alignment_keys()
        if not target_keys:
            return self._save_failed("No edited alignments are ready to save")
        self._emit_save_progress_started(target_keys)

        save_inputs = self._dirty_alignment_output_inputs(
            rehydrate_missing=rehydrate_missing,
        )
        if isinstance(save_inputs, Failed):
            return self._save_failed(save_inputs.message)
        if not save_inputs:
            return self._save_failed("No edited alignments are ready to save")

        targets: list[PreparedAlignmentSaveTarget] = []
        for key, save_input in save_inputs.items():
            state = save_input.state
            alignment = state.active_alignment
            alignments_to_save = state.alignments
            if alignment is not None:
                _, alignments_to_save = state.with_alignment_added(
                    alignment.feature,
                    alignment.track,
                )
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
        output_inputs = {
            target.key: target.output_input
            for target in prepared.targets
        }
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
                    f"{len(output_inputs)} edited alignment(s)..."
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
                    "Built CCF output dictionaries for "
                    f"{len(output_inputs)} edited alignment(s)."
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

    def prepare_save_runtime_rehydration(
        self,
    ) -> SaveRuntimeRehydrationPlan | Ok | Failed:
        """Return save-runtime reload work needed before saving dirty outputs."""
        runtime_plan = plan_save_runtime_dependencies(
            document=self.controller.document,
            data_context=self.data_context,
            runtime=self.runtime,
        )
        if not runtime_plan.unavailable:
            return Ok()
        if self.save_runtime_rehydrator is None:
            return Failed(runtime_plan.failure_message() or "Cannot save alignment.")
        return self.save_runtime_rehydrator.prepare_rehydration(runtime_plan)

    def run_save_runtime_rehydration(
        self,
        plan: SaveRuntimeRehydrationPlan,
        *,
        progress: LoadDataProgressCallback | None = None,
        cancel_token: LoadDataCancelToken | None = None,
    ) -> SaveRuntimeRehydrated | SaveRuntimeRehydrationCancelled | Failed:
        """Reload missing save runtimes without publishing app events."""
        if self.save_runtime_rehydrator is None:
            return Failed("No save-runtime rehydrator is configured.")
        return self.save_runtime_rehydrator.run_rehydration_plan(
            plan,
            progress=progress,
            cancel_token=cancel_token,
        )

    def publish_save_runtime_rehydration_result(
        self,
        result: SaveRuntimeRehydrated | SaveRuntimeRehydrationCancelled | Failed,
    ) -> Ok | Failed:
        """Publish the terminal semantic result of save-runtime rehydration."""
        if isinstance(result, SaveRuntimeRehydrated):
            return Ok()
        if isinstance(result, SaveRuntimeRehydrationCancelled):
            return self._save_failed(
                f"Reload cancelled while saving edited alignments: {result.reason}"
            )
        return self._save_failed(result.message)

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

    def _dirty_alignment_output_inputs(
        self,
        *,
        rehydrate_missing: bool = True,
    ) -> dict[AlignmentKey, AlignmentSaveInput] | Failed:
        """Collect save inputs for every dirty document alignment state."""
        dirty_items = self._dirty_alignment_items()
        if not dirty_items:
            return {}

        runtime_plan = plan_save_runtime_dependencies(
            document=self.controller.document,
            data_context=self.data_context,
            runtime=self.runtime,
        )
        if (
            rehydrate_missing
            and runtime_plan.unavailable
            and self.save_runtime_rehydrator is not None
        ):
            rehydrated = self.save_runtime_rehydrator.rehydrate_missing(runtime_plan)
            if isinstance(rehydrated, Failed):
                return rehydrated
            runtime_plan = plan_save_runtime_dependencies(
                document=self.controller.document,
                data_context=self.data_context,
                runtime=self.runtime,
            )
        if runtime_plan.unavailable:
            return Failed(runtime_plan.failure_message() or "Cannot save alignment.")
        runtime_by_key = runtime_plan.by_key

        save_inputs: dict[AlignmentKey, AlignmentSaveInput] = {}
        total = len(dirty_items)
        for input_index, (key, state) in enumerate(dirty_items, start=1):
            alignment = state.active_alignment
            assert alignment is not None
            self._emit_save_progress_updated(
                key=key,
                phase="preparing",
                status="started",
                completed=input_index - 1,
                total=total,
                message=f"Preparing channel locations for {self._describe_key(key)}...",
            )

            dependency = runtime_by_key[key]
            stream_runtime = dependency.stream_runtime
            if stream_runtime is None:
                return Failed("Cannot save alignment: stream runtime is not loaded.")

            shank_runtime = stream_runtime.shank_runtime_by_idx.get(key.shank_idx)
            if shank_runtime is None:
                return Failed(
                    "Cannot save edited alignment for "
                    f"{key.recording_id}/{key.ephys_collection} shank "
                    f"{key.shank_idx + 1}: shank runtime is not initialized."
                )
            if shank_runtime.ephysalign is None or shank_runtime.chn_coords is None:
                return Failed(
                    "Cannot save edited alignment for "
                    f"{key.recording_id}/{key.ephys_collection} shank "
                    f"{key.shank_idx + 1}: channel geometry is not initialized."
                )

            output_directory = self._output_directory_for_key(key)
            if isinstance(output_directory, Failed):
                return output_directory

            channel_locations_ras = self.derived_data_service.compute_channel_locations(
                ephysalign=shank_runtime.ephysalign,
                feature=alignment.feature,
                track=alignment.track,
            )
            save_inputs[key] = AlignmentSaveInput(
                state=state,
                output_input=AlignmentOutputInput(
                    channel_locations_ras=channel_locations_ras,
                    channel_coordinates=shank_runtime.chn_coords,
                    channel_identity=self._channel_identity_for_collection(
                        shank_runtime.collection
                    ),
                ),
                output_metadata=self._output_metadata_for_key(
                    key,
                    stream_runtime=stream_runtime,
                    probe=self._probe_for_output_metadata(key, dependency.probe),
                ),
                output_directory=output_directory,
                multi_shank=self._stream_is_multi_shank(stream_runtime),
            )
            self._emit_save_progress_updated(
                key=key,
                phase="preparing",
                status="completed",
                completed=input_index,
                total=total,
                message=f"Prepared channel locations for {self._describe_key(key)}.",
            )
        return save_inputs

    def _dirty_alignment_keys(self) -> tuple[AlignmentKey, ...]:
        """Return dirty alignment keys that have active alignment data to save."""
        return tuple(key for key, _state in self._dirty_alignment_items())

    def _dirty_alignment_items(self) -> tuple[tuple[AlignmentKey, AlignmentState], ...]:
        """Return sorted dirty alignment states with active alignment data."""
        return tuple(
            (key, state)
            for key, state in sorted(
                self.controller.document.dirty_alignment_states().items(),
                key=lambda item: (
                    item[0].recording_id,
                    item[0].ephys_collection,
                    item[0].shank_idx,
                ),
            )
            if state.active_alignment is not None
        )

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
                "Choose an output root before saving edited alignments from "
                "non-active streams."
            )

        try:
            probe = self.data_context.probe_for_stream_key(
                key.recording_id,
                key.ephys_collection,
            )
        except Exception as exc:
            return Failed(
                f"Cannot resolve output directory for edited alignment: {exc}"
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

    @staticmethod
    def _channel_identity_for_collection(collection: Any) -> ChannelOutputIdentity:
        """Return channel identity aligned with a shank/channel collection view."""
        raw_ind = collection.raw_ind
        if raw_ind is None:
            raw_ind = np.asarray(collection.rows, dtype=int).copy()

        shank_idx = collection.shank_indices
        if shank_idx is None:
            shank_idx = np.full(
                np.asarray(collection.rows).shape,
                collection.shank_idx,
                dtype=int,
            )

        return ChannelOutputIdentity(
            raw_ind=raw_ind,
            contact_id=collection.contact_ids,
            shank_idx=shank_idx,
        )

    @staticmethod
    def _output_metadata_for_key(
        key: AlignmentKey,
        *,
        stream_runtime: Any,
        probe: Any | None,
    ) -> AlignmentOutputMetadata:
        """Return stable probe/shank metadata for one output sidecar."""
        stream = getattr(stream_runtime, "stream", None)
        logical_probe = (
            getattr(probe, "logical_probe", None)
            or getattr(probe, "probe_name", None)
            or getattr(stream, "logical_probe", None)
            or key.ephys_collection
        )
        probe_id = getattr(probe, "probe_id", None) or getattr(stream, "probe_id", None)
        n_shanks = getattr(
            probe,
            "num_shanks",
            None,
        ) or getattr(
            stream,
            "n_shanks",
            1,
        )
        return AlignmentOutputMetadata(
            recording_id=key.recording_id,
            ephys_collection=key.ephys_collection,
            logical_probe=str(logical_probe),
            shank_idx=key.shank_idx,
            n_shanks=int(n_shanks),
            probe_id=probe_id,
        )

    def _probe_for_output_metadata(
        self,
        key: AlignmentKey,
        probe: Any | None,
    ) -> Any | None:
        """Return probe metadata for a save key, falling back to active probe info."""
        if probe is not None:
            return probe
        active_probe = self.data_context.probe_info
        if (
            active_probe is not None
            and active_probe.recording_id == key.recording_id
            and active_probe.ephys_collection == key.ephys_collection
        ):
            return active_probe
        return None

    @staticmethod
    def _stream_is_multi_shank(stream_runtime: Any) -> bool:
        stream = getattr(stream_runtime, "stream", None)
        if stream is not None and hasattr(stream, "n_shanks"):
            return stream.n_shanks > 1
        return len(stream_runtime.visited_shank_runtimes()) > 1

    def _active_or_given_shank(self, shank_idx: int | None) -> int:
        if shank_idx is not None:
            return shank_idx
        return self.controller.document.selected_shank

    def _alignment_repository(self) -> AlignmentRepository:
        return self.alignment_repository

    def _output_builder(self) -> Any | None:
        return self.output_builder

    def _emit_save_progress_started(
        self,
        target_keys: tuple[AlignmentKey, ...],
    ) -> None:
        self.events.emit(
            SaveProgressStarted(
                targets=target_keys,
                message=(
                    f"Saving {len(target_keys)} edited alignment"
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
