"""App-level alignment history and output persistence commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results import (
    AlignmentChoicesUpdated,
    EditedAlignmentOutputsSaved,
    PreviousAlignmentSelected,
)
from ephys_alignment_gui.application.results.alignment_persistence import (
    AlignmentOutputBuilt,
    AlignmentOutputsSaved,
    NoPreviousAlignments,
)
from ephys_alignment_gui.application.save_runtime_dependencies import (
    plan_save_runtime_dependencies,
)
from ephys_alignment_gui.application.workflow import Blocked, Failed, Ok
from ephys_alignment_gui.core.alignment_events import (
    PreviousAlignmentLoadFailed,
    PreviousAlignmentsLoaded,
    PreviousAlignmentsUnavailable,
    SaveCompleted,
    SaveDocDbStatus,
    SaveFailed,
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
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.runtime.session import SessionRuntime
from ephys_alignment_gui.services.alignment_derived_data import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.services.alignment_repository import (
    AlignmentHistory,
    AlignmentRepository,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentSaveInput:
    """Prepared runtime/document data needed to persist one alignment key."""

    state: AlignmentState
    channel_locations_ras: Any
    channel_coordinates: Any
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

    def can_load_previous_alignments(self) -> Ok | Failed:
        """Return whether previous alignments can be loaded."""
        return self.controller.can_load_previous_alignments()

    def load_previous_alignments(
        self,
        *,
        folder: Path | None,
        use_docdb: bool,
        shank_idx: int | None = None,
    ) -> AlignmentChoicesUpdated | NoPreviousAlignments | Failed:
        """Load and store previous alignments for a document-selected shank."""
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
        elif isinstance(result, Failed):
            self.events.emit(
                PreviousAlignmentLoadFailed(
                    shank_idx=target_shank,
                    message=result.message,
                )
            )
        return result

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
        return self.controller.select_previous_alignment(
            idx,
            shank_idx=self._active_or_given_shank(shank_idx),
            mark_changed=True,
        )

    def can_save_alignment_output(self) -> Ok | Blocked:
        """Return whether edited alignment outputs can be saved."""
        return self.controller.can_save_alignment_output()

    def save_edited_alignment_outputs(
        self,
        *,
        use_docdb: bool,
    ) -> EditedAlignmentOutputsSaved | Blocked | Failed:
        """Persist outputs for every dirty alignment state in the document."""
        ready = self.controller.can_save_alignment_output()
        if isinstance(ready, Blocked):
            return ready

        save_inputs = self._dirty_alignment_output_inputs()
        if isinstance(save_inputs, Failed):
            return self._save_failed(save_inputs.message)
        if not save_inputs:
            return self._save_failed("No edited alignments are ready to save")

        output_inputs = {
            key: (save_input.channel_locations_ras, save_input.channel_coordinates)
            for key, save_input in save_inputs.items()
        }
        outputs = self._build_alignment_outputs(
            output_inputs,
            multi_shank_by_key={
                key: save_input.multi_shank for key, save_input in save_inputs.items()
            },
        )
        if isinstance(outputs, Failed):
            return self._save_failed(outputs.message)

        logger.info("Saving output files to results folder...")
        saved_outputs: dict[AlignmentKey, AlignmentOutputsSaved] = {}
        for key, output in outputs.items():
            save_input = save_inputs[key]
            state = save_input.state
            alignment = state.active_alignment
            alignments_to_save = state.alignments
            if alignment is not None:
                _, alignments_to_save = state.with_alignment_added(
                    alignment.feature,
                    alignment.track,
                )
            saved = self._save_alignment_output(
                output,
                alignments_to_save,
                key.shank_idx,
                use_docdb,
                output_directory=save_input.output_directory,
            )
            if isinstance(saved, Failed):
                return self._save_failed(saved.message)
            saved_outputs[key] = saved
            state.set_alignments(alignments_to_save)
            state.mark_saved()
        self.controller.document.dirty = self.controller.document.has_unsaved_alignments

        active_choices: list[str] | None = None
        choices = self.controller.active_alignment_choices(
            self._active_or_given_shank(None)
        )
        if isinstance(choices, AlignmentChoicesUpdated):
            active_choices = choices.choices
        elif isinstance(choices, Failed):
            logger.error(choices.message)

        result = EditedAlignmentOutputsSaved(
            saved_count=len(saved_outputs),
            saved_outputs=saved_outputs,
            active_choices=active_choices,
        )
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

    def _emit_save_failed(self, message: str) -> None:
        self.events.emit(SaveFailed(message=message))

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
    ) -> dict[AlignmentKey, AlignmentSaveInput] | Failed:
        """Collect save inputs for every dirty document alignment state."""
        states_by_key = self.controller.document.dirty_alignment_states()
        if not states_by_key:
            return {}

        runtime_plan = plan_save_runtime_dependencies(
            document=self.controller.document,
            data_context=self.data_context,
            runtime=self.runtime,
        )
        if runtime_plan.unavailable:
            return Failed(runtime_plan.failure_message() or "Cannot save alignment.")
        runtime_by_key = runtime_plan.by_key

        save_inputs: dict[AlignmentKey, AlignmentSaveInput] = {}
        for key, state in sorted(
            states_by_key.items(),
            key=lambda item: (
                item[0].recording_id,
                item[0].ephys_collection,
                item[0].shank_idx,
            ),
        ):
            alignment = state.active_alignment
            if alignment is None:
                continue

            stream_runtime = runtime_by_key[key].stream_runtime
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
                channel_locations_ras=channel_locations_ras,
                channel_coordinates=shank_runtime.chn_coords,
                output_directory=output_directory,
                multi_shank=self._stream_is_multi_shank(stream_runtime),
            )
        return save_inputs

    def _build_alignment_outputs(
        self,
        alignments: dict[AlignmentKey, tuple[Any, Any]],
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
                    key: output_builder.get_alignment_results(
                        channel_locations_ras,
                        channel_coordinates,
                    )
                    for key, (
                        channel_locations_ras,
                        channel_coordinates,
                    ) in alignments.items()
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

    def _save_alignment_output(
        self,
        output: AlignmentOutputBuilt,
        alignments: AlignmentHistory,
        shank_idx: int,
        use_docdb: bool,
        *,
        output_directory: Path | None = None,
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
                use_docdb=use_docdb,
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

        output_directory = output_root / probe.recording_id / probe.probe_name
        try:
            output_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return Failed(
                f"Failed to create probe output directory {output_directory}: {exc}"
            )
        return output_directory

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
