"""App-level alignment history and output persistence commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.alignment_repository import (
    AlignmentHistory,
    AlignmentRepository,
)
from ephys_alignment_gui.alignment_state import LEGACY_AUTO_ALIGNMENT_LABEL
from ephys_alignment_gui.application.results import (
    AlignmentChoicesUpdated,
    PreviousAlignmentSelected,
    VisitedAlignmentOutputsSaved,
)
from ephys_alignment_gui.application.results.alignment_persistence import (
    AlignmentOutputBuilt,
    AlignmentOutputsSaved,
    NoPreviousAlignments,
)
from ephys_alignment_gui.application.workflow import Blocked, Failed, Ok
from ephys_alignment_gui.controller import (
    AlignmentController,
)
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.runtime.session import SessionRuntime

logger = logging.getLogger(__name__)


@dataclass
class AlignmentPersistenceCommandHandler:
    """Coordinate alignment history loading and visited-output persistence."""

    controller: AlignmentController
    data_context: AlignmentDataContext
    runtime: SessionRuntime
    derived_data_service: AlignmentDerivedDataService
    alignment_repository: AlignmentRepository
    output_builder: Any

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
            return Failed(f"Failed to load previous alignments: {exc}")

        if loaded is None:
            return NoPreviousAlignments()
        return self.controller.set_previous_alignments(
            loaded.alignments,
            shank_idx=target_shank,
        )

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
        )

    def can_save_alignment_output(self) -> Ok | Blocked:
        """Return whether visited alignment outputs can be saved."""
        return self.controller.can_save_alignment_output()

    def save_visited_alignment_outputs(
        self,
        *,
        use_docdb: bool,
    ) -> VisitedAlignmentOutputsSaved | Blocked | Failed:
        """Persist outputs for every visited alignment in the active stream."""
        ready = self.controller.can_save_alignment_output()
        if isinstance(ready, Blocked):
            return ready

        output_inputs, states_by_key = self._visited_alignment_output_inputs()
        if not output_inputs:
            return Failed("No visited alignments are ready to save")

        outputs = self._build_alignment_outputs(output_inputs)
        if isinstance(outputs, Failed):
            return outputs

        for state in states_by_key.values():
            alignment = state.active_alignment
            if alignment is not None:
                state.add_alignment(alignment.feature, alignment.track)

        logger.info("Saving output files to results folder...")
        saved_outputs: dict[AlignmentKey, AlignmentOutputsSaved] = {}
        for key, output in outputs.items():
            state = states_by_key[key]
            saved = self._save_alignment_output(
                output,
                state.alignments,
                key.shank_idx,
                use_docdb,
            )
            if isinstance(saved, Failed):
                return saved
            saved_outputs[key] = saved

        active_choices: list[str] | None = None
        choices = self.controller.active_alignment_choices(
            self._active_or_given_shank(None)
        )
        if isinstance(choices, AlignmentChoicesUpdated):
            active_choices = choices.choices
        elif isinstance(choices, Failed):
            logger.error(choices.message)

        return VisitedAlignmentOutputsSaved(
            saved_count=len(saved_outputs),
            saved_outputs=saved_outputs,
            active_choices=active_choices,
        )

    def _visited_alignment_output_inputs(
        self,
    ) -> tuple[
        dict[AlignmentKey, tuple[Any, Any]],
        dict[AlignmentKey, Any],
    ]:
        """Collect channel-location save inputs for visited shanks."""
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return {}, {}
        probe = self.data_context.probe_info
        if probe is None:
            return {}, {}

        states_for_probe = self.controller.document.alignment_states_for_current_probe()
        output_inputs: dict[AlignmentKey, tuple[Any, Any]] = {}
        states_by_key: dict[AlignmentKey, Any] = {}
        for shank_idx, shank_runtime in stream_runtime.visited_shank_runtimes().items():
            key = AlignmentKey(
                recording_id=probe.recording_id,
                ephys_collection=probe.ephys_collection,
                shank_idx=shank_idx,
            )
            state = states_for_probe.get(key)
            if state is None or state.active_alignment is None:
                continue
            if shank_runtime.ephysalign is None or shank_runtime.chn_coords is None:
                logger.info(
                    "Skipping shank %d during save because it has not been rendered",
                    shank_idx + 1,
                )
                continue
            alignment = state.active_alignment
            channel_locations_ras = (
                self.derived_data_service.compute_channel_locations(
                    ephysalign=shank_runtime.ephysalign,
                    feature=alignment.feature,
                    track=alignment.track,
                )
            )
            output_inputs[key] = (channel_locations_ras, shank_runtime.chn_coords)
            states_by_key[key] = state
        return output_inputs, states_by_key

    def _build_alignment_outputs(
        self,
        alignments: dict[AlignmentKey, tuple[Any, Any]],
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
                multi_shank=multi_shank,
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
    ) -> AlignmentOutputsSaved | Failed:
        output_directory = self.controller.document.output_directory
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

    def _active_or_given_shank(self, shank_idx: int | None) -> int:
        if shank_idx is not None:
            return shank_idx
        return self.controller.document.selected_shank

    def _alignment_repository(self) -> AlignmentRepository:
        return self.alignment_repository

    def _output_builder(self) -> Any | None:
        return self.output_builder
