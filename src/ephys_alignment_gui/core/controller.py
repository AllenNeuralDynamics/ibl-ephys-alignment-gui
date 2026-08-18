"""Qt-free controller commands for alignment workflow state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_key_context import AlignmentKeyContext
from ephys_alignment_gui.core.alignment_state import (
    AlignmentState,
    PendingReferenceLines,
)
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.core.results import (
    AlignmentChoicesUpdated,
    AlignmentEditApplied,
    AlignmentEditNoop,
    LoadDataPrepared,
    PendingReferenceLinesUpdated,
    PreviousAlignmentSelected,
    ShankRuntimeInitialized,
    ShankSelected,
)
from ephys_alignment_gui.core.workflow import (
    Failed,
    Ok,
    PolicyResult,
    WorkflowPolicy,
)
from ephys_alignment_gui.runtime.shank import ShankRuntime
from ephys_alignment_gui.services.alignment_edit import AlignmentEditService
from ephys_alignment_gui.services.alignment_repository import AlignmentHistory
from ephys_alignment_gui.services.alignment_runtime import AlignmentRuntimeService


class AlignmentController:
    """Apply validated document and alignment-domain state transitions.

    The controller is intentionally Qt-free. App command handlers own use-case
    sequencing, IO, runtime-cache lifecycle, and event publication; the
    controller owns document mutation authority and pure alignment-domain
    service calls.
    """

    def __init__(
        self,
        document: AlignmentDocument,
        alignment_key_context: AlignmentKeyContext | None = None,
        workflow_policy: WorkflowPolicy | None = None,
        alignment_edit_service: AlignmentEditService | None = None,
        alignment_runtime_service: AlignmentRuntimeService | None = None,
    ) -> None:
        self.document = document
        self.alignment_key_context = alignment_key_context or AlignmentKeyContext()
        self.workflow_policy = workflow_policy or WorkflowPolicy()
        self.alignment_edit_service = alignment_edit_service or AlignmentEditService()
        self.alignment_runtime_service = (
            alignment_runtime_service or AlignmentRuntimeService()
        )

    def can_load_data(self) -> PolicyResult:
        """Return whether the Load Data command can proceed."""
        return self.workflow_policy.can_load_data(self.document)

    def record_mouse_root_loaded(
        self,
        loaded_root: Any,
        *,
        root_changed: bool,
    ) -> None:
        """Record an already loaded mouse root in the document."""
        self.document.set_mouse_root(
            loaded_root.root,
            mouse_id=loaded_root.mouse_id,
            clear_alignment_states=root_changed,
        )
        self.alignment_key_context.clear()

    def clear_probe_selection(self) -> None:
        """Clear selected probe and dependent document state."""
        self.document.clear_probe()
        self.alignment_key_context.clear()

    def record_probe_selected(
        self,
        recording_id: str,
        probe_name: str,
    ) -> None:
        """Record the active recording/probe choice in the document."""
        self.document.select_probe(recording_id, probe_name)
        self.alignment_key_context.clear()

    def record_channel_info_loaded(self, loaded: bool = True) -> None:
        """Record whether selected-probe channel metadata is ready."""
        self.document.set_channel_info_loaded(loaded)
        if not loaded:
            self.alignment_key_context.clear()

    def record_probe_channel_info(
        self,
        probe: Any,
        *,
        n_shanks: int,
        shank_idx: int,
    ) -> None:
        """Record selected-probe channel metadata and select an alignment key."""
        self.alignment_key_context.set_from_probe(probe, n_shanks=n_shanks)
        self.document.set_channel_info_loaded(True)
        self.document.select_alignment_key(
            self.alignment_key_context.key_for_shank(shank_idx)
        )

    def record_output_root(self, output_root: Path) -> None:
        """Record the output root in the document."""
        self.document.set_output_root(output_root)

    def record_output_package_directory(
        self,
        output_package_directory: Path | None,
    ) -> None:
        """Record the mouse-level annotation output package in the document."""
        self.document.set_output_package_directory(output_package_directory)

    def record_output_directory(self, output_directory: Any) -> None:
        """Record the derived per-probe output directory in the document."""
        self.document.set_output_directory(output_directory)

    def prepare_load_data(self) -> LoadDataPrepared:
        """Mark data unloaded and return render state for the upcoming load."""
        preserve_plot_selection = self.document.data_loaded
        self.document.mark_data_loaded(False)
        return LoadDataPrepared(preserve_plot_selection=preserve_plot_selection)

    def finish_load_data(self, shank_idx: int) -> None:
        """Record successful heavy data load for the active shank."""
        self.document.mark_data_loaded(True)
        self.set_selected_shank(shank_idx)

    def set_selected_shank(self, shank_idx: int) -> None:
        """Record the active shank selected by the user."""
        if not self.alignment_key_context.is_ready:
            self.document.set_selected_shank(shank_idx)
            return
        self.document.select_alignment_key(
            self.alignment_key_context.key_for_shank(shank_idx)
        )

    def select_shank(self, shank_idx: int) -> ShankSelected | Failed:
        """Select a shank and return the before/after document keys."""
        previous_key = self.document.selected_alignment_key
        previous_shank_idx = self.document.selected_shank
        try:
            self.set_selected_shank(shank_idx)
        except ValueError as exc:
            return Failed(str(exc))
        except Exception as exc:
            return Failed(f"Failed to select shank {shank_idx + 1}: {exc}")
        return ShankSelected(
            previous_key=previous_key,
            selected_key=self.document.selected_alignment_key,
            previous_shank_idx=previous_shank_idx,
            shank_idx=self.document.selected_shank,
            data_loaded=self.document.data_loaded,
        )

    def can_load_previous_alignments(self) -> Ok | Failed:
        """Return whether previous alignments can be loaded."""
        if self.alignment_key_context.n_shanks == 0:
            return Failed("Channel info not loaded. Please select a probe first.")
        if not self.alignment_key_context.is_ready:
            return Failed("No probe selected. Please select a probe first.")
        return Ok()

    def can_save_alignment_output(self) -> PolicyResult:
        """Return whether the current alignment output can be saved."""
        return self.workflow_policy.can_save_alignment_output(self.document)

    def active_alignment_choices(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentChoicesUpdated | Failed:
        """Return dropdown choices for the active alignment state."""
        state_or_failed = self._active_state_for_shank(shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        return self._alignment_choices(state_or_failed)

    def set_previous_alignments(
        self,
        alignments: AlignmentHistory,
        shank_idx: int | None = None,
    ) -> AlignmentChoicesUpdated | Failed:
        """Replace persisted previous alignments on the active state."""
        state_or_failed = self._active_state_for_shank(shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        state_or_failed.set_alignments(alignments)
        state_or_failed.activate_default_alignment_from_history(
            replace_clean_active=True
        )
        return self._alignment_choices(state_or_failed)

    def import_previous_alignments_for_key(
        self,
        key: AlignmentKey,
        alignments: AlignmentHistory,
    ) -> AlignmentChoicesUpdated:
        """Import persisted alignment history for a key without clobbering edits."""
        state = self.document.alignment_state_for(key)
        state.import_alignments(alignments)
        return self._alignment_choices(state)

    def select_previous_alignment(
        self,
        idx: int,
        shank_idx: int | None = None,
        *,
        mark_changed: bool = False,
    ) -> PreviousAlignmentSelected | Failed:
        """Select a previous/original alignment on the active document state."""
        state_or_failed = self._active_state_for_shank(shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        state = state_or_failed
        try:
            feature_prev, track_prev = self.document.active_select_alignment_idx(idx)
        except Exception as exc:
            return Failed(f"Failed to select alignment: {exc}")

        if mark_changed:
            state.mark_alignment_changed()
            self.document.dirty = self.document.has_unsaved_alignments

        choices = list(state.prev_align)
        choice = choices[idx] if 0 <= idx < len(choices) else None
        return PreviousAlignmentSelected(
            feature_prev=feature_prev,
            track_prev=track_prev,
            choice=choice,
            choices=choices,
        )

    def set_pending_reference_lines(
        self,
        *,
        feature_positions_um: Any,
        track_positions_um: Any,
        shank_idx: int | None = None,
    ) -> PendingReferenceLinesUpdated | Failed:
        """Store active pending feature and warped-display coordinates."""
        state_or_failed = self._active_state_for_shank(shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        try:
            lines = self.document.active_set_pending_reference_lines(
                feature_positions_um,
                track_positions_um,
            )
        except Exception as exc:
            return Failed(f"Failed to store reference lines: {exc}")
        return PendingReferenceLinesUpdated(lines)

    def clear_pending_reference_lines(
        self,
        shank_idx: int | None = None,
    ) -> PendingReferenceLinesUpdated | Failed:
        """Clear active pending reference-line coordinates."""
        state_or_failed = self._active_state_for_shank(shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        self.document.active_clear_pending_reference_lines()
        return PendingReferenceLinesUpdated(None)

    def active_pending_reference_lines(
        self,
        shank_idx: int | None = None,
    ) -> PendingReferenceLines | None | Failed:
        """Return pending reference-line coordinates for the active state."""
        state_or_failed = self._active_state_for_shank(shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        return state_or_failed.pending_reference_lines

    def initialize_shank_runtime(
        self,
        shank_runtime: ShankRuntime,
        *,
        track_annotations_ras: Any,
        brain_atlas: Any,
    ) -> ShankRuntimeInitialized | Failed:
        """Initialize loaded runtime alignment state for the active document shank."""
        state_or_failed = self._active_state_for_shank(shank_runtime.shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        state = state_or_failed

        try:
            initialized = self.alignment_runtime_service.initialize_shank_runtime(
                shank_runtime,
                track_annotations_ras=track_annotations_ras,
                brain_atlas=brain_atlas,
                feature_prev=state.feature_prev,
                track_prev=state.track_prev,
            )
        except Exception as exc:
            return Failed(f"Failed to initialize alignment runtime: {exc}")

        seeded_document_alignment = False
        if state.active_alignment is None:
            state.active_alignment = ActiveAlignment(
                initialized.feature_init,
                initialized.track_init,
            )
            seeded_document_alignment = True

        return ShankRuntimeInitialized(
            feature_init=initialized.feature_init,
            track_init=initialized.track_init,
            track_annos_and_ends_ras=initialized.track_annos_and_ends_ras,
            seeded_document_alignment=seeded_document_alignment,
        )

    def initialize_shank_runtime_for_key(
        self,
        key: AlignmentKey,
        shank_runtime: ShankRuntime,
        *,
        track_annotations_ras: Any,
        brain_atlas: Any,
    ) -> ShankRuntimeInitialized | Failed:
        """Initialize runtime alignment state for an explicit document key."""
        if key.shank_idx != shank_runtime.shank_idx:
            return Failed(
                "Alignment key does not match runtime shank: "
                f"{key.shank_idx} != {shank_runtime.shank_idx}"
            )
        state = self.document.alignment_state_for(key)

        try:
            initialized = self.alignment_runtime_service.initialize_shank_runtime(
                shank_runtime,
                track_annotations_ras=track_annotations_ras,
                brain_atlas=brain_atlas,
                feature_prev=state.feature_prev,
                track_prev=state.track_prev,
            )
        except Exception as exc:
            return Failed(f"Failed to initialize alignment runtime: {exc}")

        seeded_document_alignment = False
        if state.active_alignment is None:
            state.active_alignment = ActiveAlignment(
                initialized.feature_init,
                initialized.track_init,
            )
            seeded_document_alignment = True

        return ShankRuntimeInitialized(
            feature_init=initialized.feature_init,
            track_init=initialized.track_init,
            track_annos_and_ends_ras=initialized.track_annos_and_ends_ras,
            seeded_document_alignment=seeded_document_alignment,
        )

    def offset_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        probe_tip_um: float,
        lin_fit: bool,
        track_shift_m: float = 0.0,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply an offset edit to the active document alignment state."""
        state_or_failed = self._active_state_for_shank(shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        state = state_or_failed

        try:
            result = self.alignment_edit_service.offset_from_tip(
                state.edit_history,
                tip_position_um=tip_position_um,
                probe_tip_um=probe_tip_um,
                lin_fit=lin_fit,
                track_shift_m=track_shift_m,
            )
        except Exception as exc:
            return Failed(f"Failed to offset alignment: {exc}")

        return self._edit_result(state, result)

    def fit_alignment_to_reference_lines(
        self,
        shank_runtime: ShankRuntime,
        *,
        line_features_um: Any,
        line_tracks_um: Any,
        lin_fit: bool,
        extend_feature: int,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a fit edit to the active document alignment state."""
        state_or_failed = self._active_state_for_shank(shank_runtime.shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        state = state_or_failed
        if shank_runtime.ephysalign is None:
            return Failed("Alignment runtime is not initialized.")
        if self._reference_lines_empty(line_features_um, line_tracks_um):
            return self.reset_alignment_to_initial(shank_runtime, lin_fit=lin_fit)

        try:
            result = self.alignment_edit_service.fit_to_reference_lines(
                state.edit_history,
                ephysalign=shank_runtime.ephysalign,
                line_features_um=line_features_um,
                line_tracks_um=line_tracks_um,
                lin_fit=lin_fit,
                extend_feature=extend_feature,
            )
        except Exception as exc:
            return Failed(f"Failed to fit alignment: {exc}")

        return self._edit_result(state, result)

    def go_next_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor forward, if possible."""
        state_or_failed = self._active_state_for_shank(shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        state = state_or_failed

        try:
            result = self.alignment_edit_service.go_next(state.edit_history)
        except Exception as exc:
            return Failed(f"Failed to move to next alignment edit: {exc}")

        return self._edit_result(state, result)

    def go_previous_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor backward, if possible."""
        state_or_failed = self._active_state_for_shank(shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        state = state_or_failed

        try:
            result = self.alignment_edit_service.go_previous(state.edit_history)
        except Exception as exc:
            return Failed(f"Failed to move to previous alignment edit: {exc}")

        return self._edit_result(state, result)

    def reset_alignment_to_initial(
        self,
        shank_runtime: ShankRuntime,
        *,
        lin_fit: bool,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Reset active alignment state to the unannotated runtime geometry."""
        state_or_failed = self._active_state_for_shank(shank_runtime.shank_idx)
        if isinstance(state_or_failed, Failed):
            return state_or_failed
        state = state_or_failed
        if shank_runtime.ephysalign is None:
            return Failed("Alignment runtime is not initialized.")

        original_seed = self._reset_runtime_to_unseeded_alignment(shank_runtime)
        if isinstance(original_seed, Failed):
            return original_seed
        feature_init, track_init = original_seed
        state.clear_previous_alignment_selection()

        try:
            result = self.alignment_edit_service.reset_to_initial(
                state.edit_history,
                feature_init=feature_init,
                track_init=track_init,
                lin_fit=lin_fit,
            )
        except Exception as exc:
            return Failed(f"Failed to reset alignment: {exc}")

        return self._edit_result(state, result)

    def _reset_runtime_to_unseeded_alignment(
        self,
        shank_runtime: ShankRuntime,
    ) -> tuple[Any, Any] | Failed:
        """Rebuild runtime alignment without a selected previous alignment."""
        track_annotations_ras = getattr(shank_runtime, "track_annotations_ras", None)
        brain_atlas = getattr(shank_runtime.ephysalign, "brain_atlas", None)
        if track_annotations_ras is None or brain_atlas is None:
            return (
                shank_runtime.ephysalign.feature_init,
                shank_runtime.ephysalign.track_init,
            )

        try:
            initialized = self.alignment_runtime_service.initialize_shank_runtime(
                shank_runtime,
                track_annotations_ras=track_annotations_ras,
                brain_atlas=brain_atlas,
            )
        except Exception as exc:
            return Failed(f"Failed to reset alignment runtime: {exc}")
        return initialized.feature_init, initialized.track_init

    @staticmethod
    def _reference_lines_empty(
        line_features_um: Any,
        line_tracks_um: Any,
    ) -> bool:
        return (
            np.asarray(line_features_um, dtype=float).size == 0
            or np.asarray(line_tracks_um, dtype=float).size == 0
        )

    def _active_state_for_shank(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentState | Failed:
        state = self.document.active_alignment_state
        key = self.document.selected_alignment_key
        if state is None or key is None:
            return Failed("No active alignment state selected.")
        if shank_idx is not None and key.shank_idx != shank_idx:
            return Failed(
                "Active alignment state does not match active shank: "
                f"{key.shank_idx} != {shank_idx}"
            )
        return state

    @staticmethod
    def _alignment_choices(state: AlignmentState) -> AlignmentChoicesUpdated:
        return AlignmentChoicesUpdated(
            choices=list(state.prev_align),
        )

    def _edit_result(
        self,
        state: AlignmentState,
        result: Any,
    ) -> AlignmentEditApplied | AlignmentEditNoop:
        if not result.changed or result.alignment is None:
            return AlignmentEditNoop()
        state.mark_alignment_changed()
        self.document.dirty = self.document.has_unsaved_alignments
        return AlignmentEditApplied(
            alignment=result.alignment,
            lin_fit=result.lin_fit,
        )
