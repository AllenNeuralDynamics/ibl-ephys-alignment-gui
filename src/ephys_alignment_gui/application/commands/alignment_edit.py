"""App-level alignment edit command handler."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ephys_alignment_gui.application.commands.autosave import (
    AutosaveCheckpointCommandHandler,
)
from ephys_alignment_gui.application.results import (
    AlignmentEditApplied,
    AlignmentEditNoop,
)
from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.core.alignment_events import AlignmentEdited, AlignmentEditKind
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.workflow import Failed, Ok
from ephys_alignment_gui.runtime.session import SessionRuntime
from ephys_alignment_gui.runtime.shank import ShankRuntime

logger = logging.getLogger(__name__)


@dataclass
class AlignmentEditCommandHandler:
    """Coordinate edit settings/runtime access around controller mutations."""

    controller: AlignmentController
    events: EventBus
    display_state: AlignmentDisplayState
    runtime: SessionRuntime
    autosave_checkpoints: AutosaveCheckpointCommandHandler | None = None

    def set_unit_filter(self, unit_filter: str) -> Ok:
        """Select the unit subset used when preparing ephys plot data."""
        self.display_state.set_unit_filter(unit_filter)
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is not None:
            stream_runtime.filtered_plot_payload_cache_for_shank(
                self._active_or_given_shank(None),
                unit_filter=unit_filter,
            )
        return Ok()

    def offset_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        probe_tip_um: float,
        lin_fit: bool,
        track_shift_m: float = 0.0,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply an offset edit on a document-selected shank."""
        result = self.controller.offset_alignment_from_tip(
            tip_position_um=tip_position_um,
            probe_tip_um=probe_tip_um,
            lin_fit=lin_fit,
            track_shift_m=track_shift_m,
            shank_idx=self._active_or_given_shank(shank_idx),
        )
        self._emit_alignment_edited("offset", result)
        return result

    def offset_active_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        track_shift_m: float = 0.0,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a tip-offset edit using app-owned display settings."""
        return self.offset_alignment_from_tip(
            tip_position_um=tip_position_um,
            probe_tip_um=self.display_state.depth_view.probe_tip_um,
            lin_fit=self.display_state.edit_settings.lin_fit,
            track_shift_m=track_shift_m,
        )

    def nudge_active_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        track_shift_m: float,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a bounded tip-offset nudge for the active alignment."""
        if not self._active_alignment_can_shift(track_shift_m):
            return AlignmentEditNoop()
        return self.offset_active_alignment_from_tip(
            tip_position_um=tip_position_um,
            track_shift_m=track_shift_m,
        )

    def fit_alignment_to_reference_lines(
        self,
        shank_runtime: Any,
        *,
        line_features_um: Any,
        line_tracks_um: Any,
        lin_fit: bool,
        extend_feature: int,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a reference-line fit for a document-selected shank runtime.

        ``line_tracks_um`` is the legacy name for warped display depths.
        """
        if self._reference_lines_empty(line_features_um, line_tracks_um):
            return self.reset_alignment_to_initial(
                shank_runtime,
                lin_fit=lin_fit,
            )
        result = self.controller.fit_alignment_to_reference_lines(
            shank_runtime,
            line_features_um=line_features_um,
            line_tracks_um=line_tracks_um,
            lin_fit=lin_fit,
            extend_feature=extend_feature,
        )
        self._emit_alignment_edited("fit", result)
        return result

    def fit_active_alignment_from_pending_reference_lines(
        self,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a fit edit from document-owned pending reference lines."""
        shank_runtime = self._active_shank_runtime()
        if shank_runtime is None:
            return Failed("Cannot fit alignment: active shank runtime is not loaded")

        pending_lines = self.controller.active_pending_reference_lines(
            shank_runtime.shank_idx
        )
        if isinstance(pending_lines, Failed):
            return pending_lines
        if pending_lines is None:
            line_features_um = np.array([], dtype=float)
            line_tracks_um = np.array([], dtype=float)
        else:
            line_features_um = pending_lines.feature_positions_um
            line_tracks_um = pending_lines.warped_positions_um

        return self.fit_alignment_to_reference_lines(
            shank_runtime,
            line_features_um=line_features_um,
            line_tracks_um=line_tracks_um,
            lin_fit=self.display_state.edit_settings.lin_fit,
            extend_feature=self.display_state.edit_settings.extend_feature,
        )

    def go_next_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor forward."""
        result = self.controller.go_next_alignment(
            self._active_or_given_shank(shank_idx)
        )
        self._emit_alignment_edited("next", result)
        return result

    def go_previous_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor backward."""
        result = self.controller.go_previous_alignment(
            self._active_or_given_shank(shank_idx)
        )
        self._emit_alignment_edited("previous", result)
        return result

    def reset_alignment_to_initial(
        self,
        shank_runtime: Any,
        *,
        lin_fit: bool,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Reset active alignment state to the loaded runtime's initial geometry."""
        result = self.controller.reset_alignment_to_initial(
            shank_runtime,
            lin_fit=lin_fit,
        )
        if isinstance(result, AlignmentEditApplied):
            clear_result = self.controller.clear_pending_reference_lines(
                shank_runtime.shank_idx
            )
            if isinstance(clear_result, Failed):
                logger.error(clear_result.message)
            self._emit_alignment_edited("reset", result)
        return result

    def reset_active_alignment_to_initial(
        self,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Reset active alignment using the active runtime and display settings."""
        shank_runtime = self._active_shank_runtime()
        if shank_runtime is None:
            return Failed("Cannot reset alignment: active shank runtime is not loaded")
        return self.reset_alignment_to_initial(
            shank_runtime,
            lin_fit=self.display_state.edit_settings.lin_fit,
        )

    def _active_or_given_shank(self, shank_idx: int | None) -> int:
        if shank_idx is not None:
            return shank_idx
        return self.controller.document.selected_shank

    def _active_alignment_can_shift(self, track_shift_m: float) -> bool:
        """Return whether a bounded nudge keeps the alignment inside channel depths."""
        if track_shift_m == 0:
            return True
        state = self.controller.document.active_alignment_state
        alignment = None if state is None else state.active_alignment
        shank_runtime = self._active_shank_runtime()
        if (
            alignment is None
            or shank_runtime is None
            or shank_runtime.chn_depths is None
        ):
            return False

        channel_depths_m = np.asarray(shank_runtime.chn_depths, dtype=float) / 1e6
        if channel_depths_m.size == 0:
            return False
        if track_shift_m < 0:
            return alignment.track[-1] + track_shift_m >= float(
                np.max(channel_depths_m)
            )
        return alignment.track[0] + track_shift_m <= float(np.min(channel_depths_m))

    def _active_shank_runtime(self) -> ShankRuntime | None:
        """Return active shank runtime data for command-side alignment edits."""
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return None
        return stream_runtime.shank_runtime_by_idx.get(
            self.controller.document.selected_shank
        )

    @staticmethod
    def _reference_lines_empty(line_features_um: Any, line_tracks_um: Any) -> bool:
        return (
            np.asarray(line_features_um, dtype=float).size == 0
            or np.asarray(line_tracks_um, dtype=float).size == 0
        )

    def _emit_alignment_edited(
        self,
        edit_kind: AlignmentEditKind,
        result: AlignmentEditApplied | AlignmentEditNoop | Failed,
    ) -> None:
        if not isinstance(result, AlignmentEditApplied):
            return
        active_key = self.controller.document.selected_alignment_key
        if active_key is None:
            logger.error("Cannot emit alignment edit event without an active key")
            return
        self.events.emit(
            AlignmentEdited(
                edit_kind=edit_kind,
                active_key=active_key,
                active_alignment=result.alignment,
                lin_fit=result.lin_fit,
            )
        )
        self._write_autosave_checkpoint(f"alignment {edit_kind}")

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
