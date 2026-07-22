"""UI-facing application port for the alignment workspace."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.alignment_events import (
    AlignmentEdited,
    AlignmentEditKind,
    ShankChanged,
)
from ephys_alignment_gui.alignment_read_models import ActiveAlignmentRenderState
from ephys_alignment_gui.controller import (
    AlignmentChoicesUpdated,
    AlignmentController,
    AlignmentEditApplied,
    AlignmentEditNoop,
    Failed,
    NoPreviousAlignments,
    PreviousAlignmentSelected,
    PreviousAlignmentsLoaded,
    ShankSelected,
)
from ephys_alignment_gui.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.ephys_stream_runtime import StreamKey
from ephys_alignment_gui.event_bus import EventBus
from ephys_alignment_gui.plot_menu_state import PlotMenuState, build_plot_menu_state
from ephys_alignment_gui.plot_registry import (
    PlotMenu,
    PlotSpec,
    resolve_plot_bounds,
    resolve_plot_payload,
)
from ephys_alignment_gui.session_runtime import SessionRuntime
from ephys_alignment_gui.shank_runtime import ShankRuntime
from ephys_alignment_gui.workflow import Ok

logger = logging.getLogger(__name__)


class _ReferenceLinesNotProvided:
    pass


_REFERENCE_LINES_NOT_PROVIDED = _ReferenceLinesNotProvided()
ReferenceLineCapture = tuple[Any, Any] | None | _ReferenceLinesNotProvided


@dataclass(frozen=True)
class ShankSelectionState:
    """Read model for the active shank selection."""

    shank_idx: int
    shank_id: int
    alignment_key: AlignmentKey | None
    data_loaded: bool


@dataclass
class AlignmentCommands:
    """Command-side app port.

    Methods should be added here as UI call sites migrate. The controller
    remains the command implementation; this object is the UI boundary.
    """

    _controller: AlignmentController
    _events: EventBus

    def select_shank(
        self,
        shank_idx: int,
        *,
        outgoing_reference_lines: ReferenceLineCapture = _REFERENCE_LINES_NOT_PROVIDED,
        source: str = "command",
        preserve_plot_selection: bool | None = None,
    ) -> ShankSelected | Failed:
        """Select a shank as a complete app-level transaction."""
        if (
            self._controller.document.data_loaded
            and outgoing_reference_lines is not _REFERENCE_LINES_NOT_PROVIDED
        ):
            capture_result = self._capture_outgoing_reference_lines(
                outgoing_reference_lines
            )
            if isinstance(capture_result, Failed):
                return capture_result

        result = self._controller.select_shank(shank_idx)
        if isinstance(result, ShankSelected):
            self._events.emit(
                ShankChanged(
                    source=source,
                    previous_shank_idx=result.previous_shank_idx,
                    shank_idx=result.shank_idx,
                    previous_key=result.previous_key,
                    active_key=result.selected_key,
                    data_loaded=result.data_loaded,
                    preserve_plot_selection=preserve_plot_selection,
                )
            )
        return result

    def _capture_outgoing_reference_lines(
        self,
        outgoing_reference_lines: ReferenceLineCapture,
    ) -> Any:
        outgoing_shank_idx = self._controller.document.selected_shank
        if outgoing_reference_lines is None:
            return self._controller.clear_pending_reference_lines(outgoing_shank_idx)

        if outgoing_reference_lines is _REFERENCE_LINES_NOT_PROVIDED:
            return None

        feature_positions_um, track_positions_um = outgoing_reference_lines
        return self._controller.set_pending_reference_lines(
            feature_positions_um=feature_positions_um,
            track_positions_um=track_positions_um,
            shank_idx=outgoing_shank_idx,
        )

    def load_previous_alignments(
        self,
        *,
        folder: Path | None,
        use_docdb: bool,
        shank_idx: int | None = None,
    ) -> AlignmentChoicesUpdated | NoPreviousAlignments | Failed:
        """Load and store previous alignments for a document-selected shank."""
        target_shank = self._active_or_given_shank(shank_idx)
        loaded = self._controller.load_previous_alignments(
            folder=folder,
            shank_idx=target_shank,
            use_docdb=use_docdb,
        )
        if isinstance(loaded, Failed | NoPreviousAlignments):
            return loaded
        assert isinstance(loaded, PreviousAlignmentsLoaded)
        return self._controller.set_previous_alignments(
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
        return self._controller.select_previous_alignment(
            idx,
            shank_idx=self._active_or_given_shank(shank_idx),
        )

    def _active_or_given_shank(self, shank_idx: int | None) -> int:
        if shank_idx is not None:
            return shank_idx
        return self._controller.document.selected_shank

    def can_load_previous_alignments(self) -> Ok | Failed:
        """Return whether previous alignments can be loaded."""
        return self._controller.can_load_previous_alignments()

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
        result = self._controller.offset_alignment_from_tip(
            tip_position_um=tip_position_um,
            probe_tip_um=probe_tip_um,
            lin_fit=lin_fit,
            track_shift_m=track_shift_m,
            shank_idx=self._active_or_given_shank(shank_idx),
        )
        self._emit_alignment_edited("offset", result)
        return result

    def fit_alignment_to_reference_lines(
        self,
        shank_runtime: Any,
        *,
        line_features_um: Any,
        line_tracks_um: Any,
        lin_fit: bool,
        extend_feature: int,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a reference-line fit for a document-selected shank runtime."""
        result = self._controller.fit_alignment_to_reference_lines(
            shank_runtime,
            line_features_um=line_features_um,
            line_tracks_um=line_tracks_um,
            lin_fit=lin_fit,
            extend_feature=extend_feature,
        )
        self._emit_alignment_edited("fit", result)
        return result

    def go_next_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor forward."""
        result = self._controller.go_next_alignment(
            self._active_or_given_shank(shank_idx)
        )
        self._emit_alignment_edited("next", result)
        return result

    def go_previous_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor backward."""
        result = self._controller.go_previous_alignment(
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
        result = self._controller.reset_alignment_to_initial(
            shank_runtime,
            lin_fit=lin_fit,
        )
        if isinstance(result, AlignmentEditApplied):
            clear_result = self._controller.clear_pending_reference_lines(
                shank_runtime.shank_idx
            )
            if isinstance(clear_result, Failed):
                logger.error(clear_result.message)
            self._emit_alignment_edited("reset", result)
        return result

    def _emit_alignment_edited(
        self,
        edit_kind: AlignmentEditKind,
        result: AlignmentEditApplied | AlignmentEditNoop | Failed,
    ) -> None:
        if not isinstance(result, AlignmentEditApplied):
            return
        active_key = self._controller.document.selected_alignment_key
        if active_key is None:
            logger.error("Cannot emit alignment edit event without an active key")
            return
        self._events.emit(
            AlignmentEdited(
                edit_kind=edit_kind,
                active_key=active_key,
                active_alignment=result.alignment,
                lin_fit=result.lin_fit,
            )
        )


@dataclass
class AlignmentQueries:
    """Query/read-model app port for UI rendering state."""

    document: AlignmentDocument
    runtime: SessionRuntime
    display_state: AlignmentDisplayState = field(default_factory=AlignmentDisplayState)
    derived_data_service: AlignmentDerivedDataService = field(
        default_factory=AlignmentDerivedDataService
    )

    def active_shank_selection(self) -> ShankSelectionState:
        """Return the current document-owned shank selection."""
        shank_idx = self._active_shank_idx()
        return ShankSelectionState(
            shank_idx=shank_idx,
            shank_id=shank_idx + 1,
            alignment_key=self.document.selected_alignment_key,
            data_loaded=self.document.data_loaded,
        )

    def is_loaded_stream_shank(
        self,
        stream_key: StreamKey | None,
        shank_idx: int,
    ) -> bool:
        """Return whether the requested stream/shank is already active."""
        if stream_key is None or not self.document.data_loaded:
            return False
        stream_runtime = self.runtime.active_stream_runtime
        return (
            stream_runtime is not None
            and self.runtime.current_stream_key == stream_key
            and stream_runtime.stream_key == stream_key
            and stream_runtime.current_shank_idx == shank_idx
            and self._active_shank_idx() == shank_idx
        )

    def active_plot_menu_state(
        self,
        previous_selected_keys: Mapping[PlotMenu, str | None] | None = None,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
        legacy_plotdata: Any = None,
    ) -> PlotMenuState:
        """Return available plot menu entries for the active shank."""
        plotdata = self._active_plotdata(legacy_plotdata=legacy_plotdata)
        return self._plot_menu_state_for_plotdata(
            plotdata,
            previous_selected_keys=previous_selected_keys,
            raw_image_payloads=raw_image_payloads,
        )

    def active_plot_spec(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
        legacy_plotdata: Any = None,
    ) -> PlotSpec | None:
        """Return an available plot spec for the active shank."""
        plotdata = self._active_plotdata(legacy_plotdata=legacy_plotdata)
        state = self._plot_menu_state_for_plotdata(
            plotdata,
            raw_image_payloads=raw_image_payloads,
        )
        return self._find_plot_spec(state, spec_key)

    def active_plot_payload(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
        legacy_plotdata: Any = None,
    ) -> Any:
        """Resolve a plot payload for the active shank."""
        plotdata = self._active_plotdata(legacy_plotdata=legacy_plotdata)
        state = self._plot_menu_state_for_plotdata(
            plotdata,
            raw_image_payloads=raw_image_payloads,
        )
        spec = self._find_plot_spec(state, spec_key)
        if spec is None:
            return None
        return resolve_plot_payload(plotdata, spec)

    def active_plot_bounds(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
        legacy_plotdata: Any = None,
    ) -> Any:
        """Resolve optional plot bounds for the active shank."""
        plotdata = self._active_plotdata(legacy_plotdata=legacy_plotdata)
        state = self._plot_menu_state_for_plotdata(
            plotdata,
            raw_image_payloads=raw_image_payloads,
        )
        spec = self._find_plot_spec(state, spec_key)
        if spec is None:
            return None
        return resolve_plot_bounds(plotdata, spec)

    def active_alignment_render_state(self) -> ActiveAlignmentRenderState | None:
        """Return derived render data for the active alignment, if available."""
        key = self.document.selected_alignment_key
        state = self.document.active_alignment_state
        if key is None or state is None:
            return None
        active_alignment = state.active_alignment
        if active_alignment is None:
            return None
        shank_runtime = self._active_shank_runtime()
        if shank_runtime is None or shank_runtime.ephysalign is None:
            return None
        return ActiveAlignmentRenderState(
            key=key,
            active_alignment=active_alignment,
            histology=self.derived_data_service.compute_histology(
                ephysalign=shank_runtime.ephysalign,
                feature=active_alignment.feature,
                track=active_alignment.track,
                region_annotation_source=(
                    self.display_state.region_annotation_source
                ),
                region_fp=shank_runtime.region_fp,
                region_label_fp=shank_runtime.region_label_fp,
                region_colour_fp=shank_runtime.region_colour_fp,
            ),
            projection=self.derived_data_service.compute_channel_projection(
                ephysalign=shank_runtime.ephysalign,
                feature=active_alignment.feature,
                track=active_alignment.track,
            ),
        )

    def _plot_menu_state_for_plotdata(
        self,
        plotdata: Any,
        *,
        previous_selected_keys: Mapping[PlotMenu, str | None] | None = None,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> PlotMenuState:
        return build_plot_menu_state(
            plotdata,
            previous_selected_keys=previous_selected_keys,
            raw_image_payloads=raw_image_payloads,
        )

    def _find_plot_spec(
        self,
        state: PlotMenuState,
        spec_key: str,
    ) -> PlotSpec | None:
        for spec in state.specs:
            if spec.key == spec_key:
                return spec
        logger.warning("Ignoring unavailable plot spec %s", spec_key)
        return None

    def _active_plotdata(self, *, legacy_plotdata: Any = None) -> Any:
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return legacy_plotdata
        return stream_runtime.plot_data_for_shank(self._active_shank_idx())

    def _active_shank_runtime(self) -> ShankRuntime | None:
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return None
        return stream_runtime.shank_runtime_by_idx.get(self._active_shank_idx())

    def _active_shank_idx(self) -> int:
        key = self.document.selected_alignment_key
        if key is not None:
            return key.shank_idx
        return self.document.selected_shank


@dataclass
class AlignmentApp:
    """Small public app port for desktop and future web frontends."""

    commands: AlignmentCommands
    queries: AlignmentQueries
    events: EventBus
