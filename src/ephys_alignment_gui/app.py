"""UI-facing application port for the alignment workspace."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.alignment_events import ShankChanged
from ephys_alignment_gui.controller import AlignmentController, Failed, ShankSelected
from ephys_alignment_gui.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.event_bus import EventBus
from ephys_alignment_gui.plot_menu_state import PlotMenuState, build_plot_menu_state
from ephys_alignment_gui.plot_registry import (
    PlotMenu,
    PlotSpec,
    resolve_plot_bounds,
    resolve_plot_payload,
)
from ephys_alignment_gui.session_runtime import SessionRuntime

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


@dataclass
class AlignmentQueries:
    """Query/read-model app port for UI rendering state."""

    document: AlignmentDocument
    runtime: SessionRuntime

    def active_shank_selection(self) -> ShankSelectionState:
        """Return the current document-owned shank selection."""
        shank_idx = self._active_shank_idx()
        return ShankSelectionState(
            shank_idx=shank_idx,
            shank_id=shank_idx + 1,
            alignment_key=self.document.selected_alignment_key,
            data_loaded=self.document.data_loaded,
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
