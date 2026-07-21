"""UI-facing application port for the alignment workspace."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.controller import AlignmentController
from ephys_alignment_gui.document import AlignmentDocument
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


@dataclass
class AlignmentCommands:
    """Command-side app port.

    Methods should be added here as UI call sites migrate. The controller
    remains the command implementation; this object is the UI boundary.
    """

    _controller: AlignmentController


@dataclass
class AlignmentQueries:
    """Query/read-model app port for UI rendering state."""

    document: AlignmentDocument
    runtime: SessionRuntime

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
