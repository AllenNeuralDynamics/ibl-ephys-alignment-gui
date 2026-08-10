"""Read models for the active shank screen."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.alignment_read_models import (
    ActiveShankScreenState,
    PreparedActiveShankScreenState,
)
from ephys_alignment_gui.ephys_plot_queries import EphysPlotQueries
from ephys_alignment_gui.plotting.registry import PlotMenu
from ephys_alignment_gui.slice_display_policy import SliceSelection
from ephys_alignment_gui.slice_queries import SliceQueries
from ephys_alignment_gui.workspace_state_queries import WorkspaceStateQueries


@dataclass(frozen=True)
class ActiveShankScreenQueries:
    """Compose Qt-free screen state for the active shank."""

    workspace_state_queries: WorkspaceStateQueries
    ephys_plot_queries: EphysPlotQueries
    slice_queries: SliceQueries

    def active_shank_screen_state(
        self,
        *,
        preserve_plot_selection: bool,
        previous_ephys_plot_keys: Mapping[PlotMenu, str | None] | None = None,
        raw_image_payloads: Mapping[Any, Any] | None = None,
        previous_slice_selection: SliceSelection | None = None,
        offline: bool,
    ) -> ActiveShankScreenState:
        """Return the Qt-free screen state for the active shank."""
        selection = self.workspace_state_queries.active_shank_selection()
        return ActiveShankScreenState(
            shank_idx=selection.shank_idx,
            shank_id=selection.shank_id,
            alignment_key=selection.alignment_key,
            data_loaded=selection.data_loaded,
            preserve_plot_selection=preserve_plot_selection,
            unit_filter=self.ephys_plot_queries.active_unit_filter(),
            plot_menu=self.ephys_plot_queries.active_plot_menu_state(
                previous_selected_keys=(
                    previous_ephys_plot_keys if preserve_plot_selection else None
                ),
                raw_image_payloads=raw_image_payloads,
            ),
            slice_menu=self.slice_queries.active_slice_menu_state(
                offline=offline,
                previous_selection=(
                    previous_slice_selection if preserve_plot_selection else None
                ),
            ),
        )

    def prepare_active_shank_screen_state(
        self,
        *,
        histology_available: bool,
        preserve_plot_selection: bool,
        previous_ephys_plot_keys: Mapping[PlotMenu, str | None] | None = None,
        raw_image_payloads: Mapping[Any, Any] | None = None,
        previous_slice_selection: SliceSelection | None = None,
        offline: bool,
    ) -> PreparedActiveShankScreenState:
        """Materialize active shank runtime state and return its screen DTO."""
        plot_data_state = self.ephys_plot_queries.prepare_active_shank_plot_data_state()
        if plot_data_state is None:
            return PreparedActiveShankScreenState(
                plot_data=None,
                screen=None,
                histology_available=histology_available,
                slice_data_available=False,
            )

        slice_data_state = self.slice_queries.prepare_active_slice_screen_data()
        slice_data_available = slice_data_state is not None
        if histology_available and not slice_data_available:
            return PreparedActiveShankScreenState(
                plot_data=plot_data_state,
                screen=None,
                histology_available=histology_available,
                slice_data_available=False,
            )

        screen_state = self.active_shank_screen_state(
            preserve_plot_selection=preserve_plot_selection,
            previous_ephys_plot_keys=previous_ephys_plot_keys,
            raw_image_payloads=raw_image_payloads,
            previous_slice_selection=previous_slice_selection,
            offline=offline,
        )
        return PreparedActiveShankScreenState(
            plot_data=plot_data_state,
            screen=screen_state,
            histology_available=histology_available,
            slice_data_available=slice_data_available,
        )
