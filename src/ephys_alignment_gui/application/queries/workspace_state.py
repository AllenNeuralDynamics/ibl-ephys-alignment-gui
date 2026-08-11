"""Read models for workspace selection, paths, and display settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ephys_alignment_gui.application.queries.context import AlignmentQueryContext
from ephys_alignment_gui.application.results import (
    ActiveProbeSelectionState,
    ShankSelectionState,
)
from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.core.alignment_read_models import (
    ActiveAlignmentEditScreenState,
    ActiveReferenceLineRenderState,
)
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.runtime.ephys_stream import StreamKey
from ephys_alignment_gui.runtime.session import LoadDataPlan, LoadDataTarget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceStateQueries:
    """Query workspace-level state that is not specific to one plot family."""

    context: AlignmentQueryContext
    data_context: AlignmentDataContext | None
    display_state: AlignmentDisplayState
    histology_context: Any | None
    region_lookup_service: Any | None

    def active_shank_selection(self) -> ShankSelectionState:
        """Return the current document-owned shank selection."""
        shank_idx = self.context.active_shank_idx()
        return ShankSelectionState(
            shank_idx=shank_idx,
            shank_id=shank_idx + 1,
            alignment_key=self.context.document.selected_alignment_key,
            data_loaded=self.context.document.data_loaded,
        )

    def active_probe_selection_state(self) -> ActiveProbeSelectionState | None:
        """Return selected probe metadata needed by desktop selectors."""
        document = self.context.document
        if document.selected_recording is None or document.selected_probe is None:
            return None

        shanks: list[str] = []
        n_shanks = 0
        if self.data_context is not None:
            shanks = self.data_context.shank_labels()
            n_shanks = self.data_context.n_shanks

        return ActiveProbeSelectionState(
            recording_id=document.selected_recording,
            probe_name=document.selected_probe,
            shanks=shanks,
            n_shanks=n_shanks,
            output_directory=document.output_directory,
        )

    def active_reference_line_state(
        self,
        shank_idx: int | None = None,
    ) -> ActiveReferenceLineRenderState | None:
        """Return pending or previous-alignment reference lines for rendering."""
        state = self.context.document.active_alignment_state
        key = self.context.document.selected_alignment_key
        if state is None or key is None:
            return None
        if shank_idx is not None and key.shank_idx != shank_idx:
            return None

        pending = state.pending_reference_lines
        if pending is not None:
            return ActiveReferenceLineRenderState(
                feature_positions_um=pending.feature_positions_um,
                track_positions_um=pending.track_positions_um,
            )

        feature_prev = state.feature_prev
        if feature_prev is None or not np.any(feature_prev):
            return None
        return ActiveReferenceLineRenderState(
            feature_positions_um=np.asarray(feature_prev)[1:-1] * 1e6,
        )

    def is_loaded_stream_shank(
        self,
        stream_key: StreamKey | None,
        shank_idx: int,
    ) -> bool:
        """Return whether the requested stream/shank is already active."""
        if stream_key is None or not self.context.document.data_loaded:
            return False
        return (
            self.context.runtime.is_active_stream_shank(stream_key, shank_idx)
            and self.context.active_shank_idx() == shank_idx
        )

    def plan_load_data(
        self,
        stream_key: StreamKey | None,
        shank_idx: int,
    ) -> LoadDataPlan:
        """Return the stream-cache plan for one load-data request."""
        return self.context.runtime.plan_load_data(
            LoadDataTarget(stream_key=stream_key, shank_idx=shank_idx),
            data_loaded=self.context.document.data_loaded,
        )

    def stream_key_for_selection(
        self,
        recording_id: str,
        probe_name: str,
    ) -> StreamKey | None:
        """Resolve the ephys stream key for a recording/probe selection."""
        if self.data_context is None:
            return None
        try:
            return self.data_context.stream_key_for_selection(recording_id, probe_name)
        except Exception:
            logger.warning(
                "Could not resolve stream key for %s/%s",
                recording_id,
                probe_name,
                exc_info=True,
            )
            return None

    def next_unloaded_probe_in_recording(
        self,
        recording_id: str,
        probe_name: str,
    ) -> str | None:
        """Return the next same-session probe whose stream is not cached."""
        if self.data_context is None:
            return None
        try:
            probes = self.data_context.list_probes(recording_id)
            start_idx = probes.index(probe_name) + 1
        except ValueError:
            return None
        except Exception:
            logger.warning(
                "Could not list preload candidates for %s/%s",
                recording_id,
                probe_name,
                exc_info=True,
            )
            return None

        for candidate in probes[start_idx:]:
            stream_key = self.stream_key_for_selection(recording_id, candidate)
            if stream_key is None:
                continue
            if stream_key == self.context.runtime.current_stream_key:
                continue
            if self.context.runtime.cached_stream(stream_key) is None:
                return candidate
        return None

    def histology_data_loaded(self) -> bool:
        """Whether subject-level histology runtime data is already loaded."""
        return (
            self.histology_context is not None
            and self.histology_context.brain_atlas is not None
        )

    def active_mouse_root_path(self) -> Path | None:
        """Return the active mouse-root path, if one is loaded."""
        if self.data_context is None or self.data_context.mouse_root is None:
            return None
        return self.data_context.mouse_root.root

    def mouse_root_loaded(self) -> bool:
        """Return whether an input mouse-root datapackage is loaded."""
        return self.active_mouse_root_path() is not None

    def active_output_root(self) -> Path | None:
        """Return the active output root, if one has been set."""
        return self.context.document.output_root

    def has_output_directory(self) -> bool:
        """Return whether the active probe output directory is available."""
        return self.context.document.output_directory is not None

    def active_output_directory(self) -> Path | None:
        """Return the derived active output directory, if available."""
        return self.context.document.output_directory

    def active_plot_export_directory(self) -> Path | None:
        """Return the default plot-export directory for the active shank."""
        output_directory = self.active_output_directory()
        if output_directory is None:
            return None
        shank_id = self.active_shank_selection().shank_id
        return output_directory / f"Plots_Shank_{shank_id}"

    def depth_view_settings(self) -> Any:
        """Return feature-depth display settings."""
        return self.display_state.depth_view

    def fit_depth_um(self) -> Any:
        """Return the depth grid used for fit-panel rendering."""
        return self.display_state.depth_view.fit_depth_um

    def linear_fit_enabled(self) -> bool:
        """Return whether fit commands use linear fitting."""
        return self.display_state.edit_settings.lin_fit

    def active_brain_atlas(self) -> Any | None:
        """Return loaded brain-atlas runtime data for desktop rendering."""
        if self.histology_context is None:
            return None
        return self.histology_context.brain_atlas

    def allen_structure_tree(self) -> Any | None:
        """Return Allen structure metadata for desktop rendering."""
        if self.region_lookup_service is None:
            return None
        return self.region_lookup_service.load_allen_csv()

    def region_description(self, region_id: int) -> tuple[str, str] | None:
        """Return user-facing region description and lookup label."""
        if self.region_lookup_service is None:
            return None
        return self.region_lookup_service.get_region_description(region_id)

    def active_alignment_edit_screen_state(
        self,
    ) -> ActiveAlignmentEditScreenState:
        """Return edit-history status and previous reference-line render data."""
        state = self.context.document.active_alignment_state
        if state is None:
            return ActiveAlignmentEditScreenState(current_idx=0, total_idx=0)

        previous_feature_positions_um = None
        feature_prev = state.feature_prev
        if feature_prev is not None and np.any(feature_prev):
            previous_feature_positions_um = np.asarray(feature_prev)[1:-1] * 1e6

        return ActiveAlignmentEditScreenState(
            current_idx=state.edit_history.current_idx,
            total_idx=state.edit_history.total_idx,
            previous_feature_positions_um=previous_feature_positions_um,
        )

    def resolve_shank_preserve_plot_selection(
        self,
        preserve_plot_selection: bool | None,
    ) -> bool:
        """Return whether shank redraw should preserve current plot selections."""
        if preserve_plot_selection is None:
            return self.context.document.data_loaded
        return preserve_plot_selection
