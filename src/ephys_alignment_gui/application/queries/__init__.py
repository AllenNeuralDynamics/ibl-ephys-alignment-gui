"""Grouped query-side application facade for the alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.application.queries.active_shank_screen import (
    ActiveShankScreenQueries,
)
from ephys_alignment_gui.application.queries.alignment_render import (
    AlignmentRenderQueries,
)
from ephys_alignment_gui.application.queries.context import AlignmentQueryContext
from ephys_alignment_gui.application.queries.ephys_plot import EphysPlotQueries
from ephys_alignment_gui.application.queries.slice import SliceQueries
from ephys_alignment_gui.application.queries.workspace_state import (
    WorkspaceStateQueries,
)
from ephys_alignment_gui.document import AlignmentDocument
from ephys_alignment_gui.runtime.session import SessionRuntime
from ephys_alignment_gui.runtime.slice_data_service import SliceDataRuntimeService
from ephys_alignment_gui.services.alignment_derived_data import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.slice_display_policy import SliceDisplayPolicy


@dataclass
class AlignmentQueries:
    """Grouped query/read-model app port for UI rendering state."""

    document: AlignmentDocument
    runtime: SessionRuntime
    data_context: AlignmentDataContext | None = None
    display_state: AlignmentDisplayState = field(default_factory=AlignmentDisplayState)
    derived_data_service: AlignmentDerivedDataService = field(
        default_factory=AlignmentDerivedDataService
    )
    slice_data_runtime_service: SliceDataRuntimeService = field(
        default_factory=SliceDataRuntimeService
    )
    histology_context: Any | None = None
    slice_service: Any | None = None
    region_lookup_service: Any | None = None
    slice_display_policy: SliceDisplayPolicy = field(default_factory=SliceDisplayPolicy)
    context: AlignmentQueryContext = field(init=False)
    workspace: WorkspaceStateQueries = field(init=False)
    ephys: EphysPlotQueries = field(init=False)
    alignment_render: AlignmentRenderQueries = field(init=False)
    slices: SliceQueries = field(init=False)
    active_shank: ActiveShankScreenQueries = field(init=False)

    def __post_init__(self) -> None:
        self.context = AlignmentQueryContext(
            document=self.document,
            runtime=self.runtime,
        )
        self.workspace = WorkspaceStateQueries(
            context=self.context,
            data_context=self.data_context,
            display_state=self.display_state,
            histology_context=self.histology_context,
            region_lookup_service=self.region_lookup_service,
        )
        self.ephys = EphysPlotQueries(
            context=self.context,
            display_state=self.display_state,
            derived_data_service=self.derived_data_service,
            histology_context=self.histology_context,
        )
        self.alignment_render = AlignmentRenderQueries(
            context=self.context,
            display_state=self.display_state,
            derived_data_service=self.derived_data_service,
        )
        self.slices = SliceQueries(
            context=self.context,
            render_queries=self.alignment_render,
            derived_data_service=self.derived_data_service,
            slice_data_runtime_service=self.slice_data_runtime_service,
            histology_context=self.histology_context,
            slice_service=self.slice_service,
            slice_display_policy=self.slice_display_policy,
        )
        self.active_shank = ActiveShankScreenQueries(
            workspace_state_queries=self.workspace,
            ephys_plot_queries=self.ephys,
            slice_queries=self.slices,
        )
