"""Qt-free composition root for an alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.alignment_edit_service import AlignmentEditService
from ephys_alignment_gui.alignment_output_service import AlignmentOutputService
from ephys_alignment_gui.alignment_repository import AlignmentRepository
from ephys_alignment_gui.controller import AlignmentController
from ephys_alignment_gui.document import AlignmentDocument
from ephys_alignment_gui.ephys_data_service import EphysDataService
from ephys_alignment_gui.histology_data_service import (
    HistologyDataContext,
    HistologyDataService,
)
from ephys_alignment_gui.load_data_local import LoadDataLocal
from ephys_alignment_gui.plot_data_factory import PlotDataFactory
from ephys_alignment_gui.probe_data_workflow import ProbeDataWorkflow
from ephys_alignment_gui.session_runtime import SessionRuntime
from ephys_alignment_gui.slice_display_policy import SliceDisplayPolicy
from ephys_alignment_gui.slice_service import SliceService
from ephys_alignment_gui.workflow import WorkflowPolicy

AutoAlignmentKey = tuple[str, int]
AutoAlignment = list[list[float]]


@dataclass
class AlignmentWorkspace:
    """Owns document, controllers, services, and runtime caches.

    This is intentionally Qt-free. The active :class:`ProbeSession` still lives
    in the view layer for now because it owns plot items and signal lifetimes,
    but the cache boundaries belong to the workspace.
    """

    document: AlignmentDocument = field(default_factory=AlignmentDocument)
    data_context: AlignmentDataContext = field(default_factory=AlignmentDataContext)
    ephys_data_service: EphysDataService = field(default_factory=EphysDataService)
    histology_data_service: HistologyDataService = field(
        default_factory=HistologyDataService
    )
    histology_context: HistologyDataContext = field(
        default_factory=HistologyDataContext
    )
    slice_service: SliceService = field(default_factory=SliceService)
    slice_display_policy: SliceDisplayPolicy = field(default_factory=SliceDisplayPolicy)
    workflow_policy: WorkflowPolicy = field(default_factory=WorkflowPolicy)
    alignment_repository: AlignmentRepository = field(
        default_factory=AlignmentRepository
    )
    alignment_edit_service: AlignmentEditService = field(
        default_factory=AlignmentEditService
    )
    alignment_derived_data_service: AlignmentDerivedDataService = field(
        default_factory=AlignmentDerivedDataService
    )
    alignment_output_service: AlignmentOutputService = field(init=False)
    plot_data_factory: PlotDataFactory = field(default_factory=PlotDataFactory)
    runtime: SessionRuntime = field(default_factory=SessionRuntime)
    auto_alignments: dict[AutoAlignmentKey, AutoAlignment] = field(default_factory=dict)
    probe_data_workflow: ProbeDataWorkflow = field(init=False)
    loader: LoadDataLocal = field(init=False)
    controller: AlignmentController = field(init=False)

    def __post_init__(self) -> None:
        self.probe_data_workflow = ProbeDataWorkflow(
            self.data_context,
            self.ephys_data_service,
        )
        self.loader = LoadDataLocal(
            data_context=self.data_context,
            histology_context=self.histology_context,
            slice_service=self.slice_service,
        )
        self.alignment_output_service = AlignmentOutputService(
            self.data_context,
            self.histology_context,
        )
        self.controller = AlignmentController(
            self.document,
            self.data_context,
            self.ephys_data_service,
            self.workflow_policy,
            alignment_repository=self.alignment_repository,
            output_builder=self.alignment_output_service,
        )
