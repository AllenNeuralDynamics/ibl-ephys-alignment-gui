"""Qt-free composition root for an alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.alignment_edit_service import AlignmentEditService
from ephys_alignment_gui.alignment_key_context import AlignmentKeyContext
from ephys_alignment_gui.alignment_output_service import AlignmentOutputService
from ephys_alignment_gui.alignment_repository import AlignmentRepository
from ephys_alignment_gui.alignment_runtime_service import AlignmentRuntimeService
from ephys_alignment_gui.app import AlignmentApp, AlignmentCommands, AlignmentQueries
from ephys_alignment_gui.controller import AlignmentController
from ephys_alignment_gui.document import AlignmentDocument
from ephys_alignment_gui.ephys_data_service import EphysDataService
from ephys_alignment_gui.ephys_stream_loader import EphysStreamLoader
from ephys_alignment_gui.event_bus import EventBus
from ephys_alignment_gui.histology_data_service import (
    HistologyDataContext,
    HistologyDataService,
)
from ephys_alignment_gui.histology_runtime_loader import HistologyRuntimeLoader
from ephys_alignment_gui.load_data_job import LoadDataJob
from ephys_alignment_gui.plot_data_factory import PlotDataFactory
from ephys_alignment_gui.probe_track_service import ProbeTrackService
from ephys_alignment_gui.region_lookup_service import RegionLookupService
from ephys_alignment_gui.session_runtime import SessionRuntime
from ephys_alignment_gui.slice_data_runtime_service import SliceDataRuntimeService
from ephys_alignment_gui.slice_display_policy import SliceDisplayPolicy
from ephys_alignment_gui.slice_service import SliceService
from ephys_alignment_gui.workflow import WorkflowPolicy


@dataclass
class AlignmentWorkspace:
    """Owns document, controllers, services, and runtime caches.

    This is intentionally Qt-free. The active desktop view session lives in the
    desktop layer because it owns plot items and signal lifetimes, but the
    stream-cache boundaries belong to the workspace.
    """

    document: AlignmentDocument = field(default_factory=AlignmentDocument)
    display_state: AlignmentDisplayState = field(default_factory=AlignmentDisplayState)
    data_context: AlignmentDataContext = field(default_factory=AlignmentDataContext)
    alignment_key_context: AlignmentKeyContext = field(
        default_factory=AlignmentKeyContext
    )
    ephys_data_service: EphysDataService = field(default_factory=EphysDataService)
    histology_data_service: HistologyDataService = field(
        default_factory=HistologyDataService
    )
    histology_context: HistologyDataContext = field(
        default_factory=HistologyDataContext
    )
    slice_service: SliceService = field(default_factory=SliceService)
    slice_data_runtime_service: SliceDataRuntimeService = field(
        default_factory=SliceDataRuntimeService
    )
    probe_track_service: ProbeTrackService = field(default_factory=ProbeTrackService)
    region_lookup_service: RegionLookupService = field(
        default_factory=RegionLookupService
    )
    slice_display_policy: SliceDisplayPolicy = field(default_factory=SliceDisplayPolicy)
    workflow_policy: WorkflowPolicy = field(default_factory=WorkflowPolicy)
    alignment_repository: AlignmentRepository = field(
        default_factory=AlignmentRepository
    )
    alignment_edit_service: AlignmentEditService = field(
        default_factory=AlignmentEditService
    )
    alignment_runtime_service: AlignmentRuntimeService = field(
        default_factory=AlignmentRuntimeService
    )
    alignment_derived_data_service: AlignmentDerivedDataService = field(
        default_factory=AlignmentDerivedDataService
    )
    alignment_output_service: AlignmentOutputService = field(init=False)
    plot_data_factory: PlotDataFactory = field(default_factory=PlotDataFactory)
    runtime: SessionRuntime = field(default_factory=SessionRuntime)
    events: EventBus = field(default_factory=EventBus)
    ephys_stream_loader: EphysStreamLoader = field(init=False)
    histology_runtime_loader: HistologyRuntimeLoader = field(init=False)
    load_data_job: LoadDataJob = field(init=False)
    controller: AlignmentController = field(init=False)
    app: AlignmentApp = field(init=False)

    def __post_init__(self) -> None:
        self.ephys_stream_loader = EphysStreamLoader(
            self.data_context,
            self.ephys_data_service,
        )
        self.histology_runtime_loader = HistologyRuntimeLoader(
            self.data_context,
            self.histology_data_service,
            self.histology_context,
        )
        self.load_data_job = LoadDataJob(
            ephys_stream_loader=self.ephys_stream_loader,
            histology_runtime_loader=self.histology_runtime_loader,
        )
        self.alignment_output_service = AlignmentOutputService(
            self.data_context,
            self.histology_context,
        )
        self.controller = AlignmentController(
            self.document,
            self.alignment_key_context,
            self.workflow_policy,
            alignment_edit_service=self.alignment_edit_service,
            alignment_runtime_service=self.alignment_runtime_service,
        )
        self.app = AlignmentApp(
            commands=AlignmentCommands(
                self.controller,
                self.data_context,
                self.ephys_data_service,
                self.events,
                self.display_state,
                self.runtime,
                self.load_data_job,
                self.histology_context,
                self.probe_track_service,
                self.plot_data_factory,
                self.alignment_derived_data_service,
                self.alignment_repository,
                self.alignment_output_service,
            ),
            queries=AlignmentQueries(
                document=self.document,
                runtime=self.runtime,
                data_context=self.data_context,
                display_state=self.display_state,
                derived_data_service=self.alignment_derived_data_service,
                slice_data_runtime_service=self.slice_data_runtime_service,
                histology_context=self.histology_context,
                slice_service=self.slice_service,
                slice_display_policy=self.slice_display_policy,
            ),
            events=self.events,
        )
