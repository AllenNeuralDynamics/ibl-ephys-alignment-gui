"""Qt-free composition root for an alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.application.app import AlignmentApp
from ephys_alignment_gui.application.commands import AlignmentCommands
from ephys_alignment_gui.application.commands.alignment_edit import (
    AlignmentEditCommandHandler,
)
from ephys_alignment_gui.application.commands.alignment_persistence import (
    AlignmentPersistenceCommandHandler,
)
from ephys_alignment_gui.application.commands.display import DisplayCommandHandler
from ephys_alignment_gui.application.commands.load_data import LoadDataCommandHandler
from ephys_alignment_gui.application.commands.load_data_lifecycle import (
    LoadDataExecutionLifecycle,
)
from ephys_alignment_gui.application.commands.loaded_shank import (
    LoadedShankCommandHandler,
)
from ephys_alignment_gui.application.commands.metadata_selection import (
    MetadataSelectionCommandHandler,
)
from ephys_alignment_gui.application.commands.path import PathCommandHandler
from ephys_alignment_gui.application.commands.shank_selection import (
    ShankSelectionCommandHandler,
)
from ephys_alignment_gui.application.queries import AlignmentQueries
from ephys_alignment_gui.application.save_runtime_rehydration import (
    SaveRuntimeRehydrator,
)
from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.core.alignment_key_context import AlignmentKeyContext
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.core.document import AlignmentDocument
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.settings import max_cached_streams_from_environment
from ephys_alignment_gui.core.slice_display_policy import SliceDisplayPolicy
from ephys_alignment_gui.core.workflow import WorkflowPolicy
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.io.ephys_stream_loader import EphysStreamLoader
from ephys_alignment_gui.io.load_data_job import LoadDataJob
from ephys_alignment_gui.plotting.payload_cache_factory import (
    EphysPlotPayloadCacheFactory,
)
from ephys_alignment_gui.runtime.histology_loader import HistologyRuntimeLoader
from ephys_alignment_gui.runtime.session import SessionRuntime
from ephys_alignment_gui.runtime.slice_data_service import SliceDataRuntimeService
from ephys_alignment_gui.services.alignment_derived_data import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.services.alignment_edit import AlignmentEditService
from ephys_alignment_gui.services.alignment_output import AlignmentOutputService
from ephys_alignment_gui.services.alignment_repository import AlignmentRepository
from ephys_alignment_gui.services.alignment_runtime import AlignmentRuntimeService
from ephys_alignment_gui.services.ephys_data import EphysDataService
from ephys_alignment_gui.services.histology_data import (
    HistologyDataContext,
    HistologyDataService,
)
from ephys_alignment_gui.services.probe_track import ProbeTrackService
from ephys_alignment_gui.services.region_lookup import RegionLookupService
from ephys_alignment_gui.services.slice import SliceService


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
    plot_payload_cache_factory: EphysPlotPayloadCacheFactory = field(
        default_factory=EphysPlotPayloadCacheFactory
    )
    runtime: SessionRuntime = field(
        default_factory=lambda: SessionRuntime(
            max_cached_streams=max_cached_streams_from_environment()
        )
    )
    events: EventBus = field(default_factory=EventBus)
    load_data_lifecycle: LoadDataExecutionLifecycle = field(
        default_factory=LoadDataExecutionLifecycle
    )
    preload_data_lifecycle: LoadDataExecutionLifecycle = field(
        default_factory=LoadDataExecutionLifecycle
    )
    save_runtime_rehydrator: SaveRuntimeRehydrator = field(init=False)
    ephys_stream_loader: EphysStreamLoader = field(init=False)
    histology_runtime_loader: HistologyRuntimeLoader = field(init=False)
    load_data_job: LoadDataJob = field(init=False)
    controller: AlignmentController = field(init=False)
    path_commands: PathCommandHandler = field(init=False)
    metadata_commands: MetadataSelectionCommandHandler = field(init=False)
    shank_selection_commands: ShankSelectionCommandHandler = field(init=False)
    load_data_commands: LoadDataCommandHandler = field(init=False)
    loaded_shank_commands: LoadedShankCommandHandler = field(init=False)
    persistence_commands: AlignmentPersistenceCommandHandler = field(init=False)
    edit_commands: AlignmentEditCommandHandler = field(init=False)
    display_commands: DisplayCommandHandler = field(init=False)
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
        self.path_commands = PathCommandHandler(
            controller=self.controller,
            data_context=self.data_context,
            events=self.events,
        )
        self.metadata_commands = MetadataSelectionCommandHandler(
            controller=self.controller,
            data_context=self.data_context,
            ephys_data_service=self.ephys_data_service,
            path_commands=self.path_commands,
            histology_context=self.histology_context,
        )
        self.shank_selection_commands = ShankSelectionCommandHandler(
            controller=self.controller,
            events=self.events,
        )
        self.load_data_commands = LoadDataCommandHandler(
            controller=self.controller,
            data_context=self.data_context,
            display_state=self.display_state,
            runtime=self.runtime,
            load_data_job=self.load_data_job,
            load_lifecycle=self.load_data_lifecycle,
            preload_lifecycle=self.preload_data_lifecycle,
            histology_runtime_loader=self.histology_runtime_loader,
            plot_payload_cache_factory=self.plot_payload_cache_factory,
            metadata_commands=self.metadata_commands,
            events=self.events,
        )
        self.save_runtime_rehydrator = SaveRuntimeRehydrator(
            controller=self.controller,
            runtime=self.runtime,
            ephys_data_service=self.ephys_data_service,
            load_data_job=self.load_data_job,
            histology_runtime_loader=self.histology_runtime_loader,
            plot_payload_cache_factory=self.plot_payload_cache_factory,
            histology_context=self.histology_context,
            probe_track_service=self.probe_track_service,
        )
        self.loaded_shank_commands = LoadedShankCommandHandler(
            controller=self.controller,
            data_context=self.data_context,
            runtime=self.runtime,
            histology_context=self.histology_context,
            probe_track_service=self.probe_track_service,
        )
        self.persistence_commands = AlignmentPersistenceCommandHandler(
            controller=self.controller,
            data_context=self.data_context,
            runtime=self.runtime,
            derived_data_service=self.alignment_derived_data_service,
            alignment_repository=self.alignment_repository,
            output_builder=self.alignment_output_service,
            events=self.events,
            save_runtime_rehydrator=self.save_runtime_rehydrator,
        )
        self.edit_commands = AlignmentEditCommandHandler(
            controller=self.controller,
            events=self.events,
            display_state=self.display_state,
            runtime=self.runtime,
        )
        self.display_commands = DisplayCommandHandler(
            display_state=self.display_state,
            events=self.events,
        )
        self.app = AlignmentApp(
            commands=AlignmentCommands(
                paths=self.path_commands,
                metadata=self.metadata_commands,
                shanks=self.shank_selection_commands,
                load=self.load_data_commands,
                loaded_shank=self.loaded_shank_commands,
                persistence=self.persistence_commands,
                edit=self.edit_commands,
                display=self.display_commands,
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
                region_lookup_service=self.region_lookup_service,
                slice_display_policy=self.slice_display_policy,
            ),
            events=self.events,
        )
