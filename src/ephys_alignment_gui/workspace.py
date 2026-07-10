"""Qt-free composition root for an alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass, field

from ephys_alignment_gui.alignment_repository import AlignmentRepository
from ephys_alignment_gui.controller import AlignmentController
from ephys_alignment_gui.document import AlignmentDocument
from ephys_alignment_gui.ephys_data_service import EphysDataService
from ephys_alignment_gui.load_data_local import LoadDataLocal
from ephys_alignment_gui.plot_data_factory import PlotDataFactory
from ephys_alignment_gui.probe_session import ProbeSession
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
    ephys_data_service: EphysDataService = field(default_factory=EphysDataService)
    slice_service: SliceService = field(default_factory=SliceService)
    workflow_policy: WorkflowPolicy = field(default_factory=WorkflowPolicy)
    alignment_repository: AlignmentRepository = field(
        default_factory=AlignmentRepository
    )
    plot_data_factory: PlotDataFactory = field(default_factory=PlotDataFactory)
    auto_alignments: dict[AutoAlignmentKey, AutoAlignment] = field(default_factory=dict)
    stream_cache: dict[str, ProbeSession] = field(default_factory=dict)
    current_stream_key: str | None = None
    loader: LoadDataLocal = field(init=False)
    controller: AlignmentController = field(init=False)

    def __post_init__(self) -> None:
        self.loader = LoadDataLocal(
            ephys_data_service=self.ephys_data_service,
            slice_service=self.slice_service,
        )
        self.controller = AlignmentController(
            self.document,
            self.loader,
            self.workflow_policy,
            alignment_repository=self.alignment_repository,
        )

    def cache_current_session(self, session: ProbeSession | None) -> None:
        """Store the active stream session under the current stream key."""
        if session is not None and self.current_stream_key is not None:
            self.stream_cache[self.current_stream_key] = session

    def clear_current_stream(self) -> None:
        """Forget which cached stream is currently displayed."""
        self.current_stream_key = None

    def set_current_stream(self, stream_key: str) -> None:
        """Mark a stream as the currently displayed stream."""
        self.current_stream_key = stream_key

    def cached_stream(self, stream_key: str) -> ProbeSession | None:
        """Return a cached stream session, if present."""
        return self.stream_cache.get(stream_key)

    def pop_cached_stream(self, stream_key: str) -> ProbeSession | None:
        """Remove and return a cached stream session, if present."""
        return self.stream_cache.pop(stream_key, None)

    def clear_stream_cache(self) -> list[ProbeSession]:
        """Clear stream cache ownership and return sessions for teardown."""
        sessions = list(self.stream_cache.values())
        self.stream_cache.clear()
        self.current_stream_key = None
        return sessions
