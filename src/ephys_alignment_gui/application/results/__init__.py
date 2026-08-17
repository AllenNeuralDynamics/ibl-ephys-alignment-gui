"""Qt-free app port DTOs and command results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results.alignment_persistence import (
    AlignmentOutputsSaved,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointCleared as AutosaveCheckpointCleared,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointInspected as AutosaveCheckpointInspected,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointRead as AutosaveCheckpointRead,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointRecovered as AutosaveCheckpointRecovered,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointRestored as AutosaveCheckpointRestored,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointSkippedKey as AutosaveCheckpointSkippedKey,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointWritten as AutosaveCheckpointWritten,
)
from ephys_alignment_gui.application.results.metadata import ProbeSelected
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.results import (
    AlignmentChoicesUpdated as AlignmentChoicesUpdated,
)
from ephys_alignment_gui.core.results import (
    AlignmentEditApplied as AlignmentEditApplied,
)
from ephys_alignment_gui.core.results import (
    AlignmentEditNoop as AlignmentEditNoop,
)
from ephys_alignment_gui.core.results import (
    LoadDataPrepared as LoadDataPrepared,
)
from ephys_alignment_gui.core.results import (
    PendingReferenceLinesUpdated as PendingReferenceLinesUpdated,
)
from ephys_alignment_gui.core.results import (
    PreviousAlignmentSelected as PreviousAlignmentSelected,
)
from ephys_alignment_gui.core.results import (
    ShankRuntimeInitialized as ShankRuntimeInitialized,
)
from ephys_alignment_gui.core.results import (
    ShankSelected as ShankSelected,
)
from ephys_alignment_gui.io.load_data_job import (
    LoadDataCancelToken,
    LoadDataJobRequest,
)
from ephys_alignment_gui.io.load_data_target import LoadDataJobTarget
from ephys_alignment_gui.runtime.ephys_stream import StreamKey
from ephys_alignment_gui.runtime.histology_loader import HistologyLoadResult


@dataclass(frozen=True)
class ShankSelectionState:
    """Read model for the active shank selection."""

    shank_idx: int
    shank_id: int
    alignment_key: AlignmentKey | None
    data_loaded: bool


@dataclass(frozen=True)
class ActiveProbeSelectionState:
    """Read model for active probe/shank selector presentation."""

    recording_id: str
    probe_name: str
    shanks: list[str]
    n_shanks: int
    output_directory: Path | None


@dataclass(frozen=True)
class FreshEphysDataLoaded:
    """Fresh ephys stream data was loaded and cached."""

    stream_runtime: Any
    shank_idx: int


@dataclass(frozen=True)
class CachedEphysDataActivated:
    """Cached ephys stream runtime was activated."""

    stream_runtime: Any
    shank_idx: int
    probe: ProbeSelected


@dataclass(frozen=True)
class ActiveStreamDetached:
    """The active stream was detached while cached runtimes were preserved."""

    cached_stream_count: int


@dataclass(frozen=True)
class StreamCacheEvicted:
    """Cached stream runtimes were evicted for a recording/session transition."""

    evicted_stream_count: int


@dataclass(frozen=True)
class LoadedShankPrepared:
    """Runtime state for one loaded shank is ready for rendering."""

    shank_idx: int
    n_channels: int
    histology_available: bool
    alignment_choices: list[str] | None = None


@dataclass(frozen=True)
class EditedAlignmentOutputsSaved:
    """Edited alignment outputs were persisted."""

    saved_count: int
    saved_outputs: Mapping[AlignmentKey, AlignmentOutputsSaved]
    active_choices: list[str] | None


@dataclass(frozen=True)
class LoadDataAlreadyActiveResult:
    """The requested stream/shank is already active; no load work ran."""

    stream_key: StreamKey | None
    shank_idx: int


@dataclass(frozen=True)
class LoadDataCachedActivated:
    """A cached stream was activated for desktop presentation."""

    stream_key: StreamKey
    activated: CachedEphysDataActivated


@dataclass(frozen=True)
class LoadDataFreshPrepared:
    """Fresh load state was prepared and is ready for heavy IO."""

    stream_key: StreamKey | None
    shank_idx: int
    preserve_plot_selection: bool
    target: LoadDataJobTarget


@dataclass(frozen=True)
class FreshLoadExecution:
    """Fresh-load execution handle held by UI callers."""

    load_id: int
    prepared: LoadDataFreshPrepared


@dataclass(frozen=True)
class FreshLoadJobInvocation:
    """Runnable fresh-load job request for an active execution."""

    execution: FreshLoadExecution
    request: LoadDataJobRequest
    cancel_token: LoadDataCancelToken


@dataclass(frozen=True)
class LoadDataStaleResultIgnored:
    """A completed load result was ignored because the request is obsolete."""

    load_id: int
    stream_key: StreamKey | None
    shank_idx: int
    reason: str


@dataclass(frozen=True)
class LoadDataFreshRequiredResult:
    """The requested stream is not cached and requires an explicit fresh load."""

    stream_key: StreamKey | None
    shank_idx: int


@dataclass(frozen=True)
class LoadDataPreloadSkipped:
    """A background preload request did not need to run."""

    stream_key: StreamKey | None
    shank_idx: int
    reason: str


@dataclass(frozen=True)
class LoadDataFreshCompleted:
    """Fresh ephys data and subject histology load steps completed."""

    stream_key: StreamKey | None
    target: LoadDataJobTarget
    ephys: FreshEphysDataLoaded
    histology: HistologyLoadResult
    preserve_plot_selection: bool


LoadDataBeginResult = (
    LoadDataAlreadyActiveResult | LoadDataCachedActivated | LoadDataFreshPrepared
)

ProbeSelectionCacheResult = (
    LoadDataAlreadyActiveResult | LoadDataCachedActivated | LoadDataFreshRequiredResult
)
