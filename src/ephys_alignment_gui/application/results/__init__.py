"""Qt-free app port DTOs and command results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_state import PendingReferenceLines
from ephys_alignment_gui.application.results.alignment_persistence import (
    AlignmentOutputsSaved,
)
from ephys_alignment_gui.application.results.metadata import ProbeSelected
from ephys_alignment_gui.document import AlignmentKey
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
class LoadDataPrepared:
    """State needed by a UI before heavy data loading starts."""

    preserve_plot_selection: bool


@dataclass(frozen=True)
class ShankSelected:
    """The active shank selection changed in the document."""

    previous_key: AlignmentKey | None
    selected_key: AlignmentKey | None
    previous_shank_idx: int
    shank_idx: int
    data_loaded: bool


@dataclass(frozen=True)
class AlignmentChoicesUpdated:
    """Alignment dropdown choices were updated for the active state."""

    choices: list[str]


@dataclass(frozen=True)
class PreviousAlignmentSelected:
    """A previous/original alignment choice was selected."""

    feature_prev: Any
    track_prev: Any
    choice: str | None
    choices: list[str]


@dataclass(frozen=True)
class PendingReferenceLinesUpdated:
    """Pending reference-line coordinates were updated for the active state."""

    lines: PendingReferenceLines | None


@dataclass(frozen=True)
class AlignmentEditApplied:
    """An editable alignment command changed the active alignment."""

    alignment: ActiveAlignment
    lin_fit: bool | None = None


@dataclass(frozen=True)
class AlignmentEditNoop:
    """An editable alignment command completed without changing state."""


@dataclass(frozen=True)
class ShankRuntimeInitialized:
    """Runtime alignment engine was initialized for one shank."""

    feature_init: Any
    track_init: Any
    track_annos_and_ends_ras: Any
    seeded_document_alignment: bool


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
class VisitedAlignmentOutputsSaved:
    """Visited alignment outputs were persisted."""

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


@dataclass(frozen=True)
class LoadDataFreshRequiredResult:
    """The requested stream is not cached and requires an explicit fresh load."""

    stream_key: StreamKey | None
    shank_idx: int


@dataclass(frozen=True)
class LoadDataFreshCompleted:
    """Fresh ephys data and subject histology load steps completed."""

    stream_key: StreamKey | None
    ephys: FreshEphysDataLoaded
    histology: HistologyLoadResult
    preserve_plot_selection: bool


LoadDataBeginResult = (
    LoadDataAlreadyActiveResult | LoadDataCachedActivated | LoadDataFreshPrepared
)

ProbeSelectionCacheResult = (
    LoadDataAlreadyActiveResult | LoadDataCachedActivated | LoadDataFreshRequiredResult
)
