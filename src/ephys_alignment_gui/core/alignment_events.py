"""Typed GUI events for active alignment changes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_display_state import RegionAnnotationSource
from ephys_alignment_gui.core.document import AlignmentKey

AlignmentEditKind = Literal[
    "fit",
    "offset",
    "next",
    "previous",
    "reset",
]


@dataclass(frozen=True)
class AlignmentEdited:
    """Application event emitted after editable alignment state changes.

    This is the durable command-level event. It intentionally carries no
    plot-specific payloads or desktop refresh instructions.
    """

    edit_kind: AlignmentEditKind
    active_key: AlignmentKey
    active_alignment: ActiveAlignment
    lin_fit: bool | None = None


@dataclass(frozen=True)
class ShankChanged:
    """Payload emitted after the active shank selection has changed."""

    source: str
    previous_shank_idx: int
    shank_idx: int
    previous_key: AlignmentKey | None
    active_key: AlignmentKey | None
    data_loaded: bool
    preserve_plot_selection: bool | None = None


@dataclass(frozen=True)
class ReferenceLineVisibilityChanged:
    """Payload emitted after reference-line visibility display state changes."""

    visible: bool


@dataclass(frozen=True)
class HistologyBoundariesVisibilityChanged:
    """Payload emitted after histology boundary display state changes."""

    visible: bool


@dataclass(frozen=True)
class RegionAnnotationSourceChanged:
    """Payload emitted after the displayed region annotation source changes."""

    source: RegionAnnotationSource


@dataclass(frozen=True)
class OutputRootChanged:
    """Payload emitted after the output root path changes."""

    output_root: Path
    output_directory: Path | None


@dataclass(frozen=True)
class OutputDirectoryChanged:
    """Payload emitted after the active per-probe output directory is refreshed."""

    output_root: Path | None
    output_directory: Path | None


StreamActivationSource = Literal["cached", "fresh"]
LoadDataPhase = Literal["ephys", "histology", "complete", "cancelled"]
LoadDataStatus = Literal["started", "completed", "warning", "cancelled"]
HistologyLoadStatus = Literal["already_loaded", "loaded", "unavailable"]


@dataclass(frozen=True)
class StreamActivated:
    """Payload emitted after a stream/shank becomes active runtime state."""

    source: StreamActivationSource
    stream_key: tuple[str, str] | None
    shank_idx: int
    active_key: AlignmentKey | None
    preserve_plot_selection: bool | None = None
    load_id: int | None = None


@dataclass(frozen=True)
class StreamDetached:
    """Payload emitted after the active stream is detached from runtime state."""

    cached_stream_count: int


@dataclass(frozen=True)
class StreamCacheEvicted:
    """Payload emitted after cached stream runtimes are evicted."""

    evicted_stream_count: int


@dataclass(frozen=True)
class LoadDataProgressed:
    """Payload emitted as a fresh stream load advances."""

    stream_key: tuple[str, str] | None
    shank_idx: int
    phase: LoadDataPhase
    status: LoadDataStatus
    message: str
    load_id: int | None = None


@dataclass(frozen=True)
class FreshLoadCompleted:
    """Payload emitted after fresh heavy ephys/histology IO completes."""

    stream_key: tuple[str, str] | None
    shank_idx: int
    warning_messages: tuple[str, ...] = ()
    load_id: int | None = None


@dataclass(frozen=True)
class LoadDataFailed:
    """Payload emitted when a fresh load or activation fails."""

    stream_key: tuple[str, str] | None
    shank_idx: int
    message: str
    load_id: int | None = None


@dataclass(frozen=True)
class LoadDataCancelled:
    """Payload emitted when a fresh load is cancelled."""

    stream_key: tuple[str, str] | None
    shank_idx: int
    reason: str
    load_id: int | None = None


@dataclass(frozen=True)
class HistologyLoadReported:
    """Payload emitted after fresh-load histology availability is known."""

    stream_key: tuple[str, str] | None
    shank_idx: int
    status: HistologyLoadStatus
    message: str | None = None
    load_id: int | None = None


@dataclass(frozen=True)
class SaveDocDbStatus:
    """DocDB write status for one saved alignment output."""

    probe_name: str
    error: str | None = None


SaveProgressPhase = Literal[
    "preparing",
    "rehydrating",
    "building_outputs",
    "writing_files",
]
SaveProgressStatus = Literal["started", "completed", "running", "cancelled"]


@dataclass(frozen=True)
class SaveProgressStarted:
    """Payload emitted when an edited-alignment save transaction starts."""

    targets: tuple[AlignmentKey, ...]
    message: str


@dataclass(frozen=True)
class SaveProgressUpdated:
    """Payload emitted as an edited-alignment save transaction advances."""

    key: AlignmentKey | None
    phase: SaveProgressPhase
    status: SaveProgressStatus
    completed: int
    total: int
    message: str


@dataclass(frozen=True)
class SaveCompleted:
    """Payload emitted after edited alignment outputs are persisted."""

    saved_count: int
    active_choices: tuple[str, ...] | None = None
    docdb_statuses: tuple[SaveDocDbStatus, ...] = ()


@dataclass(frozen=True)
class SaveFailed:
    """Payload emitted when edited alignment output persistence fails."""

    message: str


@dataclass(frozen=True)
class SaveCancelled:
    """Payload emitted when edited alignment output persistence is cancelled."""

    reason: str
    message: str


@dataclass(frozen=True)
class PreviousAlignmentsLoaded:
    """Payload emitted after previous alignments are loaded into the document."""

    shank_idx: int
    choices: tuple[str, ...]


@dataclass(frozen=True)
class PreviousAlignmentsUnavailable:
    """Payload emitted when no previous alignments are available."""

    shank_idx: int


@dataclass(frozen=True)
class PreviousAlignmentLoadFailed:
    """Payload emitted when previous-alignment loading fails."""

    shank_idx: int | None
    message: str
