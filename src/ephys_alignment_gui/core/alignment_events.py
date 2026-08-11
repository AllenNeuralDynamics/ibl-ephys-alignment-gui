"""Typed GUI events for active alignment changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
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


@dataclass(frozen=True)
class FreshLoadCompleted:
    """Payload emitted after fresh heavy ephys/histology IO completes."""

    stream_key: tuple[str, str] | None
    shank_idx: int
    warning_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadDataFailed:
    """Payload emitted when a fresh load or activation fails."""

    stream_key: tuple[str, str] | None
    shank_idx: int
    message: str


@dataclass(frozen=True)
class LoadDataCancelled:
    """Payload emitted when a fresh load is cancelled."""

    stream_key: tuple[str, str] | None
    shank_idx: int
    reason: str


@dataclass(frozen=True)
class HistologyLoadReported:
    """Payload emitted after fresh-load histology availability is known."""

    stream_key: tuple[str, str] | None
    shank_idx: int
    status: HistologyLoadStatus
    message: str | None = None
