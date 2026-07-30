"""Metadata-selection command result DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MouseRootLoaded:
    """A mouse root was loaded and the document was updated."""

    mouse_root: Any
    root_changed: bool


@dataclass(frozen=True)
class RecordingSelected:
    """A recording was selected and its probe choices are available."""

    recording_id: str
    probes: list[str]


@dataclass(frozen=True)
class ProbeSelected:
    """A probe was selected and channel metadata is ready."""

    recording_id: str
    probe_name: str
    shanks: list[str]
    n_shanks: int
    output_directory: Path | None
