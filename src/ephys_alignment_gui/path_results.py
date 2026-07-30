"""Path command result DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputRootSet:
    """The output root was stored and the per-probe output was refreshed."""

    output_root: Path
    output_directory: Path | None


@dataclass(frozen=True)
class OutputDirectoryDerived:
    """The per-probe output directory was refreshed."""

    output_directory: Path | None
