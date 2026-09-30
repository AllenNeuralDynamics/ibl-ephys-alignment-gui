"""Brain-surface candidates from interference-line amplitude.

Normalizes the producer's ``brain_surface.json``: every passing transition of
a line-amplitude fit per shank, none of them chosen as the surface. Depths are
probe-local ``channels.localCoordinates`` y in um.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_MAJOR = 1

# Span of a logistic between 10 % and 90 % of its rise, per unit scale.
_RISE_10_90 = math.log(9.0)


class BrainSurfaceError(ValueError):
    """``brain_surface.json`` does not match the supported schema."""


@dataclass(frozen=True)
class LineRow:
    """One amplitude row: a line as measured in one block."""

    line_index: int
    block_index: int
    freq_hz: float
    label: str | None
    excess_db: float | None


@dataclass(frozen=True)
class CandidateLine:
    """One line's part in a candidate transition of its fitted model."""

    line_index: int
    freq_hz: float
    step_db: float
    width_below_um: float
    width_above_um: float
    support: float
    agrees: bool
    baseline_db: dict[int, float]  # per block, at the tip
    trend_db_per_mm: float
    amplitude_rows: dict[int, int]  # per block

    def extent_um(self, midpoint_um: float) -> tuple[float, float]:
        """Depths spanning 10-90 % of this line's step about the midpoint."""
        return (
            midpoint_um - _RISE_10_90 * self.width_below_um,
            midpoint_um + _RISE_10_90 * self.width_above_um,
        )


@dataclass(frozen=True)
class SurfaceCandidate:
    """One passing transition.

    Candidates of one ``method`` on a shank come from one jointly fitted
    model: they share each line's baselines and trend.
    """

    method: str
    midpoint_um: float
    width_um: float | None
    interval_um: tuple[float, float]
    delta_bic: float
    n_agreeing_lines: int
    n_lines: int
    block_indices: tuple[int, ...]
    lines: tuple[CandidateLine, ...]


@dataclass(frozen=True)
class BrainSurface:
    """Normalized ``brain_surface.json``."""

    version: str
    lines: tuple[LineRow, ...]  # aligned with the amplitude rows
    candidates: dict[int, tuple[SurfaceCandidate, ...]]  # per shank

    def shank_candidates(self, shank_idx: int) -> tuple[SurfaceCandidate, ...]:
        """Candidates on one zero-based shank, strongest first."""
        return self.candidates.get(shank_idx, ())


def load_brain_surface(path: Path) -> BrainSurface | None:
    """Normalized candidates, or ``None`` when the file is absent or
    invalid; a bad file must not stop the probe loading."""
    if not path.is_file():
        return None
    try:
        return parse_brain_surface(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return None


def parse_brain_surface(raw: Any) -> BrainSurface:
    """Validate and normalize a decoded ``brain_surface.json``."""
    if not isinstance(raw, dict):
        raise BrainSurfaceError("expected a JSON object")
    version = str(raw.get("version", ""))
    try:
        major = int(version.split(".")[0])
    except ValueError as exc:
        raise BrainSurfaceError(f"unreadable version {version!r}") from exc
    if major != SUPPORTED_MAJOR:
        raise BrainSurfaceError(
            f"version {version} is not supported (major {SUPPORTED_MAJOR})"
        )
    try:
        lines = tuple(_line_row(entry) for entry in raw.get("lines") or ())
        candidates = {
            int(shank): tuple(
                _candidate(c) for c in (entry or {}).get("candidates") or ()
            )
            for shank, entry in (raw.get("shanks") or {}).items()
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise BrainSurfaceError(f"malformed entry: {exc}") from exc
    return BrainSurface(version=version, lines=lines, candidates=candidates)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _line_row(entry: dict[str, Any]) -> LineRow:
    return LineRow(
        line_index=int(entry["line_index"]),
        block_index=int(entry["block_index"]),
        freq_hz=float(entry["freq_hz"]),
        label=None if entry.get("label") is None else str(entry["label"]),
        excess_db=_optional_float(entry.get("excess_db")),
    )


def _candidate_line(entry: dict[str, Any]) -> CandidateLine:
    return CandidateLine(
        line_index=int(entry["line_index"]),
        freq_hz=float(entry["freq_hz"]),
        step_db=float(entry["step_db"]),
        width_below_um=float(entry["width_below_um"]),
        width_above_um=float(entry["width_above_um"]),
        support=float(entry["support"]),
        agrees=bool(entry["agrees"]),
        baseline_db={
            int(b): float(v) for b, v in (entry.get("baseline_db") or {}).items()
        },
        trend_db_per_mm=float(entry.get("trend_db_per_mm", 0.0)),
        amplitude_rows={
            int(b): int(r) for b, r in (entry.get("amplitude_rows") or {}).items()
        },
    )


def _candidate(entry: dict[str, Any]) -> SurfaceCandidate:
    interval = entry.get("interval_um") or (
        entry["midpoint_um"],
        entry["midpoint_um"],
    )
    return SurfaceCandidate(
        method=str(entry["method"]),
        midpoint_um=float(entry["midpoint_um"]),
        width_um=_optional_float(entry.get("width_um")),
        interval_um=(float(interval[0]), float(interval[1])),
        delta_bic=float(entry["delta_bic"]),
        n_agreeing_lines=int(entry["n_agreeing_lines"]),
        n_lines=int(entry["n_lines"]),
        block_indices=tuple(int(b) for b in entry.get("block_indices") or ()),
        lines=tuple(_candidate_line(line) for line in entry.get("lines") or ()),
    )
