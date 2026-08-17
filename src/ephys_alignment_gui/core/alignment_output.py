"""Qt-free contracts for alignment output construction and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChannelOutputIdentity:
    """Channel-table identity fields aligned with saved channel coordinates."""

    raw_ind: Any | None = None
    contact_id: Any | None = None
    shank_idx: Any | None = None

    def with_defaults(
        self,
        length: int,
        *,
        default_shank_idx: int = 0,
    ) -> ChannelOutputIdentity:
        """Return identity vectors suitable for serializing ``length`` channels."""
        return ChannelOutputIdentity(
            raw_ind=self._vector_or_default(
                self.raw_ind,
                np.arange(length, dtype=int),
                length=length,
                name="raw_ind",
            ),
            contact_id=self._optional_vector(
                self.contact_id,
                length=length,
                name="contact_id",
            ),
            shank_idx=self._vector_or_default(
                self.shank_idx,
                np.full(length, default_shank_idx, dtype=int),
                length=length,
                name="shank_idx",
            ),
        )

    @staticmethod
    def _vector_or_default(
        value: Any | None,
        default: Any,
        *,
        length: int,
        name: str,
    ) -> Any:
        if value is None:
            return default
        return ChannelOutputIdentity._optional_vector(value, length=length, name=name)

    @staticmethod
    def _optional_vector(value: Any | None, *, length: int, name: str) -> Any | None:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim != 1 or arr.shape[0] != length:
            raise ValueError(f"{name} must be a 1D vector with {length} entries")
        return arr


@dataclass(frozen=True)
class AlignmentOutputInput:
    """Runtime-derived channel data needed to build alignment output dictionaries."""

    channel_locations_ras: Any
    channel_coordinates: Any
    channel_identity: ChannelOutputIdentity | None = None


@dataclass(frozen=True)
class CcfExportIssue:
    """Summary of why CCF channel output was omitted for some rows."""

    reason: str
    message: str
    channel_count: int
    ml_range_mm: tuple[float, float] | None = None
    bounds_ml_mm: tuple[float, float] | None = None


@dataclass(frozen=True)
class CcfExportStatus:
    """Per-shank CCF export status written next to alignment outputs."""

    status: str
    total_channel_count: int
    ccf_channel_count: int
    omitted_channel_count: int
    in_brain_channel_count: int
    bounds_ml_mm: tuple[float, float] | None = None
    in_brain_ml_range_mm: tuple[float, float] | None = None
    issues: tuple[CcfExportIssue, ...] = ()


@dataclass(frozen=True)
class AlignmentOutputMetadata:
    """Probe/shank identity written next to GUI alignment output files."""

    recording_id: str
    ephys_collection: str
    logical_probe: str
    shank_idx: int
    n_shanks: int
    probe_id: str | None = None
    ccf_export: CcfExportStatus | None = None
