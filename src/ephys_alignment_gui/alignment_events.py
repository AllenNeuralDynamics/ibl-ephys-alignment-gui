"""Typed GUI events for active alignment changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentHistologyData,
    ChannelProjectionData,
)

LineUpdateMode = Literal["none", "navigation", "sync", "reset_previous"]


@dataclass(frozen=True)
class AlignmentChanged:
    """Payload emitted after the active alignment has changed."""

    source: str
    active_alignment: ActiveAlignment | None
    histology: AlignmentHistologyData
    projection: ChannelProjectionData
    line_update: LineUpdateMode = "none"
    reset_histology_range: bool = False
    refresh_perpendicular: bool = True
    update_status: bool = True
