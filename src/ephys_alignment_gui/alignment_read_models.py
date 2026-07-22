"""Qt-free read models for alignment rendering."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentHistologyData,
    ChannelProjectionData,
)
from ephys_alignment_gui.document import AlignmentKey


@dataclass(frozen=True)
class ActiveAlignmentRenderState:
    """Derived data needed to render the active alignment."""

    key: AlignmentKey
    active_alignment: ActiveAlignment
    histology: AlignmentHistologyData
    projection: ChannelProjectionData
