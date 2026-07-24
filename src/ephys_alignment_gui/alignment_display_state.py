"""Qt-free display state for the alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RegionAnnotationSource = Literal["Allen", "FranklinPaxinos"]


@dataclass
class AlignmentDisplayState:
    """Frontend-agnostic display choices that do not change alignment state."""

    region_annotation_source: RegionAnnotationSource = "Allen"
    unit_filter: str = "all"

    def reset_region_annotation_source(self) -> None:
        """Reset displayed region annotations to Allen atlas labels."""
        self.region_annotation_source = "Allen"

    def reset_unit_filter(self) -> None:
        """Reset displayed ephys plots to all units."""
        self.unit_filter = "all"

    def set_unit_filter(self, unit_filter: str) -> str:
        """Set and return the unit subset used for displayed ephys plots."""
        self.unit_filter = unit_filter
        return self.unit_filter

    def toggle_region_annotation_source(self) -> RegionAnnotationSource:
        """Toggle and return the displayed region annotation source."""
        self.region_annotation_source = (
            "FranklinPaxinos" if self.region_annotation_source == "Allen" else "Allen"
        )
        return self.region_annotation_source
