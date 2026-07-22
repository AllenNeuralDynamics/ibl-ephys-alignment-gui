"""Qt-free display state for the alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RegionAnnotationSource = Literal["Allen", "FranklinPaxinos"]


@dataclass
class AlignmentDisplayState:
    """Frontend-agnostic display choices that do not change alignment state."""

    region_annotation_source: RegionAnnotationSource = "Allen"

    def reset_region_annotation_source(self) -> None:
        """Reset displayed region annotations to Allen atlas labels."""
        self.region_annotation_source = "Allen"

    def toggle_region_annotation_source(self) -> RegionAnnotationSource:
        """Toggle and return the displayed region annotation source."""
        self.region_annotation_source = (
            "FranklinPaxinos"
            if self.region_annotation_source == "Allen"
            else "Allen"
        )
        return self.region_annotation_source
