"""Qt-free display state for the alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

RegionAnnotationSource = Literal["Allen", "FranklinPaxinos"]


@dataclass
class DepthViewSettings:
    """Feature-depth display and fit-grid settings."""

    probe_tip_um: float = 0.0
    probe_top_um: float = 3840.0
    probe_extra_um: float = 100.0
    view_min_um: float = -2000.0
    view_max_um: float = 6000.0
    fit_depth_step_um: float = 20.0

    @property
    def view_range_um(self) -> tuple[float, float]:
        """Return the full fit-panel view range in micrometres."""
        return self.view_min_um, self.view_max_um

    @property
    def plot_y_range_um(self) -> tuple[float, float]:
        """Return default data-panel depth limits in micrometres."""
        return (
            self.probe_tip_um - self.probe_extra_um,
            self.probe_top_um + self.probe_extra_um,
        )

    @property
    def fit_depth_um(self) -> NDArray[np.float64]:
        """Return the depth grid used for linear fit visualization."""
        return np.arange(
            self.view_min_um,
            self.view_max_um,
            self.fit_depth_step_um,
            dtype=float,
        )

    def set_probe_limits(self, probe_tip_um: float, probe_top_um: float) -> None:
        """Update the probe tip/top display markers."""
        self.probe_tip_um = float(probe_tip_um)
        self.probe_top_um = float(probe_top_um)


@dataclass
class AlignmentEditSettings:
    """Frontend-agnostic edit options used by alignment commands."""

    lin_fit: bool = True
    extend_feature: int = 1

    def set_lin_fit(self, enabled: bool) -> bool:
        """Set and return whether fit commands should use linear fitting."""
        self.lin_fit = bool(enabled)
        return self.lin_fit


@dataclass
class AlignmentDisplayState:
    """Frontend-agnostic display choices that do not change alignment state."""

    region_annotation_source: RegionAnnotationSource = "Allen"
    unit_filter: str = "all"
    reference_lines_visible: bool = True
    histology_boundaries_visible: bool = True
    depth_view: DepthViewSettings = field(default_factory=DepthViewSettings)
    edit_settings: AlignmentEditSettings = field(
        default_factory=AlignmentEditSettings
    )

    def reset_region_annotation_source(self) -> None:
        """Reset displayed region annotations to Allen atlas labels."""
        self.region_annotation_source = "Allen"

    def reset_unit_filter(self) -> None:
        """Reset displayed ephys plots to all units."""
        self.unit_filter = "all"

    def reset_visibility_toggles(self) -> None:
        """Reset simple plot visibility toggles to defaults."""
        self.reference_lines_visible = True
        self.histology_boundaries_visible = True

    def reset_depth_view(self) -> None:
        """Reset feature-depth display settings to defaults."""
        self.depth_view = DepthViewSettings()

    def reset_edit_settings(self) -> None:
        """Reset alignment edit settings to defaults."""
        self.edit_settings = AlignmentEditSettings()

    def reset_for_active_stream(self) -> None:
        """Reset frontend-agnostic display state for a stream transition."""
        self.reset_region_annotation_source()
        self.reset_unit_filter()
        self.reset_visibility_toggles()
        self.reset_depth_view()
        self.reset_edit_settings()

    def set_unit_filter(self, unit_filter: str) -> str:
        """Set and return the unit subset used for displayed ephys plots."""
        self.unit_filter = unit_filter
        return self.unit_filter

    def toggle_reference_lines_visible(self) -> bool:
        """Toggle and return whether reference lines are visible."""
        self.reference_lines_visible = not self.reference_lines_visible
        return self.reference_lines_visible

    def toggle_histology_boundaries_visible(self) -> bool:
        """Toggle and return whether reference histology boundaries are visible."""
        self.histology_boundaries_visible = not self.histology_boundaries_visible
        return self.histology_boundaries_visible

    def toggle_region_annotation_source(self) -> RegionAnnotationSource:
        """Toggle and return the displayed region annotation source."""
        self.region_annotation_source = (
            "FranklinPaxinos" if self.region_annotation_source == "Allen" else "Allen"
        )
        return self.region_annotation_source
