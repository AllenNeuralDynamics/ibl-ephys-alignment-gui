"""Desktop display-region composition."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.desktop.displays.ephys_display import (
    DesktopEphysDisplay,
    DesktopEphysDisplayConfig,
)
from ephys_alignment_gui.desktop.displays.histology_display import (
    DesktopHistologyDisplay,
    DesktopHistologyDisplayConfig,
)
from ephys_alignment_gui.desktop.displays.reference_line_display import (
    DesktopReferenceLineDisplay,
    ReferenceLinePlotBindings,
)
from ephys_alignment_gui.desktop.displays.slice_display import (
    DesktopSliceDisplay,
    DesktopSliceDisplayConfig,
)


@dataclass(frozen=True)
class DesktopDisplayConfig:
    """External style/callback dependencies needed to build display regions."""

    ephys: DesktopEphysDisplayConfig
    histology: DesktopHistologyDisplayConfig
    slice: DesktopSliceDisplayConfig


@dataclass(frozen=True)
class DesktopDisplays:
    """Own concrete desktop display-region clusters."""

    ephys: DesktopEphysDisplay
    histology: DesktopHistologyDisplay
    reference_lines: DesktopReferenceLineDisplay
    slice: DesktopSliceDisplay

    @classmethod
    def create(
        cls,
        *,
        config: DesktopDisplayConfig,
    ) -> DesktopDisplays:
        """Build all desktop display regions from desktop dependencies."""
        ephys = DesktopEphysDisplay.create(config=config.ephys)
        slice_display = DesktopSliceDisplay.create(config=config.slice)
        histology = DesktopHistologyDisplay.create(
            config=config.histology,
            perpendicular_plot=slice_display.perpendicular_plot,
        )
        _link_depth_plots(
            ephys=ephys,
            histology=histology,
            slice_display=slice_display,
        )
        reference_lines = DesktopReferenceLineDisplay.create(
            bindings=ReferenceLinePlotBindings(
                histology_plot=histology.aligned_plot,
                reference_plot=histology.reference_plot,
                image_plot=ephys.panel.plots.image,
                line_plot=ephys.panel.plots.line,
                probe_plot=ephys.panel.plots.probe,
                perpendicular_plot=slice_display.perpendicular_plot,
                fit_plot=histology.fit_plot,
            )
        )
        return cls(
            ephys=ephys,
            histology=histology,
            reference_lines=reference_lines,
            slice=slice_display,
        )


def _link_depth_plots(
    *,
    ephys: DesktopEphysDisplay,
    histology: DesktopHistologyDisplay,
    slice_display: DesktopSliceDisplay,
) -> None:
    """Link pyqtgraph y-axes across the desktop depth plots."""
    ephys.panel.plots.image.setYLink(ephys.panel.plots.line)
    ephys.panel.plots.image.setYLink(histology.aligned_plot)
    ephys.panel.plots.line.setYLink(histology.aligned_plot)
    ephys.panel.plots.probe.setYLink(ephys.panel.plots.image)
    slice_display.set_perpendicular_depth_link(histology.aligned_plot)
