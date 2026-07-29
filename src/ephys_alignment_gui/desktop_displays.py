"""Desktop display-region composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop_ephys_display import (
    DesktopEphysDisplay,
    DesktopEphysDisplayPorts,
)
from ephys_alignment_gui.desktop_histology_display import (
    DesktopHistologyDisplay,
    DesktopHistologyDisplayPorts,
)
from ephys_alignment_gui.desktop_reference_line_display import (
    DesktopReferenceLineDisplay,
    DesktopReferenceLineDisplayPorts,
)
from ephys_alignment_gui.desktop_slice_display import (
    DesktopSliceDisplay,
    DesktopSliceDisplayPorts,
)


@dataclass(frozen=True)
class DesktopDisplayPorts:
    """Desktop handles and callbacks needed to build display regions."""

    ephys: DesktopEphysDisplayPorts
    histology: DesktopHistologyDisplayPorts
    reference_lines: DesktopReferenceLineDisplayPorts
    slice: DesktopSliceDisplayPorts


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
        app: Any,
        ports: DesktopDisplayPorts,
    ) -> DesktopDisplays:
        """Build all desktop display regions from desktop ports."""
        return cls(
            ephys=DesktopEphysDisplay.create(app=app, ports=ports.ephys),
            histology=DesktopHistologyDisplay.create(app=app, ports=ports.histology),
            reference_lines=DesktopReferenceLineDisplay.create(
                ports=ports.reference_lines,
            ),
            slice=DesktopSliceDisplay.create(app=app, ports=ports.slice),
        )
