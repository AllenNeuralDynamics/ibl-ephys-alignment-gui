"""Desktop reference-line overlay display composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from random import randrange
from typing import Any

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui

from ephys_alignment_gui.desktop.displays.reference_line_layer import (
    ReferenceLineLayer,
    ReferenceLinePlots,
)


@dataclass(frozen=True)
class ReferenceLinePlotBindings:
    """Sibling display plot handles that receive linked reference-line overlays."""

    histology_plot: Any
    reference_plot: Any
    image_plot: Any
    line_plot: Any
    probe_plot: Any
    perpendicular_plot: Any
    fit_plot: Any


@dataclass(frozen=True)
class DesktopReferenceLineDisplay:
    """Own linked reference-line overlay handles across desktop plots."""

    layer: ReferenceLineLayer

    @classmethod
    def create(
        cls,
        *,
        bindings: ReferenceLinePlotBindings,
    ) -> DesktopReferenceLineDisplay:
        """Build reference-line overlays from desktop plot handles."""
        return cls(
            layer=ReferenceLineLayer(
                plots=ReferenceLinePlots(
                    histology=bindings.histology_plot,
                    reference=bindings.reference_plot,
                    image=bindings.image_plot,
                    line=bindings.line_plot,
                    probe=bindings.probe_plot,
                    perpendicular=bindings.perpendicular_plot,
                    fit=bindings.fit_plot,
                ),
                style_factory=default_reference_line_style,
                on_lines_changed=lambda: None,
            )
        )

    def set_lines_changed_callback(self, callback: Callable[[], None]) -> None:
        """Set the callback invoked when reference-line positions change."""
        self.layer.set_on_lines_changed(callback)

    def set_track_display_transform(
        self,
        *,
        track_to_warped_position: Callable[[Any], Any],
        warped_position_to_track: Callable[[Any], Any],
    ) -> None:
        """Set conversion callbacks for warped track-space overlay handles."""
        self.layer.set_track_display_transform(
            track_to_warped_position=track_to_warped_position,
            warped_position_to_track=warped_position_to_track,
        )

    def has_lines(self) -> bool:
        """Return whether the display has reference-line handles."""
        return self.layer.has_lines()

    def positions(self) -> Any:
        """Return feature/track reference-line positions in um."""
        return self.layer.positions()

    def clear(self) -> None:
        """Remove, disconnect, and forget all reference-line handles."""
        self.layer.clear()

    def remove_from_plots(self) -> None:
        """Remove current line handles from plots without deleting them."""
        self.layer.remove_from_plots()

    def add_to_plots(self) -> None:
        """Add current line handles back to their plots."""
        self.layer.add_to_plots()

    def reattach(self) -> None:
        """Refresh current reference-line handles on their plots."""
        self.remove_from_plots()
        self.add_to_plots()

    def create_lines(self, positions: Any, track_positions: Any = None) -> None:
        """Create linked feature/track reference lines."""
        self.layer.create_lines(positions, track_positions)

    def replace_lines(self, positions: Any, track_positions: Any = None) -> None:
        """Replace linked feature/track reference lines without user-edit capture."""
        self.layer.replace_lines(positions, track_positions)

    def sync_track_to_feature(self) -> None:
        """Move track-space reference lines to current feature-line positions."""
        self.layer.sync_track_to_feature()

    def select_line(self, line: Any) -> bool:
        """Select a managed reference-line handle."""
        return self.layer.select_line(line)

    def clear_selection(self) -> None:
        """Clear selected reference-line handle."""
        self.layer.clear_selection()

    def delete_selected(self) -> bool:
        """Delete the selected reference-line group."""
        return self.layer.delete_selected()


def default_reference_line_style() -> tuple[Any, Any]:
    """Create a random reference-line pen and brush."""
    colours = [
        "#cc0000",
        "#6aa84f",
        "#ff8d00",
        "#00FFF7",
        "#03fc84",
        "#fc03e7",
        "#1c03fc",
        "#000000",
    ]
    styles = [
        QtCore.Qt.SolidLine,
        QtCore.Qt.DashLine,
        QtCore.Qt.DashDotLine,
    ]
    colour = QtGui.QColor(colours[randrange(len(colours))])
    style = styles[randrange(len(styles))]
    return (
        pg.mkPen(color=colour, style=style, width=2),
        pg.mkBrush(color=colour),
    )
