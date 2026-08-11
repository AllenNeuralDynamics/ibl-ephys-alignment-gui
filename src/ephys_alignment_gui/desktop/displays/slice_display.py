"""Desktop slice display composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.displays.slice_panel_view import (
    SlicePanelView,
)


@dataclass(frozen=True)
class DesktopSliceDisplayConfig:
    """External style/callback dependencies needed to build the slice display."""

    depth_view: Any
    dotted_pen: Any
    solid_pen: Any
    reference_line_pen: Any
    set_axis: Callable[..., Any]
    padding_provider: Callable[[], float]
    histology_exists: Callable[[], bool]


@dataclass(frozen=True)
class DesktopSliceDisplay:
    """Own the slice panel view and pyqtgraph handle lifecycle."""

    view: SlicePanelView

    @classmethod
    def create(
        cls,
        *,
        config: DesktopSliceDisplayConfig,
        view_factory: Callable[..., SlicePanelView] = SlicePanelView.create,
    ) -> DesktopSliceDisplay:
        """Build the slice display cluster from desktop dependencies."""
        view = view_factory(
            depth_view=config.depth_view,
            padding=config.padding_provider(),
            set_axis=config.set_axis,
            dotted_pen=config.dotted_pen,
            solid_pen=config.solid_pen,
            reference_line_pen=config.reference_line_pen,
            histology_exists=config.histology_exists,
        )
        return cls(view=view)

    @property
    def area(self) -> Any:
        """Return the top-level coronal slice panel widget."""
        return self.view.plots.area

    @property
    def coronal_plot(self) -> Any:
        """Return the coronal slice plot handle."""
        return self.view.plots.coronal

    @property
    def perpendicular_plot(self) -> Any:
        """Return the perpendicular slice plot handle."""
        return self.view.plots.perpendicular

    def set_perpendicular_depth_link(self, linked_plot: Any) -> None:
        """Link the perpendicular slice y-axis to the histology depth plot."""
        self.view.set_perpendicular_depth_link(linked_plot)

    def capture_export_geometry(self) -> tuple[float, float, Any]:
        """Capture slice plot geometry for zoomed plot export."""
        return self.view.capture_export_geometry()

    def clear(self) -> None:
        """Clear slice-panel plot items and forget desktop handles."""
        self.view.clear()

    def toggle_channel_visibility(self) -> None:
        """Toggle channel, tip, trajectory, and perpendicular overlays."""
        self.view.toggle_channel_visibility()
