"""Tests for desktop slice display composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop.displays.slice_display import (
    DesktopSliceDisplay,
    DesktopSliceDisplayConfig,
)
from ephys_alignment_gui.desktop.displays.slice_panel_view import (
    SlicePanelPlots,
    SlicePanelStyle,
    SlicePanelView,
)


def test_slice_display_composes_slice_panel_view() -> None:
    captured_kwargs: dict[str, Any] = {}
    view = SlicePanelView(
        plots=SlicePanelPlots(
            coronal="coronal",
            coronal_layout=object(),
            histogram_alt=object(),
            perpendicular="perpendicular",
            area="area",
        ),
        style=SlicePanelStyle(
            dotted_pen=None,
            solid_pen=None,
            reference_line_pen=None,
        ),
        histology_exists=lambda: True,
    )

    display = DesktopSliceDisplay.create(
        config=DesktopSliceDisplayConfig(
            depth_view=SimpleNamespace(plot_y_range_um=(0.0, 1.0)),
            dotted_pen=None,
            solid_pen=None,
            reference_line_pen=None,
            set_axis=lambda *args, **kwargs: None,
            padding_provider=lambda: 0.0,
            histology_exists=lambda: True,
        ),
        view_factory=lambda **kwargs: captured_kwargs.update(kwargs) or view,
    )

    assert display.view is view
    assert display.area == "area"
    assert display.coronal_plot == "coronal"
    assert display.perpendicular_plot == "perpendicular"
    assert captured_kwargs["depth_view"].plot_y_range_um == (0.0, 1.0)
