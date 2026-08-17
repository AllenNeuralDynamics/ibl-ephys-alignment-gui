"""Tests for desktop histology display composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop.displays.histology_display import (
    DesktopHistologyDisplay,
    DesktopHistologyDisplayConfig,
)
from ephys_alignment_gui.desktop.displays.histology_panel_view import (
    FitPanelItems,
    HistologyPanelAxes,
    HistologyPanelPlots,
    HistologyPanelStyle,
    HistologyPanelView,
)


class FakeLayout:
    def scene(self) -> str:
        return "histology-scene"


def _panel() -> HistologyPanelView:
    return HistologyPanelView(
        plots=HistologyPanelPlots(
            aligned="aligned",
            reference="reference",
            scale="scale",
            scale_colorbar="scale-colorbar",
            area="area",
            layout=FakeLayout(),
            depth_ruler="depth-ruler",
            scale_axis="scale-axis",
        ),
        axes=HistologyPanelAxes(
            aligned="aligned-axis",
            reference="reference-axis",
        ),
        style=HistologyPanelStyle(dotted_pen="dotted"),
        set_axis=lambda *_args, **_kwargs: None,
        padding_provider=lambda: 0.05,
        fit_items=FitPanelItems(
            fit_curve=SimpleNamespace(setData=lambda **_kwargs: None),
            fit_scatter=SimpleNamespace(setData=lambda **_kwargs: None),
            linear_fit_curve=SimpleNamespace(setData=lambda **_kwargs: None),
            plot_widget="fit-plot",
            linear_fit_checkbox="linear-fit-checkbox",
        ),
    )


def _config() -> DesktopHistologyDisplayConfig:
    return DesktopHistologyDisplayConfig(
        depth_view=SimpleNamespace(plot_y_range_um=(0.0, 1.0)),
        dotted_pen="dotted",
        fit_pen="fit-pen",
        linear_fit_pen="linear-fit-pen",
        baseline_pen="baseline-pen",
        set_axis=lambda *_args, **_kwargs: None,
        padding_provider=lambda: 0.05,
        on_linear_fit_changed=lambda *_args, **_kwargs: None,
        on_mouse_double_clicked=lambda *_args, **_kwargs: None,
        on_mouse_hover=lambda *_args, **_kwargs: None,
        linear_fit_enabled=lambda: False,
    )


def test_histology_display_composes_panel_handles() -> None:
    captured_kwargs: dict[str, Any] = {}
    panel = _panel()

    display = DesktopHistologyDisplay.create(
        config=_config(),
        perpendicular_plot="perpendicular",
        view_factory=lambda **kwargs: captured_kwargs.update(kwargs) or panel,
    )

    assert display.panel is panel
    assert display.aligned_plot == "aligned"
    assert display.reference_plot == "reference"
    assert display.depth_ruler == "depth-ruler"
    assert display.export_scene() == "histology-scene"
    assert captured_kwargs["perpendicular_plot"] == "perpendicular"
    assert captured_kwargs["depth_view"].plot_y_range_um == (0.0, 1.0)
