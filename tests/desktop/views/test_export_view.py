"""Tests for desktop export view composition helpers."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.desktop.views.export_view import DesktopExportView


def _view() -> tuple[DesktopExportView, dict[str, Any]]:
    calls: dict[str, Any] = {
        "reset_axis": 0,
        "set_view": [],
        "set_axis": [],
        "set_font": [],
        "add_lines": 0,
    }
    view = DesktopExportView(
        ephys_graphics_layout="graphics-layout",
        ephys_data_area="data-area",
        slice_plot="slice-plot",
        slice_trajectory_pen="trajectory-pen",
        reset_axis=lambda: calls.__setitem__(
            "reset_axis", calls["reset_axis"] + 1
        ),
        set_view=lambda **kwargs: calls["set_view"].append(kwargs),
        set_axis=lambda *args, **kwargs: calls["set_axis"].append((args, kwargs)),
        set_font=lambda *args, **kwargs: calls["set_font"].append((args, kwargs)),
        ephys_sizes=lambda: (30.0, 5.0),
        slice_geometry=lambda: (100.0, 200.0, "rect"),
    )
    return view, calls


def test_export_view_builds_ephys_layout_and_callbacks() -> None:
    view, calls = _view()
    callbacks = view.ephys_callbacks(
        add_lines_points=lambda: calls.__setitem__(
            "add_lines", calls["add_lines"] + 1
        )
    )

    layout = view.ephys_layout()
    sizes = callbacks.sizes()
    callbacks.reset_axis()
    callbacks.set_view(view=1, configure=False)
    callbacks.add_lines_points()

    assert layout.graphics_layout == "graphics-layout"
    assert layout.data_area == "data-area"
    assert sizes.probe_width == 30.0
    assert sizes.axis_width == 5.0
    assert calls["reset_axis"] == 1
    assert calls["set_view"] == [{"view": 1, "configure": False}]
    assert calls["add_lines"] == 1


def test_export_view_builds_slice_and_plot_export_dtos() -> None:
    view, _calls = _view()

    slice_handles = view.slice_handles(slice_display="slice-display")
    slice_style = view.slice_style()
    plot_callbacks = view.plot_callbacks()
    geometry = plot_callbacks.slice_geometry()

    assert slice_handles.slice_display == "slice-display"
    assert slice_handles.slice_plot == "slice-plot"
    assert slice_style.trajectory_pen == "trajectory-pen"
    assert plot_callbacks.set_axis is view.set_axis
    assert plot_callbacks.set_font is view.set_font
    assert geometry.width == 100.0
    assert geometry.height == 200.0
    assert geometry.rect == "rect"
