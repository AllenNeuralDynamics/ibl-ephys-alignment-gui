"""Tests for desktop plot export orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ephys_alignment_gui.desktop.displays.plot_exporter import (
    DesktopPlotExportCallbacks,
    DesktopPlotExporter,
    HistologyExportHandles,
    SliceExportGeometry,
    SliceExportHandles,
    SliceExportStyle,
)


class FakeAction:
    def __init__(self, label: str) -> None:
        self.label = label
        self.checked = False
        self.triggers = 0
        self.on_trigger = lambda: None

    def text(self) -> str:
        return self.label

    def setChecked(self, checked: bool) -> None:
        self.checked = checked

    def trigger(self) -> None:
        self.triggers += 1
        self.on_trigger()


class FakeActionGroup:
    def __init__(self, actions: list[FakeAction]) -> None:
        self._actions = actions
        self.index = 0 if actions else None
        for index, action in enumerate(actions):
            action.on_trigger = lambda idx=index: self._set_checked_index(idx)

    def checkedAction(self) -> FakeAction | None:
        if self.index is None:
            return None
        return self._actions[self.index]

    def actions(self) -> list[FakeAction]:
        return self._actions

    def _set_checked_index(self, index: int) -> None:
        self.index = index


class FakeEphysExporter:
    def __init__(self, calls: list[Any]) -> None:
        self.calls = calls

    def export(self, output_dir: Path, *, sess_info: str = "") -> None:
        self.calls.append(("ephys", output_dir, sess_info))


class FakeSliceMenuCoordinator:
    def __init__(self, action_group: FakeActionGroup) -> None:
        self.action_group = action_group

    def current_selection(self) -> str:
        return "slice-selection"


class FakeSlicePanelPresenter:
    def __init__(self) -> None:
        self.toggles = 0
        self.trajectory_calls: list[tuple[Any, Any]] = []
        self.plot_channels_calls: list[Any] = []

    def toggle_channel_visibility(self) -> None:
        self.toggles += 1

    def render_export_trajectory_overlay(
        self,
        pen: Any,
        *,
        selection: Any = None,
    ) -> None:
        self.trajectory_calls.append((pen, selection))

    def plot_channels(self, *, selection: Any = None) -> None:
        self.plot_channels_calls.append(selection)

    def current_channel_locations_ras(self, selection: Any = None) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 10.0],
                [3.0, 0.0, 20.0],
            ]
        )


class FakeSlicePlot:
    def __init__(self) -> None:
        self.x_ranges: list[dict[str, Any]] = []
        self.y_ranges: list[dict[str, Any]] = []
        self.resizes: list[tuple[float, float]] = []
        self.ranges: list[dict[str, Any]] = []

    def setXRange(self, **kwargs: Any) -> None:
        self.x_ranges.append(kwargs)

    def setYRange(self, **kwargs: Any) -> None:
        self.y_ranges.append(kwargs)

    def resize(self, width: float, height: float) -> None:
        self.resizes.append((width, height))

    def setRange(self, **kwargs: Any) -> None:
        self.ranges.append(kwargs)


class FakeHistologyDisplay:
    depth_ruler = "depth-ruler"
    aligned_plot = "aligned-histology"
    reference_plot = "reference-histology"

    def export_scene(self) -> str:
        return "histology-scene"


class FakeImageExporter:
    def __init__(self, item: Any, calls: list[Any]) -> None:
        self.item = item
        self.calls = calls

    def export(self, path: str) -> None:
        self.calls.append(("export", self.item, path))


def _exporter() -> tuple[
    DesktopPlotExporter,
    list[Any],
    list[FakeAction],
    FakeSlicePanelPresenter,
    FakeSlicePlot,
]:
    calls: list[Any] = []
    actions = [FakeAction("ccf"), FakeAction("registration")]
    action_group = FakeActionGroup(actions)
    slice_menu = FakeSliceMenuCoordinator(action_group)
    slice_panel = FakeSlicePanelPresenter()
    slice_plot = FakeSlicePlot()

    def image_exporter_factory(item: Any) -> FakeImageExporter:
        return FakeImageExporter(item, calls)

    exporter = DesktopPlotExporter(
        ephys_exporter=FakeEphysExporter(calls),
        slice_handles=SliceExportHandles(
            slice_display=object(),
            slice_panel_presenter=slice_panel,
            slice_menu_coordinator=slice_menu,
            slice_plot=slice_plot,
        ),
        slice_style=SliceExportStyle(trajectory_pen="trajectory-pen"),
        histology_handles=HistologyExportHandles(
            histology_display=FakeHistologyDisplay(),
        ),
        callbacks=DesktopPlotExportCallbacks(
            set_axis=lambda *args, **kwargs: calls.append(("set_axis", args, kwargs)),
            set_font=lambda *args, **kwargs: calls.append(("set_font", args, kwargs)),
            slice_geometry=lambda: SliceExportGeometry(
                width=120,
                height=80,
                rect="slice-rect",
            ),
            make_overview=lambda *args, **kwargs: calls.append(
                ("overview", args, kwargs)
            ),
        ),
        add_lines_points=lambda: calls.append(("add_lines_points",)),
        image_exporter_factory=image_exporter_factory,
    )
    return exporter, calls, actions, slice_panel, slice_plot


def test_desktop_plot_exporter_exports_all_plot_groups() -> None:
    exporter, calls, actions, slice_display, _slice_plot = _exporter()

    exporter.export(Path("/tmp/out"), sess_info="session_")

    assert calls[0] == ("ephys", Path("/tmp/out"), "session_")
    assert ("export", _slice_plot, "/tmp/out/session_slice_ccf.png") in calls
    assert (
        "export",
        _slice_plot,
        "/tmp/out/session_slice_registration.png",
    ) in calls
    assert ("export", _slice_plot, "/tmp/out/session_slice_zoom_ccf.png") in calls
    assert (
        "export",
        _slice_plot,
        "/tmp/out/session_slice_zoom_registration.png",
    ) in calls
    assert ("export", "histology-scene", "/tmp/out/session_hist.png") in calls
    assert calls[-2] == (
        "overview",
        (Path("/tmp/out"), "session_"),
        {"save_folder": Path("/tmp/out")},
    )
    assert calls[-1] == ("add_lines_points",)
    assert slice_display.toggles == 4
    assert slice_display.trajectory_calls == [
        ("trajectory-pen", "slice-selection")
    ] * 4
    assert slice_display.plot_channels_calls == ["slice-selection"] * 4
    assert [action.triggers for action in actions] == [2, 2]


def test_desktop_plot_exporter_restores_zoomed_slice_geometry() -> None:
    exporter, calls, _actions, _slice_display, slice_plot = _exporter()

    exporter.export(Path("/tmp/out"))

    assert slice_plot.x_ranges == [
        {"min": 1.0 - 200 / 1e6, "max": 3.0 + 200 / 1e6},
        {"min": 1.0 - 200 / 1e6, "max": 3.0 + 200 / 1e6},
    ]
    assert slice_plot.y_ranges == [
        {"min": 10.0 - 500 / 1e6, "max": 20.0 + 500 / 1e6},
        {"min": 10.0 - 500 / 1e6, "max": 20.0 + 500 / 1e6},
    ]
    assert slice_plot.resizes == [(50, 80), (120, 80), (50, 80), (120, 80)]
    assert slice_plot.ranges == [{"rect": "slice-rect"}, {"rect": "slice-rect"}]
    assert ("set_axis", ("depth-ruler", "left"), {}) in calls
    assert ("set_axis", ("depth-ruler", "left"), {"pen": "k"}) in calls
    assert (
        "set_axis",
        ("aligned-histology", "bottom"),
        {"label": "aligned"},
    ) in calls
    assert (
        "set_axis",
        ("reference-histology", "bottom"),
        {"label": "original"},
    ) in calls
