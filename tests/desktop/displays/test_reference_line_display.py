"""Tests for desktop reference-line display composition."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.desktop.displays.reference_line_display import (
    DesktopReferenceLineDisplay,
    ReferenceLinePlotBindings,
)


class FakeLayer:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.on_lines_changed = lambda: None

    def set_on_lines_changed(self, callback) -> None:
        self.calls.append(("set-callback", callback))
        self.on_lines_changed = callback

    def has_lines(self) -> bool:
        self.calls.append("has_lines")
        return True

    def positions(self) -> tuple[list[float], list[float]]:
        self.calls.append("positions")
        return [1.0], [2.0]

    def clear(self) -> None:
        self.calls.append("clear")

    def remove_from_plots(self) -> None:
        self.calls.append("remove")

    def add_to_plots(self) -> None:
        self.calls.append("add")

    def create_lines(self, positions: Any, track_positions: Any = None) -> None:
        self.calls.append(("create", positions, track_positions))

    def replace_lines(self, positions: Any, track_positions: Any = None) -> None:
        self.calls.append(("replace", positions, track_positions))

    def sync_track_to_feature(self) -> None:
        self.calls.append("sync")

    def select_line(self, line: Any) -> bool:
        self.calls.append(("select", line))
        return True

    def clear_selection(self) -> None:
        self.calls.append("clear_selection")

    def delete_selected(self) -> bool:
        self.calls.append("delete")
        return True


def test_reference_line_display_constructs_layer_from_bindings() -> None:
    changed: list[str] = []
    display = DesktopReferenceLineDisplay.create(
        bindings=ReferenceLinePlotBindings(
            histology_plot="histology",
            image_plot="image",
            line_plot="line",
            probe_plot="probe",
            perpendicular_plot="perpendicular",
            fit_plot="fit",
        )
    )

    plots = display.layer._plots
    assert plots.histology == "histology"
    assert plots.image == "image"
    assert plots.line == "line"
    assert plots.probe == "probe"
    assert plots.perpendicular == "perpendicular"
    assert plots.fit == "fit"
    display.set_lines_changed_callback(lambda: changed.append("changed"))
    display.layer._on_lines_changed()
    assert changed == ["changed"]


def test_reference_line_display_delegates_overlay_operations() -> None:
    layer = FakeLayer()
    display = DesktopReferenceLineDisplay(layer=layer)

    assert display.has_lines()
    display.set_lines_changed_callback(lambda: None)
    assert display.positions() == ([1.0], [2.0])
    display.clear()
    display.reattach()
    display.create_lines([3.0], [4.0])
    display.replace_lines([6.0], [7.0])
    display.sync_track_to_feature()
    assert display.select_line("line")
    display.clear_selection()
    assert display.delete_selected()

    assert layer.calls == [
        "has_lines",
        ("set-callback", layer.on_lines_changed),
        "positions",
        "clear",
        "remove",
        "add",
        ("create", [3.0], [4.0]),
        ("replace", [6.0], [7.0]),
        "sync",
        ("select", "line"),
        "clear_selection",
        "delete",
    ]
