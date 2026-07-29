"""Tests for desktop reference-line display composition."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.desktop_reference_line_display import (
    DesktopReferenceLineDisplay,
    DesktopReferenceLineDisplayPorts,
)


class FakeLayer:
    def __init__(self) -> None:
        self.calls: list[Any] = []

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


def test_reference_line_display_constructs_layer_from_ports() -> None:
    changed: list[str] = []
    display = DesktopReferenceLineDisplay.create(
        ports=DesktopReferenceLineDisplayPorts(
            histology_plot="histology",
            image_plot="image",
            line_plot="line",
            probe_plot="probe",
            perpendicular_plot="perpendicular",
            fit_plot="fit",
            on_lines_changed=lambda: changed.append("changed"),
        )
    )

    plots = display.layer._plots
    assert plots.histology == "histology"
    assert plots.image == "image"
    assert plots.line == "line"
    assert plots.probe == "probe"
    assert plots.perpendicular == "perpendicular"
    assert plots.fit == "fit"
    display.layer._on_lines_changed()
    assert changed == ["changed"]


def test_reference_line_display_delegates_overlay_operations() -> None:
    layer = FakeLayer()
    display = DesktopReferenceLineDisplay(layer=layer)

    assert display.has_lines()
    assert display.positions() == ([1.0], [2.0])
    display.clear()
    display.reattach()
    display.create_lines([3.0], [4.0])
    display.create_previous_feature_lines([5.0])
    display.sync_track_to_feature()
    assert display.select_line("line")
    display.clear_selection()
    assert display.delete_selected()

    assert layer.calls == [
        "has_lines",
        "positions",
        "clear",
        "remove",
        "add",
        ("create", [3.0], [4.0]),
        ("create", [5.0], None),
        "sync",
        ("select", "line"),
        "clear_selection",
        "delete",
    ]
