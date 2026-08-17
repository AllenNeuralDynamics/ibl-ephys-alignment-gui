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

    def set_track_display_transform(
        self,
        *,
        track_to_warped_position,
        warped_position_to_track,
    ) -> None:
        self.calls.append(
            (
                "set-track-transform",
                track_to_warped_position,
                warped_position_to_track,
            )
        )

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

    def replace_lines_from_raw_track(
        self,
        positions: Any,
        raw_track_positions: Any,
    ) -> None:
        self.calls.append(("replace-raw", positions, raw_track_positions))

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
            reference_plot="reference",
            image_plot="image",
            line_plot="line",
            probe_plot="probe",
            perpendicular_plot="perpendicular",
            fit_plot="fit",
        )
    )

    plots = display.layer._plots
    assert plots.histology == "histology"
    assert plots.reference == "reference"
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

    def track_to_warped(value: Any) -> Any:
        return value

    def warped_to_track(value: Any) -> Any:
        return value

    assert display.has_lines()
    display.set_lines_changed_callback(lambda: None)
    display.set_track_display_transform(
        track_to_warped_position=track_to_warped,
        warped_position_to_track=warped_to_track,
    )
    assert display.positions() == ([1.0], [2.0])
    display.clear()
    display.reattach()
    display.create_lines([3.0], [4.0])
    display.replace_lines([6.0], [7.0])
    display.replace_lines_from_raw_track([8.0], [9.0])
    display.sync_track_to_feature()
    assert display.select_line("line")
    display.clear_selection()
    assert display.delete_selected()

    assert layer.calls == [
        "has_lines",
        ("set-callback", layer.on_lines_changed),
        ("set-track-transform", track_to_warped, warped_to_track),
        "positions",
        "clear",
        "remove",
        "add",
        ("create", [3.0], [4.0]),
        ("replace", [6.0], [7.0]),
        ("replace-raw", [8.0], [9.0]),
        "sync",
        ("select", "line"),
        "clear_selection",
        "delete",
    ]
