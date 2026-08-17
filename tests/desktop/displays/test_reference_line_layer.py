"""Tests for desktop reference-line overlay bookkeeping."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ephys_alignment_gui.desktop.displays.reference_line_layer import (
    ReferenceLineLayer,
    ReferenceLinePlots,
)


class FakePosition:
    def __init__(self, y: float) -> None:
        self._y = y

    def y(self) -> float:
        return self._y


class FakeSignal:
    def __init__(self) -> None:
        self.disconnects = 0
        self.connects = []

    def connect(self, callback) -> None:
        self.connects.append(callback)

    def disconnect(self) -> None:
        self.disconnects += 1


class FakeLine:
    def __init__(self, y: float | None = None, **kwargs) -> None:
        self._y = kwargs.get("pos", y)
        self.sigPositionChanged = FakeSignal()
        self.blocked_states: list[bool] = []
        self._blocked = False
        self.on_set = None
        self.pen = "normal"
        self.hover_pen = "normal"
        self.z_value = None

    def pos(self) -> FakePosition:
        return FakePosition(self._y)

    def value(self) -> float:
        return self._y

    def setPos(self, y: float) -> None:
        self._y = y
        if self.on_set is not None and not self._blocked:
            self.on_set(self)

    def getYPos(self) -> float:
        return self._y

    def blockSignals(self, blocked: bool) -> bool:
        previous = self._blocked
        self._blocked = blocked
        self.blocked_states.append(blocked)
        return previous

    def setPen(self, pen) -> None:
        self.pen = pen

    def setHoverPen(self, pen) -> None:
        self.hover_pen = pen

    def setZValue(self, value) -> None:
        self.z_value = value


class FakePoint:
    def __init__(self) -> None:
        self.data = None

    def setData(self, **kwargs) -> None:
        self.data = kwargs


class FakePlot:
    def __init__(self, *, fail_remove: bool = False) -> None:
        self.added = []
        self.removed = []
        self.fail_remove = fail_remove

    def addItem(self, item) -> None:
        self.added.append(item)

    def removeItem(self, item) -> None:
        if self.fail_remove:
            raise ValueError("item is not attached")
        self.removed.append(item)


def _populate(layer: ReferenceLineLayer) -> tuple[FakeLine, FakeLine]:
    feature = [
        FakeLine(10.0),
        FakeLine(10.0),
        FakeLine(10.0),
    ]
    track = [FakeLine(20.0), FakeLine(20.0), FakeLine(20.0)]
    layer.lines_features = np.array([feature], dtype=object)
    layer.lines_tracks = np.array([track], dtype=object)
    layer.points = np.array([[FakePoint()]], dtype=object)
    for line in (*feature, *track):
        layer._remember_line_style(line, "normal", "highlight")
    return feature[0], track[0]


def _layer() -> tuple[ReferenceLineLayer, ReferenceLinePlots, list[str]]:
    plots = ReferenceLinePlots(
        histology=FakePlot(),
        reference=FakePlot(),
        image=FakePlot(),
        line=FakePlot(),
        probe=FakePlot(),
        perpendicular=FakePlot(),
        fit=FakePlot(),
    )
    changes: list[str] = []
    layer = ReferenceLineLayer(
        plots=plots,
        style_factory=lambda: (None, None),
        on_lines_changed=lambda: changes.append("changed"),
    )
    return layer, plots, changes


def test_positions_return_feature_and_warped_display_lines() -> None:
    layer, _, _ = _layer()
    _populate(layer)

    feature, track = layer.positions()

    np.testing.assert_array_equal(feature, np.array([10.0]))
    np.testing.assert_array_equal(track, np.array([20.0]))


def test_create_lines_defaults_warped_display_to_feature_position(
    monkeypatch,
) -> None:
    layer, _, _ = _layer()
    monkeypatch.setattr(pg, "InfiniteLine", FakeLine)
    monkeypatch.setattr(pg, "PlotDataItem", FakePoint)
    layer.set_track_display_transform(
        track_to_warped_position=lambda values: np.asarray(values) - 5.0,
        warped_position_to_track=lambda values: np.asarray(values) + 5.0,
    )

    layer.create_lines([30.0])

    assert [line.getYPos() for line in layer.lines_features[0]] == [
        30.0,
        30.0,
        30.0,
    ]
    assert layer.lines_tracks[0][0].getYPos() == 30.0
    assert layer.lines_tracks[0][1].getYPos() == 30.0
    assert layer.lines_tracks[0][2].getYPos() == 35.0
    assert layer.points[0][0].data["x"] == [30.0]
    assert layer.points[0][0].data["y"] == [30.0]
    np.testing.assert_array_equal(layer.positions()[0], [30.0])
    np.testing.assert_array_equal(layer.positions()[1], [30.0])


def test_sync_track_to_feature_updates_track_handles_and_fit_point() -> None:
    layer, _, changes = _layer()
    _populate(layer)

    layer.sync_track_to_feature()

    assert layer.lines_tracks[0][0].getYPos() == 10.0
    assert layer.lines_tracks[0][1].getYPos() == 10.0
    assert layer.lines_tracks[0][2].getYPos() == 10.0
    assert layer.points[0][0].data == {"x": [10.0], "y": [10.0]}
    assert changes == ["changed"]


def test_replace_lines_updates_handles_without_notifying() -> None:
    layer, _, changes = _layer()
    feature_line, track_line = _populate(layer)

    layer.replace_lines([30.0], [40.0])

    assert feature_line.getYPos() == 30.0
    assert track_line.getYPos() == 40.0
    assert layer.lines_features[0][1].getYPos() == 30.0
    assert layer.lines_tracks[0][1].getYPos() == 40.0
    assert layer.lines_tracks[0][2].getYPos() == 40.0
    assert layer.points[0][0].data == {"x": [30.0], "y": [40.0]}
    assert feature_line.blocked_states == [True, False]
    assert track_line.blocked_states == [True, False]
    assert changes == []


def test_track_lines_use_and_return_warped_display_positions() -> None:
    layer, _, changes = _layer()
    layer.set_track_display_transform(
        track_to_warped_position=lambda values: np.asarray(values) - 5.0,
        warped_position_to_track=lambda values: np.asarray(values) + 5.0,
    )
    _populate(layer)

    layer.replace_lines([30.0], [40.0])

    assert layer.lines_tracks[0][0].getYPos() == 40.0
    assert layer.lines_tracks[0][1].getYPos() == 40.0
    assert layer.lines_tracks[0][2].getYPos() == 45.0
    np.testing.assert_array_equal(layer.positions()[0], [30.0])
    np.testing.assert_array_equal(layer.positions()[1], [40.0])

    layer.lines_tracks[0][1].setPos(50.0)
    layer.update_track_line(layer.lines_tracks[0][1])

    assert layer.lines_tracks[0][0].getYPos() == 50.0
    assert layer.lines_tracks[0][1].getYPos() == 50.0
    assert layer.lines_tracks[0][2].getYPos() == 55.0
    np.testing.assert_array_equal(layer.positions()[1], [50.0])
    assert changes == ["changed"]


def test_replace_lines_from_raw_track_projects_to_warped_display() -> None:
    layer, _, changes = _layer()
    _populate(layer)
    layer.set_track_display_transform(
        track_to_warped_position=lambda values: np.asarray(values) - 5.0,
        warped_position_to_track=lambda values: np.asarray(values) + 5.0,
    )

    layer.replace_lines_from_raw_track([30.0], [35.0])

    assert layer.lines_features[0][0].getYPos() == 30.0
    assert layer.lines_tracks[0][0].getYPos() == 30.0
    assert layer.lines_tracks[0][1].getYPos() == 30.0
    assert layer.lines_tracks[0][2].getYPos() == 35.0
    assert layer.points[0][0].data == {"x": [30.0], "y": [30.0]}
    np.testing.assert_array_equal(layer.positions()[0], [30.0])
    np.testing.assert_array_equal(layer.positions()[1], [30.0])
    assert changes == []


def test_feature_line_update_suppresses_recursive_sibling_signals() -> None:
    layer, _, changes = _layer()
    feature_line, _track_line = _populate(layer)
    sibling_line = layer.lines_features[0][1]
    sibling_line.on_set = layer.update_feature_line

    feature_line.setPos(15.0)
    layer.update_feature_line(feature_line)

    assert sibling_line.getYPos() == 15.0
    assert sibling_line.blocked_states == [True, False]
    assert changes == ["changed"]


def test_track_line_update_suppresses_recursive_sibling_signals() -> None:
    layer, _, changes = _layer()
    _feature_line, track_line = _populate(layer)
    sibling_line = layer.lines_tracks[0][1]
    sibling_line.on_set = layer.update_track_line

    track_line.setPos(25.0)
    layer.update_track_line(track_line)

    assert sibling_line.getYPos() == 25.0
    assert sibling_line.blocked_states == [True, False]
    assert changes == ["changed"]


def test_original_reference_track_line_is_not_selectable() -> None:
    layer, _, _ = _layer()
    _populate(layer)

    assert not layer.select_line(layer.lines_tracks[0][2])
    assert layer.selected_line == []


def test_select_feature_line_highlights_feature_group_only() -> None:
    layer, _, _ = _layer()
    _populate(layer)

    assert layer.select_line(layer.lines_features[0][1])

    assert [line.pen for line in layer.lines_features[0]] == [
        "highlight",
        "highlight",
        "highlight",
    ]
    assert [line.pen for line in layer.lines_tracks[0]] == [
        "normal",
        "normal",
        "normal",
    ]

    layer.clear_selection()

    assert [line.pen for line in layer.lines_features[0]] == [
        "normal",
        "normal",
        "normal",
    ]


def test_select_warped_track_line_highlights_warped_track_group_only() -> None:
    layer, _, _ = _layer()
    _populate(layer)

    assert layer.select_line(layer.lines_tracks[0][1])

    assert [line.pen for line in layer.lines_features[0]] == [
        "normal",
        "normal",
        "normal",
    ]
    assert [line.pen for line in layer.lines_tracks[0]] == [
        "highlight",
        "highlight",
        "normal",
    ]

    layer.clear_selection()

    assert [line.pen for line in layer.lines_tracks[0]] == [
        "normal",
        "normal",
        "normal",
    ]


def test_replace_lines_can_notify_when_requested() -> None:
    layer, _, changes = _layer()
    _populate(layer)

    layer.replace_lines([30.0], [40.0], notify=True)

    assert changes == ["changed"]


def test_replace_lines_with_no_positions_clears_without_notifying() -> None:
    layer, plots, changes = _layer()
    _populate(layer)

    layer.replace_lines([], [])

    assert layer.lines_features.shape == (0, 3)
    assert layer.lines_tracks.shape == (0, 3)
    assert plots.image.removed
    assert changes == []


def test_delete_selected_removes_one_line_group_and_notifies() -> None:
    layer, plots, changes = _layer()
    _populate(layer)
    layer.select_line(layer.lines_features[0][1])

    assert layer.delete_selected()

    assert layer.lines_features.shape == (0, 3)
    assert layer.lines_tracks.shape == (0, 3)
    assert layer.points.shape == (0, 1)
    assert plots.image.removed
    assert plots.histology.removed
    assert plots.reference.removed
    assert plots.perpendicular.removed
    assert plots.fit.removed
    assert changes == ["changed"]


def test_stale_line_signal_is_ignored() -> None:
    layer, _, changes = _layer()

    layer.update_feature_line(FakeLine(99.0))
    layer.update_track_line(FakeLine(99.0))

    assert changes == []


def test_clear_removes_disconnects_and_resets_handles() -> None:
    layer, plots, _ = _layer()
    feature_line, track_line = _populate(layer)
    layer.select_line(feature_line)

    layer.clear()

    assert layer.lines_features.shape == (0, 3)
    assert layer.lines_tracks.shape == (0, 3)
    assert layer.points.shape == (0, 1)
    assert layer.selected_line == []
    assert feature_line.sigPositionChanged.disconnects == 1
    assert track_line.sigPositionChanged.disconnects == 1
    assert plots.image.removed
    assert plots.perpendicular.removed


def test_remove_from_plots_tolerates_already_detached_items() -> None:
    plots = ReferenceLinePlots(
        histology=FakePlot(fail_remove=True),
        reference=FakePlot(),
        image=FakePlot(),
        line=FakePlot(),
        probe=FakePlot(),
        perpendicular=FakePlot(),
        fit=FakePlot(),
    )
    layer = ReferenceLineLayer(
        plots=plots,
        style_factory=lambda: (None, None),
        on_lines_changed=lambda: None,
    )
    _populate(layer)

    layer.remove_from_plots()

    assert plots.reference.removed
    assert plots.perpendicular.removed
