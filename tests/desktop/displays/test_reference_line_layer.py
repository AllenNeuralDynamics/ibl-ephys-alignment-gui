"""Tests for desktop reference-line overlay bookkeeping."""

from __future__ import annotations

import numpy as np

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

    def disconnect(self) -> None:
        self.disconnects += 1


class FakeLine:
    def __init__(self, y: float) -> None:
        self._y = y
        self.sigPositionChanged = FakeSignal()

    def pos(self) -> FakePosition:
        return FakePosition(self._y)

    def value(self) -> float:
        return self._y

    def setPos(self, y: float) -> None:
        self._y = y

    def getYPos(self) -> float:
        return self._y


class FakePoint:
    def __init__(self) -> None:
        self.data = None

    def setData(self, **kwargs) -> None:
        self.data = kwargs


class FakePlot:
    def __init__(self) -> None:
        self.added = []
        self.removed = []

    def addItem(self, item) -> None:
        self.added.append(item)

    def removeItem(self, item) -> None:
        self.removed.append(item)


def _populate(layer: ReferenceLineLayer) -> tuple[FakeLine, FakeLine]:
    feature = [FakeLine(10.0), FakeLine(10.0), FakeLine(10.0)]
    track = [FakeLine(20.0), FakeLine(20.0)]
    layer.lines_features = np.array([feature], dtype=object)
    layer.lines_tracks = np.array([track], dtype=object)
    layer.points = np.array([[FakePoint()]], dtype=object)
    return feature[0], track[0]


def _layer() -> tuple[ReferenceLineLayer, ReferenceLinePlots, list[str]]:
    plots = ReferenceLinePlots(
        histology=FakePlot(),
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


def test_positions_return_logical_feature_and_track_lines() -> None:
    layer, _, _ = _layer()
    _populate(layer)

    feature, track = layer.positions()

    np.testing.assert_array_equal(feature, np.array([10.0]))
    np.testing.assert_array_equal(track, np.array([20.0]))


def test_sync_track_to_feature_updates_track_handles_and_fit_point() -> None:
    layer, _, changes = _layer()
    _populate(layer)

    layer.sync_track_to_feature()

    assert layer.lines_tracks[0][0].getYPos() == 10.0
    assert layer.lines_tracks[0][1].getYPos() == 10.0
    assert layer.points[0][0].data == {"x": [10.0], "y": [10.0]}
    assert changes == ["changed"]


def test_delete_selected_removes_one_line_group_and_notifies() -> None:
    layer, plots, changes = _layer()
    _populate(layer)
    layer.select_line(layer.lines_features[0][1])

    assert layer.delete_selected()

    assert layer.lines_features.shape == (0, 3)
    assert layer.lines_tracks.shape == (0, 2)
    assert layer.points.shape == (0, 1)
    assert plots.image.removed
    assert plots.histology.removed
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
    assert layer.lines_tracks.shape == (0, 2)
    assert layer.points.shape == (0, 1)
    assert layer.selected_line == []
    assert feature_line.sigPositionChanged.disconnects == 1
    assert track_line.sigPositionChanged.disconnects == 1
    assert plots.image.removed
    assert plots.perpendicular.removed
