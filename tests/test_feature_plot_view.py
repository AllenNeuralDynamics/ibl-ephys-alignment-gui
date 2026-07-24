"""Tests for desktop feature plot handle ownership."""

from __future__ import annotations

from ephys_alignment_gui.feature_plot_view import FeaturePlotView


class FakeSignal:
    def __init__(self) -> None:
        self.connected = []
        self.disconnects = 0

    def connect(self, callback) -> None:
        self.connected.append(callback)

    def disconnect(self) -> None:
        self.disconnects += 1
        self.connected = []


class FakePosition:
    def __init__(self, y: float) -> None:
        self._y = y

    def y(self) -> float:
        return self._y


class FakePlot:
    def __init__(self, y: float = 4.0) -> None:
        self.sigClicked = FakeSignal()
        self._y = y
        self.scene_positions = []

    def mapFromScene(self, scene_pos) -> FakePosition:
        self.scene_positions.append(scene_pos)
        return FakePosition(self._y)


def test_set_data_plot_disconnects_previous_plot_and_stores_metadata() -> None:
    previous = FakePlot()
    current = FakePlot()
    view = FeaturePlotView()
    view.set_data_plot(previous, x_scale=2, y_scale=3, xrange=(1, 2))

    view.set_data_plot(
        current,
        x_scale=5,
        y_scale=7,
        xrange=(10, 20),
        cluster_x_values=[11, 12],
    )

    assert previous.sigClicked.disconnects == 1
    assert view.data_plot is current
    assert view.x_scale == 5.0
    assert view.y_scale == 7.0
    assert view.xrange == (10, 20)
    assert view.cluster_x_values == [11, 12]


def test_connect_clicked_uses_active_plot_signal() -> None:
    callback = object()
    plot = FakePlot()
    view = FeaturePlotView()
    view.set_data_plot(plot)

    view.connect_clicked(callback)

    assert plot.sigClicked.connected == [callback]


def test_feature_y_from_scene_maps_to_feature_units() -> None:
    plot = FakePlot(y=11.0)
    view = FeaturePlotView()
    view.set_data_plot(plot, y_scale=20)

    assert view.feature_y_from_scene("scene-pos") == 220.0
    assert plot.scene_positions == ["scene-pos"]


def test_clear_disconnects_and_resets_state() -> None:
    plot = FakePlot()
    view = FeaturePlotView()
    view.set_data_plot(
        plot,
        x_scale=2,
        y_scale=3,
        xrange=(1, 2),
        cluster_x_values=[1],
    )

    view.clear()

    assert plot.sigClicked.disconnects == 1
    assert view.data_plot is None
    assert view.x_scale == 1.0
    assert view.y_scale == 1.0
    assert view.xrange is None
    assert view.cluster_x_values is None
    assert view.feature_y_from_scene("scene-pos") is None


def test_cluster_index_for_plot_x_uses_active_cluster_values() -> None:
    view = FeaturePlotView()

    assert view.cluster_index_for_plot_x(12) is None

    view.set_data_plot(FakePlot(), cluster_x_values=[10, 12, 14])

    assert view.cluster_index_for_plot_x(12) == 1
    assert view.cluster_index_for_plot_x(13) is None
