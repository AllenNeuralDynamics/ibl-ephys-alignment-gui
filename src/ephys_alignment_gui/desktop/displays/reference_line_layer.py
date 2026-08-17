"""Desktop pyqtgraph reference-line overlay lifecycle."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceLinePlots:
    """Plot handles used by the desktop reference-line overlay."""

    histology: Any
    reference: Any
    image: Any
    line: Any
    probe: Any
    perpendicular: Any
    fit: Any


class ReferenceLineLayer:
    """Own pyqtgraph handles for linked alignment reference lines."""

    def __init__(
        self,
        *,
        plots: ReferenceLinePlots,
        style_factory: Callable[[], tuple[Any, Any]],
        on_lines_changed: Callable[[], None],
        track_to_warped_position: Callable[[Any], Any] | None = None,
        warped_position_to_track: Callable[[Any], Any] | None = None,
    ) -> None:
        self._plots = plots
        self._style_factory = style_factory
        self._on_lines_changed = on_lines_changed
        self._track_to_warped_position = track_to_warped_position or self._identity
        self._warped_position_to_track = warped_position_to_track or self._identity
        self.lines_features: np.ndarray = np.empty((0, 3), dtype=object)
        self.lines_tracks: np.ndarray = np.empty((0, 3), dtype=object)
        self.points: np.ndarray = np.empty((0, 1), dtype=object)
        self.selected_line: Any = []
        self._updating_linked_lines = False
        self._line_styles: dict[int, tuple[Any, Any]] = {}
        self._highlighted_lines: list[Any] = []

    def has_lines(self) -> bool:
        """Return whether the session has reference-line handles."""
        return len(self.lines_features) > 0 and len(self.lines_tracks) > 0

    def set_on_lines_changed(self, callback: Callable[[], None]) -> None:
        """Set the callback invoked when managed line coordinates change."""
        self._on_lines_changed = callback

    def set_track_display_transform(
        self,
        *,
        track_to_warped_position: Callable[[Any], Any],
        warped_position_to_track: Callable[[Any], Any],
    ) -> None:
        """Set conversion callbacks between raw track and warped display depth."""
        self._track_to_warped_position = track_to_warped_position
        self._warped_position_to_track = warped_position_to_track

    def positions(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return feature/track reference-line positions in um."""
        if not self.has_lines():
            return None
        feature = np.array(
            [line[0].pos().y() for line in self.lines_features],
            dtype=float,
        )
        track = np.array(
            [line[2].pos().y() for line in self.lines_tracks],
            dtype=float,
        )
        return feature, track

    def remove_from_plots(self) -> None:
        """Remove current line handles from their plots without deleting them."""
        for line_feature, line_track, point in zip(
            self.lines_features,
            self.lines_tracks,
            self.points,
        ):
            self._remove_item(self._plots.image, line_feature[0])
            self._remove_item(self._plots.line, line_feature[1])
            self._remove_item(self._plots.probe, line_feature[2])
            self._remove_item(self._plots.histology, line_track[0])
            self._remove_item(self._plots.perpendicular, line_track[1])
            self._remove_item(self._plots.reference, line_track[2])
            self._remove_item(self._plots.fit, point[0])

    def add_to_plots(self) -> None:
        """Add current line handles back to their plots."""
        for line_feature, line_track, point in zip(
            self.lines_features,
            self.lines_tracks,
            self.points,
        ):
            self._plots.image.addItem(line_feature[0])
            self._plots.line.addItem(line_feature[1])
            self._plots.probe.addItem(line_feature[2])
            self._plots.histology.addItem(line_track[0])
            self._plots.perpendicular.addItem(line_track[1])
            self._plots.reference.addItem(line_track[2])
            self._plots.fit.addItem(point[0])

    def disconnect(self) -> None:
        """Disconnect pyqtgraph callbacks for current line handles."""
        for arr in (self.lines_features, self.lines_tracks):
            for group in arr:
                for item in group if hasattr(group, "__iter__") else [group]:
                    try:
                        item.sigPositionChanged.disconnect()
                    except (TypeError, AttributeError, RuntimeError):
                        pass

    def clear(self) -> None:
        """Remove, disconnect, and forget all reference-line handles."""
        self.remove_from_plots()
        self.disconnect()
        self._clear_highlight()
        self.lines_features = np.empty((0, 3), dtype=object)
        self.lines_tracks = np.empty((0, 3), dtype=object)
        self.points = np.empty((0, 1), dtype=object)
        self.selected_line = []
        self._updating_linked_lines = False
        self._line_styles.clear()

    def create_lines(
        self,
        positions: Any,
        track_positions: Any = None,
    ) -> None:
        """Create linked feature/track reference lines from coordinate arrays."""
        feature_positions = np.asarray(positions, dtype=float)
        if track_positions is None:
            track_positions = self._warped_positions_to_track(feature_positions)
        else:
            track_positions = np.asarray(track_positions, dtype=float)
        if feature_positions.shape != track_positions.shape:
            logger.error(
                "Cannot create reference lines: feature/track positions differ"
            )
            return

        warped_positions = self._track_to_warped_positions(track_positions)

        for feature_pos, track_pos, warped_pos in zip(
            feature_positions,
            track_positions,
            warped_positions,
        ):
            self._create_line(
                feature_pos=feature_pos,
                track_pos=track_pos,
                warped_pos=warped_pos,
            )

    def sync_track_to_feature(self) -> None:
        """Move track-space reference lines to current feature-line positions."""
        with self._linked_line_update():
            for line_feature, line_track, point in zip(
                self.lines_features,
                self.lines_tracks,
                self.points,
            ):
                warped_pos = line_feature[0].getYPos()
                track_pos = self._warped_positions_to_track([warped_pos])[0]
                self._set_line_pos(line_track[0], warped_pos)
                self._set_line_pos(line_track[1], warped_pos)
                self._set_line_pos(line_track[2], track_pos)
                point[0].setData(
                    x=[line_feature[0].pos().y()],
                    y=[track_pos],
                )
        self._on_lines_changed()

    def replace_lines(
        self,
        positions: Any,
        track_positions: Any = None,
        *,
        notify: bool = False,
    ) -> None:
        """Replace managed line positions with explicit feature/track coordinates."""
        feature_positions = np.asarray(positions, dtype=float)
        if track_positions is None:
            track_positions = self._warped_positions_to_track(feature_positions)
        else:
            track_positions = np.asarray(track_positions, dtype=float)
        if feature_positions.shape != track_positions.shape:
            logger.error(
                "Cannot replace reference lines: feature/track positions differ"
            )
            return

        if feature_positions.size == 0:
            self.clear()
            if notify:
                self._on_lines_changed()
            return

        if self.lines_features.shape[0] != feature_positions.size:
            self.clear()
            self.create_lines(feature_positions, track_positions)
            if notify:
                self._on_lines_changed()
            return

        warped_positions = self._track_to_warped_positions(track_positions)

        with self._linked_line_update():
            for (
                feature_pos,
                track_pos,
                warped_pos,
                line_feature,
                line_track,
                point,
            ) in zip(
                feature_positions,
                track_positions,
                warped_positions,
                self.lines_features,
                self.lines_tracks,
                self.points,
            ):
                for line in line_feature:
                    self._set_line_pos(line, feature_pos)
                self._set_line_pos(line_track[0], warped_pos)
                self._set_line_pos(line_track[1], warped_pos)
                self._set_line_pos(line_track[2], track_pos)
                point[0].setData(
                    x=[feature_pos],
                    y=[track_pos],
                )
        if notify:
            self._on_lines_changed()

    def update_feature_line(self, line: Any) -> None:
        """Mirror a moved feature-space line across feature plots."""
        if self._updating_linked_lines:
            return
        idx = np.where(self.lines_features == line)
        if idx[0].size == 0:
            return
        line_idx = idx[0][0]
        fig_idx = np.setdiff1d(
            np.arange(0, self.lines_features.shape[1]),
            idx[1][0],
        )

        with self._linked_line_update():
            for j in fig_idx:
                self._set_line_pos(self.lines_features[line_idx][j], line.value())

            self.points[line_idx][0].setData(
                x=[self.lines_features[line_idx][0].pos().y()],
                y=[self.lines_tracks[line_idx][2].pos().y()],
            )
        self._on_lines_changed()

    def update_track_line(self, line: Any) -> None:
        """Mirror a moved track-space line across track plots."""
        if self._updating_linked_lines:
            return
        idx = np.where(self.lines_tracks == line)
        if idx[0].size == 0:
            return
        line_idx = idx[0][0]
        plot_idx = idx[1][0]
        if plot_idx == 2:
            track_pos = line.value()
            warped_pos = self._track_to_warped_positions([track_pos])[0]
        else:
            warped_pos = line.value()
            track_pos = self._warped_positions_to_track([warped_pos])[0]

        with self._linked_line_update():
            self._set_line_pos(self.lines_tracks[line_idx][0], warped_pos)
            self._set_line_pos(self.lines_tracks[line_idx][1], warped_pos)
            self._set_line_pos(self.lines_tracks[line_idx][2], track_pos)

            self.points[line_idx][0].setData(
                x=[self.lines_features[line_idx][0].pos().y()],
                y=[track_pos],
            )
        self._on_lines_changed()

    def select_line(self, line: Any) -> bool:
        """Select a managed reference-line handle."""
        line_group = self._line_group(line)
        if line_group is None:
            self.clear_selection()
            return False
        self.clear_selection()
        self.selected_line = line
        line_idx, group_name = line_group
        self._highlight_line_group(line_idx, group_name)
        return True

    def clear_selection(self) -> None:
        """Clear selected reference-line handle."""
        self._clear_highlight()
        self.selected_line = []

    def delete_selected(self) -> bool:
        """Delete the selected reference-line group."""
        if not self.selected_line:
            return False

        line_idx = self._line_index(self.selected_line)
        if line_idx is None:
            self.selected_line = []
            return False

        self._clear_highlight()
        line_feature = self.lines_features[line_idx]
        line_track = self.lines_tracks[line_idx]
        point = self.points[line_idx]
        self._remove_item(self._plots.image, self.lines_features[line_idx, 0])
        self._remove_item(self._plots.line, self.lines_features[line_idx, 1])
        self._remove_item(self._plots.probe, self.lines_features[line_idx, 2])
        self._remove_item(self._plots.histology, self.lines_tracks[line_idx, 0])
        self._remove_item(self._plots.perpendicular, self.lines_tracks[line_idx, 1])
        self._remove_item(self._plots.reference, self.lines_tracks[line_idx, 2])
        self._remove_item(self._plots.fit, self.points[line_idx, 0])
        for item in (*line_feature, *line_track, point[0]):
            self._forget_line_style(item)
        self.lines_features = np.delete(self.lines_features, line_idx, axis=0)
        self.lines_tracks = np.delete(self.lines_tracks, line_idx, axis=0)
        self.points = np.delete(self.points, line_idx, axis=0)
        self.selected_line = []
        self._on_lines_changed()
        return True

    def _line_index(self, line: Any) -> int | None:
        line_group = self._line_group(line)
        if line_group is None:
            return None
        line_idx, _group_name = line_group
        return line_idx

    def _line_group(self, line: Any) -> tuple[int, str] | None:
        feature_idx = np.where(self.lines_features == line)[0]
        if feature_idx.size != 0:
            return int(feature_idx[0]), "feature"
        track_idx = np.where(self.lines_tracks[:, :2] == line)[0]
        if track_idx.size != 0:
            return int(track_idx[0]), "track"
        return None

    def _highlight_line_group(self, line_idx: int, group_name: str) -> None:
        if group_name == "feature":
            lines = list(self.lines_features[line_idx])
        else:
            lines = list(self.lines_tracks[line_idx, :2])
        for line in lines:
            self._set_line_highlighted(line, highlighted=True)
        self._highlighted_lines = lines

    def _clear_highlight(self) -> None:
        for line in self._highlighted_lines:
            self._set_line_highlighted(line, highlighted=False)
        self._highlighted_lines = []

    def _set_line_highlighted(self, line: Any, *, highlighted: bool) -> None:
        style = self._line_styles.get(id(line))
        if style is None:
            return
        pen, hover_pen = style
        active_pen = hover_pen if highlighted else pen
        set_pen = getattr(line, "setPen", None)
        if callable(set_pen):
            set_pen(active_pen)
        set_hover_pen = getattr(line, "setHoverPen", None)
        if callable(set_hover_pen):
            set_hover_pen(active_pen)

    def _remember_line_style(self, line: Any, pen: Any, hover_pen: Any) -> None:
        self._line_styles[id(line)] = (pen, hover_pen)

    def _forget_line_style(self, line: Any) -> None:
        self._line_styles.pop(id(line), None)

    def _linked_line_update(self) -> _LinkedLineUpdate:
        return _LinkedLineUpdate(self)

    @staticmethod
    def _set_line_pos(line: Any, position: float) -> None:
        block_signals = getattr(line, "blockSignals", None)
        previous_blocked = None
        if callable(block_signals):
            previous_blocked = block_signals(True)
        try:
            line.setPos(position)
        finally:
            if callable(block_signals) and previous_blocked is not None:
                block_signals(previous_blocked)

    @staticmethod
    def _remove_item(plot: Any, item: Any) -> None:
        try:
            plot.removeItem(item)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    @staticmethod
    def _identity(values: Any) -> Any:
        return values

    def _track_to_warped_positions(self, positions: Any) -> np.ndarray:
        return np.asarray(self._track_to_warped_position(positions), dtype=float)

    def _warped_positions_to_track(self, positions: Any) -> np.ndarray:
        return np.asarray(self._warped_position_to_track(positions), dtype=float)

    @staticmethod
    def _make_hover_pen(pen: Any) -> Any:
        hover_pen = pg.mkPen(pen)
        width = getattr(hover_pen, "width", None)
        set_width = getattr(hover_pen, "setWidth", None)
        if callable(width) and callable(set_width):
            set_width(max(width() + 2, 4))
        return hover_pen

    def _create_line(
        self,
        *,
        feature_pos: float,
        track_pos: float,
        warped_pos: float,
    ) -> None:
        pen, brush = self._style_factory()
        hover_pen = self._make_hover_pen(pen)
        line_track = pg.InfiniteLine(
            pos=warped_pos,
            angle=0,
            pen=pen,
            hoverPen=hover_pen,
            movable=True,
        )
        line_track.sigPositionChanged.connect(self.update_track_line)
        line_track.setZValue(100)
        line_feature1 = pg.InfiniteLine(
            pos=feature_pos,
            angle=0,
            pen=pen,
            hoverPen=hover_pen,
            movable=True,
        )
        line_feature1.setZValue(100)
        line_feature1.sigPositionChanged.connect(self.update_feature_line)
        line_feature2 = pg.InfiniteLine(
            pos=feature_pos,
            angle=0,
            pen=pen,
            hoverPen=hover_pen,
            movable=True,
        )
        line_feature2.setZValue(100)
        line_feature2.sigPositionChanged.connect(self.update_feature_line)
        line_feature3 = pg.InfiniteLine(
            pos=feature_pos,
            angle=0,
            pen=pen,
            hoverPen=hover_pen,
            movable=True,
        )
        line_feature3.setZValue(100)
        line_feature3.sigPositionChanged.connect(self.update_feature_line)
        line_track_perp = pg.InfiniteLine(
            pos=warped_pos,
            angle=0,
            pen=pen,
            hoverPen=hover_pen,
            movable=True,
        )
        line_track_perp.setZValue(100)
        line_track_perp.sigPositionChanged.connect(self.update_track_line)
        line_track_reference = pg.InfiniteLine(
            pos=track_pos,
            angle=0,
            pen=pen,
            hoverPen=hover_pen,
            movable=False,
        )
        line_track_reference.setZValue(100)
        self._plots.histology.addItem(line_track)
        self._plots.image.addItem(line_feature1)
        self._plots.line.addItem(line_feature2)
        self._plots.probe.addItem(line_feature3)
        self._plots.perpendicular.addItem(line_track_perp)
        self._plots.reference.addItem(line_track_reference)
        for line in (
            line_track,
            line_feature1,
            line_feature2,
            line_feature3,
            line_track_perp,
            line_track_reference,
        ):
            self._remember_line_style(line, pen, hover_pen)

        self.lines_features = np.vstack(
            [
                self.lines_features,
                [line_feature1, line_feature2, line_feature3],
            ]
        )
        self.lines_tracks = np.vstack(
            [self.lines_tracks, [line_track, line_track_perp, line_track_reference]]
        )

        point = pg.PlotDataItem()
        point.setData(
            x=[line_feature1.pos().y()],
            y=[line_track_reference.pos().y()],
            symbolBrush=brush,
            symbol="o",
            symbolSize=10,
        )
        self._plots.fit.addItem(point)
        self.points = np.vstack([self.points, point])


class _LinkedLineUpdate:
    def __init__(self, layer: ReferenceLineLayer) -> None:
        self._layer = layer
        self._previous = False

    def __enter__(self) -> None:
        self._previous = self._layer._updating_linked_lines
        self._layer._updating_linked_lines = True

    def __exit__(self, *_args: object) -> None:
        self._layer._updating_linked_lines = self._previous
