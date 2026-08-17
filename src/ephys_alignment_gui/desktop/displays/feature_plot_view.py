"""Desktop feature-plot view state and coordinate mapping."""

from __future__ import annotations

from typing import Any

import numpy as np


class FeaturePlotView:
    """Own the active feature plot item and its display coordinate metadata."""

    def __init__(self) -> None:
        self.data_plot: Any = None
        self.x_scale: float = 1.0
        self.y_scale: float = 1.0
        self.xrange: Any = None
        self.cluster_x_values: Any = None

    def set_data_plot(
        self,
        data_plot: Any,
        *,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        xrange: Any = None,
        cluster_x_values: Any = None,
    ) -> None:
        """Record the active feature plot item and its transform metadata."""
        self.disconnect_clicked()
        self.data_plot = data_plot
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)
        self.xrange = xrange
        self.cluster_x_values = cluster_x_values

    def clear(self) -> None:
        """Disconnect the active plot item and forget display metadata."""
        self.disconnect_clicked()
        self.data_plot = None
        self.x_scale = 1.0
        self.y_scale = 1.0
        self.xrange = None
        self.cluster_x_values = None

    def connect_clicked(self, callback: Any) -> None:
        """Connect the active plot's click signal when it exists."""
        if self.data_plot is None:
            return
        try:
            self.data_plot.sigClicked.connect(callback)
        except AttributeError:
            pass

    def disconnect_clicked(self) -> None:
        """Disconnect click callbacks from the active plot item."""
        if self.data_plot is None:
            return
        try:
            self.data_plot.sigClicked.disconnect()
        except (TypeError, AttributeError, RuntimeError):
            pass

    def feature_y_from_scene(self, scene_pos: Any) -> float | None:
        """Map a scene position to feature-space y in um."""
        if self.data_plot is None:
            return None
        scene_rect = getattr(self.data_plot, "sceneBoundingRect", None)
        if callable(scene_rect):
            try:
                rect = scene_rect()
                contains = getattr(rect, "contains", None)
                if callable(contains) and not contains(scene_pos):
                    return None
            except (AttributeError, RuntimeError, TypeError):
                pass
        pos = self.data_plot.mapFromScene(scene_pos)
        return pos.y() * self.y_scale

    def cluster_index_for_plot_x(self, x_value: float) -> int | None:
        """Return the cluster index represented by a plotted x coordinate."""
        if self.cluster_x_values is None:
            return None
        matches = np.argwhere(np.asarray(self.cluster_x_values) == x_value).ravel()
        if matches.size == 0:
            return None
        return int(matches[0])
