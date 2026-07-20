"""Desktop feature-plot view state and coordinate mapping."""

from __future__ import annotations

from typing import Any


class FeaturePlotView:
    """Own the active feature plot item and its display coordinate metadata."""

    def __init__(self) -> None:
        self.data_plot: Any = None
        self.x_scale: float = 1.0
        self.y_scale: float = 1.0
        self.xrange: Any = None

    def set_data_plot(
        self,
        data_plot: Any,
        *,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        xrange: Any = None,
    ) -> None:
        """Record the active feature plot item and its transform metadata."""
        self.disconnect_clicked()
        self.data_plot = data_plot
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)
        self.xrange = xrange

    def clear(self) -> None:
        """Disconnect the active plot item and forget display metadata."""
        self.disconnect_clicked()
        self.data_plot = None
        self.x_scale = 1.0
        self.y_scale = 1.0
        self.xrange = None

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
        pos = self.data_plot.mapFromScene(scene_pos)
        return pos.y() * self.y_scale
