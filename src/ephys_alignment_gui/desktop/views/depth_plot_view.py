"""Desktop view operations for linked feature-depth plots."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.core.view_limits import default_feature_y_limits


@dataclass(frozen=True)
class DesktopDepthPlotView:
    """Own desktop-only depth-limit and y-range operations."""

    default_range_plots: Sequence[Any]
    range_plots: Mapping[str, Any]
    probe_tip_lines: Sequence[Any]
    probe_top_lines: Sequence[Any]
    padding: Callable[[], float]
    range_anchor_name: str | None = None

    def set_probe_limits(self, min_um: float, max_um: float) -> None:
        """Apply probe tip/top guide lines to desktop plots."""
        for top_line in self.probe_top_lines:
            top_line.setY(max_um)
        for tip_line in self.probe_tip_lines:
            tip_line.setY(min_um)

    def default_feature_y_limits(
        self,
        *,
        depth_view: Any,
        in_brain_depths_um: Any,
    ) -> tuple[float, float]:
        """Return the current default feature-depth display limits."""
        return default_feature_y_limits(
            probe_tip_um=depth_view.probe_tip_um,
            probe_top_um=depth_view.probe_top_um,
            probe_extra_um=depth_view.probe_extra_um,
            in_brain_depths_um=in_brain_depths_um,
        )

    def set_default_feature_y_range(
        self,
        *,
        depth_view: Any,
        in_brain_depths_um: Any,
    ) -> None:
        """Apply the default feature-depth range to linked depth plots."""
        y_min, y_max = self.default_feature_y_limits(
            depth_view=depth_view,
            in_brain_depths_um=in_brain_depths_um,
        )
        for plot in self._range_update_plots():
            plot.setYRange(min=y_min, max=y_max, padding=self.padding())

    def capture_y_ranges(self) -> dict[str, tuple[float, float]]:
        """Capture current y-ranges on tracked depth plots."""
        ranges: dict[str, tuple[float, float]] = {}
        for name, plot in self.range_plots.items():
            try:
                y_min, y_max = plot.viewRange()[1]
            except (AttributeError, IndexError, TypeError):
                continue
            ranges[name] = (float(y_min), float(y_max))
        return ranges

    def restore_y_ranges(self, ranges: Mapping[str, tuple[float, float]]) -> None:
        """Restore y-ranges captured before an alignment redraw."""
        anchor_plot = self._anchor_plot()
        if anchor_plot is not None and self.range_anchor_name in ranges:
            y_min, y_max = ranges[self.range_anchor_name]
            if y_min != y_max:
                anchor_plot.setYRange(min=y_min, max=y_max, padding=0)
            return

        for name, (y_min, y_max) in ranges.items():
            plot = self.range_plots.get(name)
            if plot is None or y_min == y_max:
                continue
            plot.setYRange(min=y_min, max=y_max, padding=0)

    def _range_update_plots(self) -> tuple[Any, ...]:
        anchor_plot = self._anchor_plot()
        if anchor_plot is not None:
            return (anchor_plot,)
        return tuple(self.default_range_plots)

    def _anchor_plot(self) -> Any | None:
        if self.range_anchor_name is None:
            return None
        return self.range_plots.get(self.range_anchor_name)
