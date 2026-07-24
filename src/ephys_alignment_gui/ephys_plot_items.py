"""Desktop pyqtgraph item ownership for ephys data panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EphysPlotItems:
    """Own pyqtgraph items for image, line, and probe ephys panels."""

    image_plots: list[Any] = field(default_factory=list)
    line_plots: list[Any] = field(default_factory=list)
    probe_plots: list[Any] = field(default_factory=list)
    image_colorbars: list[Any] = field(default_factory=list)
    probe_colorbars: list[Any] = field(default_factory=list)
    probe_bounds: list[Any] = field(default_factory=list)

    def clear_image(self, image_fig: Any, colorbar_fig: Any) -> None:
        """Remove image/scatter plot items and image colorbars."""
        self._remove_all(image_fig, self.image_plots)
        self._remove_all(colorbar_fig, self.image_colorbars)

    def clear_line(self, line_fig: Any) -> None:
        """Remove line plot items."""
        self._remove_all(line_fig, self.line_plots)

    def clear_probe(self, probe_fig: Any, colorbar_fig: Any) -> None:
        """Remove probe image items, probe colorbars, and probe-bound lines."""
        self._remove_all(probe_fig, self.probe_plots)
        self._remove_all(colorbar_fig, self.probe_colorbars)
        self._remove_all(probe_fig, self.probe_bounds)

    def detach(self, figures: dict[str, Any]) -> None:
        """Remove all owned ephys panel items from their figures."""
        image_fig = figures.get("img")
        image_colorbar_fig = figures.get("img_cb")
        if image_fig is not None:
            self._remove_all(image_fig, self.image_plots)
        if image_colorbar_fig is not None:
            self._remove_all(image_colorbar_fig, self.image_colorbars)

        line_fig = figures.get("line")
        if line_fig is not None:
            self.clear_line(line_fig)

        probe_fig = figures.get("probe")
        probe_colorbar_fig = figures.get("probe_cb")
        if probe_fig is not None:
            self._remove_all(probe_fig, self.probe_plots)
            self._remove_all(probe_fig, self.probe_bounds)
        if probe_colorbar_fig is not None:
            self._remove_all(probe_colorbar_fig, self.probe_colorbars)

    @staticmethod
    def _remove_all(fig: Any, items: list[Any]) -> None:
        for item in items:
            fig.removeItem(item)
        items.clear()
