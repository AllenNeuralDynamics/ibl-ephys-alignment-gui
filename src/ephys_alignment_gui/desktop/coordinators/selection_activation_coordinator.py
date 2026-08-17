"""Coordinate selection-driven loading for the desktop shell."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DesktopSelectionActivationCoordinator:
    """Route selection changes and explicit load actions through one load path."""

    session_selection_coordinator: Any
    probe_selection_coordinator: Any
    shank_selection_actions: Any
    load_preflight_coordinator: Any
    preserve_plot_selection: Callable[[], bool] = lambda: False

    def session_selected(self, idx: int | None = None) -> bool:
        """Select a recording/session and render probe choices without loading."""
        return self.session_selection_coordinator.session_selected(idx)

    def probe_selected(self, idx: int | None = None) -> bool:
        """Select a probe, then load or activate the selected stream."""
        preserve_plot_selection = self.preserve_plot_selection()
        if not self.probe_selection_coordinator.probe_selected(idx):
            return False
        return self.activate_selected_stream(
            preserve_plot_selection=preserve_plot_selection,
        )

    def shank_selected(self, _idx: int | None = None) -> bool:
        """Select a shank, then load or activate the selected stream."""
        preserve_plot_selection = self.preserve_plot_selection()
        if not self.shank_selection_actions.shank_selected():
            return False
        return self.activate_selected_stream(
            preserve_plot_selection=preserve_plot_selection,
        )

    def activate_selected_stream(
        self,
        *,
        preserve_plot_selection: bool | None = None,
    ) -> bool:
        """Run desktop load preflight and enter the selected stream/shank."""
        return self.load_preflight_coordinator.activate_selected_stream(
            preserve_plot_selection=preserve_plot_selection,
        )
