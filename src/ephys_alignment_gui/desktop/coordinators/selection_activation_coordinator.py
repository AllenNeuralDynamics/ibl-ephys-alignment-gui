"""Coordinate selection-driven loading for the desktop shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DesktopSelectionActivationCoordinator:
    """Route selection changes and explicit load actions through one load path."""

    session_selection_coordinator: Any
    probe_selection_coordinator: Any
    shank_selection_actions: Any
    load_preflight_coordinator: Any

    def session_selected(self, idx: int | None = None) -> bool:
        """Select a recording/session, then load or activate the selected stream."""
        if not self.session_selection_coordinator.session_selected(idx):
            return False
        return self.load_or_activate_selected_stream()

    def probe_selected(self, idx: int | None = None) -> bool:
        """Select a probe, then load or activate the selected stream."""
        if not self.probe_selection_coordinator.probe_selected(idx):
            return False
        return self.load_or_activate_selected_stream()

    def shank_selected(self, _idx: int | None = None) -> bool:
        """Select a shank, then load or activate the selected stream."""
        if not self.shank_selection_actions.shank_selected():
            return False
        return self.load_or_activate_selected_stream()

    def load_or_activate_selected_stream(self) -> bool:
        """Run desktop load preflight and enter the selected stream/shank."""
        return self.load_preflight_coordinator.load_data_button_pressed()
