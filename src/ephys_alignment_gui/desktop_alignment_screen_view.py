"""Desktop view operations for alignment edit rendering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ephys_alignment_gui.desktop_depth_plot_view import DesktopDepthPlotView


@dataclass(frozen=True)
class DesktopAlignmentScreenView:
    """Own desktop-only alignment render state and view refresh helpers."""

    depth_plots: DesktopDepthPlotView
    display_state: Any
    reference_lines: Any
    active_alignment_state: Callable[[], Any]
    lin_fit_checkbox: Any
    current_index_label: Any
    total_index_label: Any

    def restore_lin_fit_from_edit(self, lin_fit: bool | None) -> None:
        """Restore the linear-fit checkbox from an applied alignment edit."""
        if lin_fit is None:
            return
        self.display_state.edit_settings.set_lin_fit(lin_fit)
        self.lin_fit_checkbox.blockSignals(True)
        self.lin_fit_checkbox.setChecked(self.display_state.edit_settings.lin_fit)
        self.lin_fit_checkbox.blockSignals(False)

    def capture_depth_plot_y_ranges(self) -> dict[str, tuple[float, float]]:
        """Capture current y-ranges on the linked depth plots."""
        return self.depth_plots.capture_y_ranges()

    def restore_depth_plot_y_ranges(
        self,
        ranges: Mapping[str, tuple[float, float]],
    ) -> None:
        """Restore y-ranges captured before an alignment redraw."""
        self.depth_plots.restore_y_ranges(ranges)

    def create_reference_lines_for_previous_alignment(self) -> None:
        """Create editable reference lines from the previous alignment."""
        state = self.active_alignment_state()
        feature_prev = None if state is None else state.feature_prev
        if feature_prev is not None and np.any(feature_prev):
            self.reference_lines.create_previous_feature_lines(
                np.asarray(feature_prev)[1:-1] * 1e6
            )

    def set_default_feature_y_range(self) -> None:
        """Apply the default feature-depth range to the linked depth plots."""
        self.depth_plots.set_default_feature_y_range()

    def update_status(self) -> None:
        """Update edit-history status labels."""
        state = self.active_alignment_state()
        current_idx = 0 if state is None else state.edit_history.current_idx
        total_idx = 0 if state is None else state.edit_history.total_idx
        self.current_index_label.setText(f"Current Index = {current_idx}")
        self.total_index_label.setText(f"Total Index = {total_idx}")
