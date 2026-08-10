"""Desktop view operations for alignment edit rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop_depth_plot_view import DesktopDepthPlotView


@dataclass(frozen=True)
class DesktopAlignmentScreenView:
    """Own desktop-only alignment render state and view refresh helpers."""

    depth_plots: DesktopDepthPlotView
    reference_lines: Any
    lin_fit_checkbox: Any
    current_index_label: Any
    total_index_label: Any

    def set_linear_fit_checked(self, enabled: bool) -> None:
        """Render the linear-fit checkbox without recursively emitting changes."""
        self.lin_fit_checkbox.blockSignals(True)
        self.lin_fit_checkbox.setChecked(enabled)
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

    def create_reference_lines_for_previous_alignment(self, state: Any) -> None:
        """Create editable reference lines from the previous alignment."""
        if state.previous_feature_positions_um is not None:
            self.reference_lines.create_previous_feature_lines(
                state.previous_feature_positions_um
            )

    def set_default_feature_y_range(
        self,
        *,
        depth_view: Any,
        in_brain_depths_um: Any,
    ) -> None:
        """Apply the default feature-depth range to the linked depth plots."""
        self.depth_plots.set_default_feature_y_range(
            depth_view=depth_view,
            in_brain_depths_um=in_brain_depths_um,
        )

    def update_status(self, state: Any) -> None:
        """Update edit-history status labels."""
        self.current_index_label.setText(f"Current Index = {state.current_idx}")
        self.total_index_label.setText(f"Total Index = {state.total_idx}")
