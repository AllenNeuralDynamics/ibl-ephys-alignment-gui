"""Desktop view operations for active-shank screen refresh."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ephys_alignment_gui.alignment_read_models import ActiveShankPlotDataState
from ephys_alignment_gui.desktop_depth_plot_view import DesktopDepthPlotView
from ephys_alignment_gui.desktop_shank_presenter import DesktopShankSelectionState


@dataclass
class DesktopShankScreenView:
    """Own desktop-only shank screen state and view refresh helpers."""

    depth_plots: DesktopDepthPlotView
    init_menubar: Callable[[], None]
    set_view: Callable[..., None]
    configure: bool = True
    raw_image_payloads: dict[str, Any] = field(default_factory=dict)

    def reset_raw_image_payloads(self) -> None:
        """Clear cached raw-image payloads used by dynamic ephys plot menus."""
        self.raw_image_payloads = {}

    def raw_image_payload_mapping(self) -> Mapping[Any, Any]:
        """Return raw-image payloads available to plot-menu read models."""
        return self.raw_image_payloads

    def capture_plot_selection(
        self,
        preserve_plot_selection: bool,
        *,
        displays: Any,
    ) -> DesktopShankSelectionState:
        """Capture desktop plot selections to preserve across shank redraw."""
        prev_slice = displays.slice.capture_selection()
        prev_ephys_plot_keys = (
            displays.ephys.current_plot_keys()
            if preserve_plot_selection and displays.ephys.has_plot_menus()
            else None
        )
        return DesktopShankSelectionState(
            previous_slice_selection=prev_slice.selection,
            previous_slice_label=prev_slice.label,
            previous_ephys_plot_keys=prev_ephys_plot_keys,
        )

    def apply_plot_data_state(self, state: ActiveShankPlotDataState) -> None:
        """Apply prepared shank plot-data bounds to desktop depth plots."""
        self.depth_plots.set_probe_limits(
            min(0.0, float(state.channel_min_um)),
            float(state.channel_max_um),
        )
        self.reset_raw_image_payloads()

    def render_plot_menus(self, plot_menu_state: Any, *, displays: Any) -> None:
        """Refresh ephys plot menus for the selected shank."""
        if not displays.ephys.has_plot_menus():
            self.init_menubar()
        displays.ephys.render_menus(plot_menu_state)

    def configure_view_after_render(self, preserve_plot_selection: bool) -> None:
        """Apply one-time view configuration after shank rendering."""
        self.set_view(view=1, configure=self.configure and not preserve_plot_selection)
        if not preserve_plot_selection:
            self.configure = False
