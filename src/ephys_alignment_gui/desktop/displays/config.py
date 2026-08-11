"""Build desktop display-region config from the MainWindow shell."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.desktop.displays import DesktopDisplayConfig
from ephys_alignment_gui.desktop.displays.axis_style import set_axis
from ephys_alignment_gui.desktop.displays.ephys_display import DesktopEphysDisplayConfig
from ephys_alignment_gui.desktop.displays.histology_display import (
    DesktopHistologyDisplayConfig,
)
from ephys_alignment_gui.desktop.displays.slice_display import DesktopSliceDisplayConfig


def desktop_ephys_display_config_from_main_window(
    window: Any,
) -> DesktopEphysDisplayConfig:
    """Adapt MainWindow callbacks/style to ephys display config."""
    actions = window.shell_actions
    return DesktopEphysDisplayConfig(
        line_pen=window.kpen_solid,
        depth_guide_pen=window.kpen_dot,
        padding_provider=lambda: window.pad,
        raw_image_payloads=lambda: window.shank_screen_view.raw_image_payload_mapping(),
        set_axis=set_axis,
        reset_axis=actions.reset_axis_button_pressed,
        cluster_clicked=actions.cluster_clicked,
        on_mouse_double_clicked=actions.on_mouse_double_clicked,
        on_mouse_hover=actions.on_mouse_hover,
    )


def desktop_histology_display_config_from_main_window(
    window: Any,
) -> DesktopHistologyDisplayConfig:
    """Adapt MainWindow callbacks/style to histology display config."""
    actions = window.shell_actions
    histology_available = window.app.queries.workspace.histology_data_loaded
    return DesktopHistologyDisplayConfig(
        dotted_pen=window.kpen_dot,
        fit_pen=window.bpen_solid,
        linear_fit_pen=window.rpen_dot,
        baseline_pen=window.kpen_dot,
        set_axis=set_axis,
        padding_provider=lambda: window.pad,
        on_linear_fit_changed=actions.lin_fit_option_changed,
        on_mouse_double_clicked=actions.on_mouse_double_clicked,
        on_mouse_hover=actions.on_mouse_hover,
        histology_available=histology_available,
    )


def desktop_slice_display_config_from_main_window(
    window: Any,
) -> DesktopSliceDisplayConfig:
    """Adapt MainWindow callbacks/style to slice display config."""
    histology_available = window.app.queries.workspace.histology_data_loaded
    return DesktopSliceDisplayConfig(
        dotted_pen=window.kpen_dot,
        solid_pen=window.kpen_solid,
        reference_line_pen=window.reference_line_kpen,
        set_axis=set_axis,
        padding_provider=lambda: window.pad,
        histology_exists=histology_available,
    )


def desktop_display_config_from_main_window(window: Any) -> DesktopDisplayConfig:
    """Adapt MainWindow style/callback dependencies to display config."""
    return DesktopDisplayConfig(
        ephys=desktop_ephys_display_config_from_main_window(window),
        histology=desktop_histology_display_config_from_main_window(window),
        slice=desktop_slice_display_config_from_main_window(window),
    )
