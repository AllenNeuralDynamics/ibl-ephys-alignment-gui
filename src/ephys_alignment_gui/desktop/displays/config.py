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
    style = window.style
    return DesktopEphysDisplayConfig(
        depth_view=window.app.queries.workspace.depth_view_settings(),
        line_pen=style.solid_pen,
        depth_guide_pen=style.dotted_pen,
        padding_provider=lambda: style.padding,
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
    style = window.style
    return DesktopHistologyDisplayConfig(
        depth_view=window.app.queries.workspace.depth_view_settings(),
        dotted_pen=style.dotted_pen,
        fit_pen=style.fit_pen,
        linear_fit_pen=style.linear_fit_pen,
        baseline_pen=style.dotted_pen,
        set_axis=set_axis,
        padding_provider=lambda: style.padding,
        on_linear_fit_changed=actions.lin_fit_option_changed,
        on_mouse_double_clicked=actions.on_mouse_double_clicked,
        on_mouse_hover=actions.on_mouse_hover,
        linear_fit_enabled=window.app.queries.workspace.linear_fit_enabled,
    )


def desktop_slice_display_config_from_main_window(
    window: Any,
) -> DesktopSliceDisplayConfig:
    """Adapt MainWindow callbacks/style to slice display config."""
    histology_available = window.app.queries.workspace.histology_data_loaded
    style = window.style
    return DesktopSliceDisplayConfig(
        depth_view=window.app.queries.workspace.depth_view_settings(),
        dotted_pen=style.dotted_pen,
        solid_pen=style.solid_pen,
        reference_line_pen=style.reference_line_pen,
        set_axis=set_axis,
        padding_provider=lambda: style.padding,
        histology_exists=histology_available,
    )


def desktop_display_config_from_main_window(window: Any) -> DesktopDisplayConfig:
    """Adapt MainWindow style/callback dependencies to display config."""
    return DesktopDisplayConfig(
        ephys=desktop_ephys_display_config_from_main_window(window),
        histology=desktop_histology_display_config_from_main_window(window),
        slice=desktop_slice_display_config_from_main_window(window),
    )
