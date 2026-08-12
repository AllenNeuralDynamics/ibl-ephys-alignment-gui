"""Build desktop display-region config from explicit shell handles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.displays import DesktopDisplayConfig
from ephys_alignment_gui.desktop.displays.axis_style import set_axis
from ephys_alignment_gui.desktop.displays.ephys_display import DesktopEphysDisplayConfig
from ephys_alignment_gui.desktop.displays.histology_display import (
    DesktopHistologyDisplayConfig,
)
from ephys_alignment_gui.desktop.displays.slice_display import DesktopSliceDisplayConfig


@dataclass(frozen=True)
class DesktopDisplayConfigHandles:
    """Shell handles needed to configure desktop display regions."""

    depth_view: Callable[[], Any]
    linear_fit_enabled: Callable[[], bool]
    histology_data_loaded: Callable[[], bool]
    reset_axis: Callable[..., Any]
    cluster_clicked: Callable[..., Any]
    on_mouse_double_clicked: Callable[..., Any]
    on_mouse_hover: Callable[..., Any]
    on_linear_fit_changed: Callable[..., Any]
    solid_pen: Any
    dotted_pen: Any
    fit_pen: Any
    linear_fit_pen: Any
    reference_line_pen: Any
    padding: Callable[[], Any]


def desktop_ephys_display_config_from_handles(
    handles: DesktopDisplayConfigHandles,
) -> DesktopEphysDisplayConfig:
    """Adapt explicit shell handles to ephys display config."""
    return DesktopEphysDisplayConfig(
        depth_view=handles.depth_view(),
        line_pen=handles.solid_pen,
        depth_guide_pen=handles.dotted_pen,
        padding_provider=handles.padding,
        set_axis=set_axis,
        reset_axis=handles.reset_axis,
        cluster_clicked=handles.cluster_clicked,
        on_mouse_double_clicked=handles.on_mouse_double_clicked,
        on_mouse_hover=handles.on_mouse_hover,
    )


def desktop_histology_display_config_from_handles(
    handles: DesktopDisplayConfigHandles,
) -> DesktopHistologyDisplayConfig:
    """Adapt explicit shell handles to histology display config."""
    return DesktopHistologyDisplayConfig(
        depth_view=handles.depth_view(),
        dotted_pen=handles.dotted_pen,
        fit_pen=handles.fit_pen,
        linear_fit_pen=handles.linear_fit_pen,
        baseline_pen=handles.dotted_pen,
        set_axis=set_axis,
        padding_provider=handles.padding,
        on_linear_fit_changed=handles.on_linear_fit_changed,
        on_mouse_double_clicked=handles.on_mouse_double_clicked,
        on_mouse_hover=handles.on_mouse_hover,
        linear_fit_enabled=handles.linear_fit_enabled,
    )


def desktop_slice_display_config_from_handles(
    handles: DesktopDisplayConfigHandles,
) -> DesktopSliceDisplayConfig:
    """Adapt explicit shell handles to slice display config."""
    return DesktopSliceDisplayConfig(
        depth_view=handles.depth_view(),
        dotted_pen=handles.dotted_pen,
        solid_pen=handles.solid_pen,
        reference_line_pen=handles.reference_line_pen,
        set_axis=set_axis,
        padding_provider=handles.padding,
        histology_exists=handles.histology_data_loaded,
    )


def desktop_display_config_from_handles(
    handles: DesktopDisplayConfigHandles,
) -> DesktopDisplayConfig:
    """Adapt explicit shell handle dependencies to display config."""
    return DesktopDisplayConfig(
        ephys=desktop_ephys_display_config_from_handles(handles),
        histology=desktop_histology_display_config_from_handles(handles),
        slice=desktop_slice_display_config_from_handles(handles),
    )
