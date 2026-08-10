"""Build desktop display-region ports from the MainWindow shell."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.desktop.displays import DesktopDisplayPorts
from ephys_alignment_gui.desktop.ephys_display import DesktopEphysDisplayPorts
from ephys_alignment_gui.desktop.histology_display import DesktopHistologyDisplayPorts
from ephys_alignment_gui.desktop.reference_line_display import (
    DesktopReferenceLineDisplayPorts,
)
from ephys_alignment_gui.desktop.slice_display import DesktopSliceDisplayPorts


def desktop_ephys_display_ports_from_main_window(
    window: Any,
) -> DesktopEphysDisplayPorts:
    """Adapt MainWindow plot handles to ephys display ports."""
    return DesktopEphysDisplayPorts(
        image_plot=window.fig_img,
        image_colorbar=window.fig_img_cb,
        line_plot=window.fig_line,
        probe_plot=window.fig_probe,
        probe_colorbar=window.fig_probe_cb,
        graphics_layout=window.fig_data_layout,
        line_pen=window.kpen_solid,
        raw_image_payloads=lambda: window.shank_screen_view.raw_image_payload_mapping(),
        set_axis=window.set_axis,
        reset_axis=window.reset_axis_button_pressed,
        cluster_clicked=lambda *args: window.desktop_workbench.cluster_clicked(*args),
    )


def desktop_histology_display_ports_from_main_window(
    window: Any,
) -> DesktopHistologyDisplayPorts:
    """Adapt MainWindow plot handles to histology display ports."""
    histology_available = window.app.queries.workspace.histology_data_loaded
    return DesktopHistologyDisplayPorts(
        aligned_plot=window.fig_hist,
        reference_plot=window.fig_hist_ref,
        scale_plot=window.fig_scale,
        scale_colorbar=window.fig_scale_cb,
        aligned_axis=window.ax_hist,
        reference_axis=window.ax_hist_ref,
        layout=window.fig_hist_layout,
        extra_y_axis=window.fig_hist_extra_yaxis,
        dotted_pen=window.kpen_dot,
        fit_curve=window.fit_plot,
        fit_scatter=window.fit_scatter,
        linear_fit_curve=window.fit_plot_lin,
        set_axis=window.set_axis,
        padding_provider=lambda: window.pad,
        scale_factor_y_range=window._scale_factor_y_range,
        histology_available=histology_available,
    )


def desktop_reference_line_display_ports_from_main_window(
    window: Any,
) -> DesktopReferenceLineDisplayPorts:
    """Adapt MainWindow plot handles to reference-line display ports."""
    return DesktopReferenceLineDisplayPorts(
        histology_plot=window.fig_hist,
        image_plot=window.fig_img,
        line_plot=window.fig_line,
        probe_plot=window.fig_probe,
        perpendicular_plot=window.fig_hist_perp,
        fit_plot=window.fig_fit,
    )


def desktop_slice_display_ports_from_main_window(
    window: Any,
) -> DesktopSliceDisplayPorts:
    """Adapt MainWindow plot handles to slice display ports."""
    histology_available = window.app.queries.workspace.histology_data_loaded
    return DesktopSliceDisplayPorts(
        coronal_plot=window.fig_slice,
        coronal_layout=window.fig_slice_layout,
        histogram_alt=window.fig_slice_hist_alt,
        perpendicular_plot=window.fig_hist_perp,
        dotted_pen=window.kpen_dot,
        solid_pen=window.kpen_solid,
        reference_line_pen=window.reference_line_kpen,
        histology_exists=histology_available,
        slice_item=window.slice_item,
    )


def desktop_display_ports_from_main_window(window: Any) -> DesktopDisplayPorts:
    """Adapt MainWindow plot handles to desktop display-region ports."""
    return DesktopDisplayPorts(
        ephys=desktop_ephys_display_ports_from_main_window(window),
        histology=desktop_histology_display_ports_from_main_window(window),
        reference_lines=desktop_reference_line_display_ports_from_main_window(window),
        slice=desktop_slice_display_ports_from_main_window(window),
    )
