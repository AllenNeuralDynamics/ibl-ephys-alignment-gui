"""Desktop pyqtgraph figure construction."""

from __future__ import annotations

import pyqtgraph as pg
import pyqtgraph.exporters as pg_exporters
from PyQt5 import QtWidgets

from ephys_alignment_gui.desktop.plot_elements import replace_axis


def configure_pyqtgraph() -> None:
    """Apply global pyqtgraph defaults used by the desktop GUI."""
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")


def initialize_figures(window) -> None:
    """Create all pyqtgraph figures used by the main window."""
    depth_view = window.app.queries.workspace.depth_view_settings()
    window.probe_top_lines = []
    window.probe_tip_lines = []

    _initialize_ephys_figures(window, depth_view)
    _initialize_histology_figures(window, depth_view)
    _initialize_slice_figures(window)
    _initialize_fit_figure(window, depth_view)


def _add_depth_guides(window, plot, depth_view) -> None:
    window.probe_tip_lines.append(
        plot.addLine(y=depth_view.probe_tip_um, pen=window.kpen_dot, z=50)
    )
    window.probe_top_lines.append(
        plot.addLine(y=depth_view.probe_top_um, pen=window.kpen_dot, z=50)
    )


def _set_depth_range(window, plot, depth_view) -> None:
    y_min, y_max = depth_view.plot_y_range_um
    plot.setYRange(min=y_min, max=y_max, padding=window.pad)


def _initialize_ephys_figures(window, depth_view) -> None:
    window.fig_img = pg.PlotItem()
    _set_depth_range(window, window.fig_img, depth_view)
    window.fig_img.setMouseEnabled(x=False, y=True)
    _add_depth_guides(window, window.fig_img, depth_view)
    window.set_axis(window.fig_img, "bottom")
    window.fig_data_ax = window.set_axis(
        window.fig_img, "left", label="Distance from probe tip (uV)"
    )

    window.fig_img_cb = pg.PlotItem()
    window.fig_img_cb.setMaximumHeight(70)
    window.fig_img_cb.setMouseEnabled(x=False, y=False)
    window.set_axis(window.fig_img_cb, "bottom", show=False)
    window.set_axis(window.fig_img_cb, "left", pen="w")
    window.set_axis(window.fig_img_cb, "top", pen="w")

    window.fig_line = pg.PlotItem()
    window.fig_line.setMouseEnabled(x=False, y=True)
    _set_depth_range(window, window.fig_line, depth_view)
    _add_depth_guides(window, window.fig_line, depth_view)
    window.set_axis(window.fig_line, "bottom")
    window.set_axis(window.fig_line, "left", show=False)

    window.fig_probe = pg.PlotItem()
    window.fig_probe.setMouseEnabled(x=False, y=False)
    window.fig_probe.setMaximumWidth(50)
    _set_depth_range(window, window.fig_probe, depth_view)
    _add_depth_guides(window, window.fig_probe, depth_view)
    window.set_axis(window.fig_probe, "bottom", pen="w")
    window.set_axis(window.fig_probe, "left", show=False)

    window.fig_probe_cb = pg.PlotItem()
    window.fig_probe_cb.setMouseEnabled(x=False, y=False)
    window.fig_probe_cb.setMaximumHeight(70)
    window.set_axis(window.fig_probe_cb, "bottom", show=False)
    window.set_axis(window.fig_probe_cb, "left", show=False)
    window.set_axis(window.fig_probe_cb, "top", pen="w")

    window.fig_data_area = pg.GraphicsLayoutWidget()
    window.fig_data_area.scene().sigMouseClicked.connect(window.on_mouse_double_clicked)
    window.fig_data_area.scene().sigMouseHover.connect(window.on_mouse_hover)
    window.fig_data_layout = pg.GraphicsLayout()

    window.fig_data_layout.addItem(window.fig_img_cb, 0, 0)
    window.fig_data_layout.addItem(window.fig_probe_cb, 0, 1, 1, 2)
    window.fig_data_layout.addItem(window.fig_img, 1, 0)
    window.fig_data_layout.addItem(window.fig_line, 1, 1)
    window.fig_data_layout.addItem(window.fig_probe, 1, 2)
    window.fig_data_layout.layout.setColumnStretchFactor(0, 6)
    window.fig_data_layout.layout.setColumnStretchFactor(1, 1)
    window.fig_data_layout.layout.setColumnStretchFactor(2, 1)
    window.fig_data_layout.layout.setRowStretchFactor(0, 1)
    window.fig_data_layout.layout.setRowStretchFactor(1, 10)

    window.fig_data_area.addItem(window.fig_data_layout)


def _initialize_histology_figures(window, depth_view) -> None:
    window.fig_hist = pg.PlotItem()
    window.fig_hist.setContentsMargins(0, 0, 0, 0)
    window.fig_hist.setMouseEnabled(x=False)
    _set_depth_range(window, window.fig_hist, depth_view)
    window.set_axis(window.fig_hist, "bottom", pen="w")

    window.fig_img.setYLink(window.fig_line)
    window.fig_img.setYLink(window.fig_hist)
    window.fig_line.setYLink(window.fig_hist)
    window.fig_probe.setYLink(window.fig_img)

    replace_axis(window.fig_hist)
    window.ax_hist = window.set_axis(window.fig_hist, "left", pen=None)
    window.ax_hist.setWidth(0)

    window.fig_scale = pg.PlotItem()
    window.fig_scale.setMaximumWidth(50)
    window.fig_scale.setMouseEnabled(x=False)
    window.scale_label = pg.LabelItem(color="k")
    window.set_axis(window.fig_scale, "bottom", pen="w")
    window.set_axis(window.fig_scale, "left", show=False)
    window.fig_scale.setYLink(window.fig_hist)

    window.fig_scale_cb = pg.PlotItem()
    window.fig_scale_cb.setMouseEnabled(x=False, y=False)
    window.fig_scale_cb.setMaximumHeight(70)
    window.set_axis(window.fig_scale_cb, "bottom", show=False)
    window.set_axis(window.fig_scale_cb, "left", show=False)
    window.fig_scale_ax = window.set_axis(window.fig_scale_cb, "top", pen="w")
    window.set_axis(window.fig_scale_cb, "right", show=False)

    window.fig_hist_ref = pg.PlotItem()
    window.fig_hist_ref.setMouseEnabled(x=False)
    _set_depth_range(window, window.fig_hist_ref, depth_view)
    window.fig_hist_ref.setYLink(window.fig_hist)
    window.set_axis(window.fig_hist_ref, "bottom", pen="w")
    window.set_axis(window.fig_hist_ref, "left", show=False)
    replace_axis(window.fig_hist_ref, orientation="right", pos=(2, 2))
    window.ax_hist_ref = window.set_axis(window.fig_hist_ref, "right", pen=None)
    window.ax_hist_ref.setWidth(0)

    window.fig_hist_perp = pg.PlotItem()
    window.fig_hist_perp.setContentsMargins(0, 0, 0, 0)
    window.fig_hist_perp.setMouseEnabled(x=False)
    _set_depth_range(window, window.fig_hist_perp, depth_view)
    window.set_axis(window.fig_hist_perp, "bottom", pen="w")
    window.set_axis(window.fig_hist_perp, "left", show=False)
    window.fig_hist_perp.setYLink(window.fig_hist)

    window.fig_hist_area = pg.GraphicsLayoutWidget()
    window.fig_hist_area.setMouseTracking(True)
    window.fig_hist_area.scene().sigMouseClicked.connect(window.on_mouse_double_clicked)
    window.fig_hist_area.scene().sigMouseHover.connect(window.on_mouse_hover)

    window.fig_hist_extra_yaxis = pg.PlotItem()
    window.fig_hist_extra_yaxis.setMouseEnabled(x=False, y=False)
    window.fig_hist_extra_yaxis.setMaximumWidth(2)
    _set_depth_range(window, window.fig_hist_extra_yaxis, depth_view)

    window.set_axis(window.fig_hist_extra_yaxis, "bottom", pen="w")
    window.ax_hist2 = window.set_axis(window.fig_hist_extra_yaxis, "left", pen=None)
    window.ax_hist2.setWidth(10)

    window.fig_hist_layout = pg.GraphicsLayout()
    window.fig_hist_layout.addItem(window.fig_scale_cb, 0, 0, 1, 5)
    window.fig_hist_layout.addItem(window.fig_hist_extra_yaxis, 1, 0)
    window.fig_hist_layout.addItem(window.fig_hist, 1, 1)
    window.fig_hist_layout.addItem(window.fig_hist_perp, 1, 2)
    window.fig_hist_layout.addItem(window.fig_scale, 1, 3)
    window.fig_hist_layout.addItem(window.fig_hist_ref, 1, 4)
    window.fig_hist_layout.layout.setColumnStretchFactor(0, 1)
    window.fig_hist_layout.layout.setColumnStretchFactor(1, 4)
    window.fig_hist_layout.layout.setColumnStretchFactor(2, 5)
    window.fig_hist_layout.layout.setColumnStretchFactor(3, 1)
    window.fig_hist_layout.layout.setColumnStretchFactor(4, 4)
    window.fig_hist_layout.layout.setRowStretchFactor(0, 1)
    window.fig_hist_layout.layout.setRowStretchFactor(1, 10)
    window.fig_hist_area.addItem(window.fig_hist_layout)


def _initialize_slice_figures(window) -> None:
    window.fig_slice_area = pg.GraphicsLayoutWidget()
    window.fig_slice_layout = pg.GraphicsLayout()
    window.fig_slice_hist_alt = pg.ViewBox()
    window.fig_slice = pg.ViewBox()
    window.fig_slice_layout.addItem(window.fig_slice, 0, 0)
    window.fig_slice_layout.addItem(window.fig_slice_hist_alt, 0, 1)
    window.fig_slice_layout.layout.setColumnStretchFactor(0, 3)
    window.fig_slice_layout.layout.setColumnStretchFactor(1, 1)
    window.fig_slice_area.addItem(window.fig_slice_layout)
    window.slice_item = window.fig_slice_hist_alt


def _initialize_fit_figure(window, depth_view) -> None:
    window.fig_fit = pg.PlotWidget(background="w")
    window.fig_fit.setMouseEnabled(x=False, y=False)
    window.fig_fit_exporter = pg_exporters.ImageExporter(window.fig_fit.plotItem)
    window.fig_fit.sigDeviceRangeChanged.connect(
        lambda *args: position_linear_fit_checkbox(window)
    )
    view_min, view_max = depth_view.view_range_um
    window.fig_fit.setXRange(min=view_min, max=view_max)
    window.fig_fit.setYRange(min=view_min, max=view_max)
    window.set_axis(window.fig_fit, "bottom", label="Ephys reference depth (μm)")
    window.set_axis(window.fig_fit, "left", label="Atlas reference depth (μm)")

    plot = pg.PlotCurveItem()
    plot.setData(
        x=depth_view.fit_depth_um,
        y=depth_view.fit_depth_um,
        pen=window.kpen_dot,
    )
    window.fit_plot = pg.PlotCurveItem(pen=window.bpen_solid)
    window.fit_scatter = pg.ScatterPlotItem(size=7, symbol="o", brush="w", pen="b")
    window.fit_plot_lin = pg.PlotCurveItem(pen=window.rpen_dot)
    window.fig_fit.addItem(plot)
    window.fig_fit.addItem(window.fit_plot)
    window.fig_fit.addItem(window.fit_plot_lin)
    window.fig_fit.addItem(window.fit_scatter)

    window.lin_fit_option = QtWidgets.QCheckBox("Linear fit", window.fig_fit)
    window.lin_fit_option.setChecked(window.app.queries.workspace.linear_fit_enabled())
    window.lin_fit_option.stateChanged.connect(window.lin_fit_option_changed)
    position_linear_fit_checkbox(window)


def position_linear_fit_checkbox(window) -> None:
    window.lin_fit_option.move(70, 10)
