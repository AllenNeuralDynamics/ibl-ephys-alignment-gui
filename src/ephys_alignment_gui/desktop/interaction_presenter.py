"""Desktop popup and mouse-interaction presentation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

import ephys_alignment_gui.ephys_gui_setup as ephys_gui

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopInteractionWidgets:
    """Desktop widgets/plots used by interaction presentation."""

    struct_list: Any
    struct_view: Any
    struct_description: Any
    scale_plot: Any
    histology_plot: Any
    histology_reference_plot: Any
    scale_axis: Any
    bar_colour: Any
    line_pen: Any


@dataclass(frozen=True)
class DesktopInteractionCallbacks:
    """Callbacks back into app/view objects used by desktop interactions."""

    histology_available: Callable[[], bool]
    activate_window: Callable[[], None]
    set_axis: Callable[..., Any]
    capture_pending_reference_lines: Callable[[], None]


@dataclass
class DesktopInteractionPresenter:
    """Coordinate desktop popups, hover dispatch, and reference-line clicks."""

    app: Any
    popup_manager: Any
    ephys_panel: Any
    histology_display: Any
    reference_line_display: Any
    widgets: DesktopInteractionWidgets
    callbacks: DesktopInteractionCallbacks
    popup_window_factory: Callable[..., Any] = ephys_gui.PopupWindow
    text_edit_factory: Callable[[], Any] = QtWidgets.QTextEdit
    plot_item_factory: Callable[[], Any] = pg.PlotItem
    bar_graph_item_factory: Callable[..., Any] = pg.BarGraphItem
    plot_curve_item_factory: Callable[[], Any] = pg.PlotCurveItem
    infinite_line_type: type = pg.InfiniteLine
    linear_region_type: type = pg.LinearRegionItem

    @staticmethod
    def _resolve_widget(widget_or_factory: Any) -> Any:
        """Return a widget from either a direct handle or a zero-arg factory."""
        return widget_or_factory() if callable(widget_or_factory) else widget_or_factory

    def _struct_list(self) -> Any:
        return self._resolve_widget(self.widgets.struct_list)

    def _struct_view(self) -> Any:
        return self._resolve_widget(self.widgets.struct_view)

    def _struct_description(self) -> Any:
        return self._resolve_widget(self.widgets.struct_description)

    def initialize_region_lookup(
        self,
        init_region_lookup: Callable[[Any], None],
    ) -> None:
        """Populate desktop region lookup widgets from app atlas metadata."""
        allen = self.app.queries.workspace.allen_structure_tree()
        if allen is None:
            raise RuntimeError("Allen structure metadata is unavailable")
        init_region_lookup(allen)

    def display_session_notes(self) -> None:
        """Show session notes for the active stream."""
        notes_window = self.popup_window_factory(
            title="Session notes from Alyx",
            size=(200, 100),
            graphics=False,
        )
        notes = self.text_edit_factory()
        notes.setReadOnly(True)
        notes.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        notes.setText(self.app.queries.ephys.active_session_notes())
        notes_window.layout.addWidget(notes)
        self.popup_manager.notes_window = notes_window

    def popup_closed(self, popup: Any) -> None:
        """Forget a closed cluster popup."""
        self.popup_manager.remove_cluster_popup(popup)

    def popup_moved(self) -> None:
        """Bring the main window back to front after popup movement."""
        self.callbacks.activate_window()

    def close_popups(self) -> None:
        """Close cluster detail popups."""
        self.popup_manager.close_cluster_popups()

    def minimise_popups(self) -> None:
        """Toggle cluster detail popups between minimized and normal."""
        self.popup_manager.toggle_cluster_minimized()
        self.callbacks.activate_window()

    def cluster_clicked(self, _item: Any, point: Any) -> Any | None:
        """Open cluster detail popup for a clicked ephys cluster point."""
        point_pos = point[0].pos()
        clust_idx = self.ephys_panel.cluster_index_for_plot_x(point_pos.x())
        if clust_idx is None:
            logger.error("Cannot show cluster detail: clicked point is not a cluster")
            return None

        detail = self.app.queries.ephys.active_cluster_detail(clust_idx)
        if detail is None:
            logger.error(
                "Cannot show cluster detail: active ephys stream is not loaded"
            )
            return None

        autocorr_plot = self.plot_item_factory()
        autocorr_plot.setXRange(
            min=np.min(detail.t_autocorr),
            max=np.max(detail.t_autocorr),
        )
        autocorr_plot.setYRange(min=0, max=1.05 * np.max(detail.autocorr))
        self.callbacks.set_axis(autocorr_plot, "bottom", label="T (ms)")
        self.callbacks.set_axis(autocorr_plot, "left", label="Number of spikes")
        autocorr_bars = self.bar_graph_item_factory(
            x=detail.t_autocorr,
            height=detail.autocorr,
            width=0.24,
            brush=self.widgets.bar_colour,
        )
        autocorr_plot.addItem(autocorr_bars)

        template_plot = self.plot_item_factory()
        template_curve = self.plot_curve_item_factory()
        template_plot.setXRange(
            min=np.min(detail.t_template),
            max=np.max(detail.t_template),
        )
        self.callbacks.set_axis(template_plot, "bottom", label="T (ms)")
        self.callbacks.set_axis(template_plot, "left", label="Amplitude (a.u.)")
        template_curve.setData(
            x=detail.t_template,
            y=detail.template_waveform,
            pen=self.widgets.line_pen,
        )
        template_plot.addItem(template_curve)

        clust_win = self.popup_window_factory(title=f"Cluster {detail.cluster_no}")
        clust_win.closed.connect(self.popup_closed)
        clust_win.moved.connect(self.popup_moved)
        clust_win.popup_widget.addItem(autocorr_plot, 0, 0)
        clust_win.popup_widget.addItem(template_plot, 1, 0)
        self.popup_manager.add_cluster_popup(clust_win)
        self.callbacks.activate_window()
        return detail.cluster_no

    def describe_labels_pressed(self) -> bool:
        """Show region information for the selected histology label."""
        if not self.callbacks.histology_available():
            return False

        idx = self.histology_display.selected_region_index()
        if idx is None:
            return False
        region_id = self.app.queries.alignment_render.active_histology_region_id(idx)
        if region_id is None:
            return False
        region_description = self.app.queries.workspace.region_description(region_id)
        if region_description is None:
            return False
        description, lookup = region_description
        if not self._select_structure(lookup, description, scroll=True):
            return False

        if self.popup_manager.label_window is None:
            label_window = self.popup_window_factory(
                title="Structure Information",
                size=(500, 700),
                graphics=False,
            )
            label_window.layout.addWidget(self._struct_view())
            label_window.layout.addWidget(self._struct_description())
            label_window.layout.setRowStretch(0, 7)
            label_window.layout.setRowStretch(1, 3)
            label_window.closed.connect(self.label_closed)
            label_window.moved.connect(self.label_moved)
            self.popup_manager.label_window = label_window
            self.callbacks.activate_window()
        else:
            self.popup_manager.label_window.show()
            self.callbacks.activate_window()
        return True

    def label_closed(self, _popup: Any) -> None:
        """Hide the label popup without forgetting reusable widgets."""
        if self.popup_manager.label_window is not None:
            self.popup_manager.label_window.hide()

    def label_moved(self) -> None:
        """Bring the main window back to front after label popup movement."""
        self.callbacks.activate_window()

    def label_pressed(self, item: Any) -> None:
        """Render region information for a clicked structure tree item."""
        idx = int(item.model().itemFromIndex(item).accessibleText())
        region_description = self.app.queries.workspace.region_description(idx)
        if region_description is None:
            return
        description, lookup = region_description
        self._select_structure(lookup, description)

    def on_mouse_double_clicked(self, event: Any) -> bool:
        """Add a reference line from a double-clicked feature plot position."""
        if not self.callbacks.histology_available():
            return False
        if not event.double():
            return False

        feature_y_um = self.ephys_panel.feature_y_from_scene(event.scenePos())
        if feature_y_um is None:
            return False
        self.reference_line_display.create_lines([feature_y_um])
        self.callbacks.capture_pending_reference_lines()
        return True

    def on_mouse_hover(self, items: list[Any]) -> None:
        """Dispatch hover interactions to reference-line and histology views."""
        if len(items) <= 1:
            return

        self.reference_line_display.clear_selection()
        if isinstance(items[0], self.infinite_line_type):
            self.reference_line_display.select_line(items[0])
        elif items[0] is self.widgets.scale_plot and isinstance(
            items[1],
            self.linear_region_type,
        ):
            scale_factor = self.histology_display.scale_factor_for_region_item(items[1])
            if scale_factor is not None:
                self.widgets.scale_axis.setLabel(
                    "Scale Factor = " + str(np.around(scale_factor, 2))
                )
        elif items[0] is self.widgets.histology_plot and isinstance(
            items[1],
            self.linear_region_type,
        ):
            self.histology_display.select_region(items[1])
        elif items[0] is self.widgets.histology_reference_plot and isinstance(
            items[1],
            self.linear_region_type,
        ):
            self.histology_display.select_region(items[1])

    def _select_structure(
        self,
        lookup: str,
        description: str,
        *,
        scroll: bool = False,
    ) -> bool:
        struct_list = self._struct_list()
        items = struct_list.findItems(
            lookup,
            flags=QtCore.Qt.MatchRecursive,
        )
        if not items:
            logger.error("Could not find structure %s in region tree", lookup)
            return False
        struct_view = self._struct_view()
        struct_description = self._struct_description()
        model_item = struct_list.indexFromItem(items[0])
        if scroll:
            struct_view.collapseAll()
            struct_view.scrollTo(model_item)
        struct_view.setCurrentIndex(model_item)
        struct_description.setText(description)
        return True
