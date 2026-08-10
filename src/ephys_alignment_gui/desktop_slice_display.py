"""Desktop slice display composition."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.alignment_read_models import ActiveSliceMenuState
from ephys_alignment_gui.slice_display_policy import SliceSelection
from ephys_alignment_gui.slice_panel_presenter import (
    SlicePanelPlots,
    SlicePanelPresenter,
    SlicePanelStyle,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopSliceDisplayPorts:
    """Desktop handles and callbacks needed to build the slice display."""

    coronal_plot: Any
    coronal_layout: Any
    histogram_alt: Any
    perpendicular_plot: Any
    dotted_pen: Any
    solid_pen: Any
    reference_line_pen: Any
    histology_exists: Callable[[], bool]
    slice_item: Any = None


@dataclass(frozen=True)
class SliceSelectionSnapshot:
    """Selected slice menu entry captured before a shank redraw."""

    selection: SliceSelection | None = None
    label: str | None = None


@dataclass
class _SliceMenuHandles:
    action_group: Any = None
    initial_action: Any = None


@dataclass
class DesktopSliceMenuPresenter:
    """Own desktop QAction state for the Slice Plots menu."""

    app: Any
    panel: SlicePanelPresenter
    handles: _SliceMenuHandles
    action_factory: Callable[..., Any] = QtWidgets.QAction
    action_group_factory: Callable[..., Any] = QtWidgets.QActionGroup

    def attach_menu(self, menu_bar: Any, *, parent: Any, offline: bool) -> None:
        """Create the Slice Plots menu on a desktop menu bar."""
        slice_options = menu_bar.addMenu("Slice Plots")
        action_group = self.action_group_factory(slice_options)
        action_group.setExclusive(True)
        self.handles.action_group = action_group
        self.handles.initial_action = None

        menu_state = self.app.queries.slices.active_slice_menu_state(offline=offline)
        if menu_state is None:
            return

        for item in menu_state.items:
            action = self.action_factory(
                item.label,
                parent,
                checkable=True,
                checked=False,
            )
            action.setData(item.selection.to_payload())
            action.triggered.connect(
                lambda _checked=False, selection=item.selection: (
                    self.panel.plot_slice_selection(selection)
                )
            )
            slice_options.addAction(action)
            action_group.addAction(action)
            if item.selection == menu_state.default_selection:
                self.handles.initial_action = action

        if self.handles.initial_action is None and action_group.actions():
            self.handles.initial_action = action_group.actions()[0]

    def capture_selection(self) -> SliceSelectionSnapshot:
        """Capture the checked slice menu action, if one exists."""
        action = self.checked_action()
        return SliceSelectionSnapshot(
            selection=SliceSelection.from_payload(
                action.data() if action is not None else None
            ),
            label=action.text() if action is not None else None,
        )

    def restore_selection(
        self,
        slice_menu_state: ActiveSliceMenuState | None,
        previous_selection: SliceSelection | None,
        previous_label: str | None,
    ) -> None:
        """Restore or choose the active slice menu selection after shank redraw."""
        if slice_menu_state is None:
            logger.warning("No default slice selection is available")
            return

        choice = slice_menu_state.selection
        selected_action = self.panel.action_for_selection(choice.selection)
        if selected_action is None:
            selected_action = self.handles.initial_action
        if selected_action is None:
            logger.warning("No slice action is available")
            return

        if (
            previous_selection is not None
            and SliceSelection.from_payload(selected_action.data())
            != previous_selection
            and not choice.used_previous
        ):
            logger.info(
                "Slice selection '%s' not available for this probe; "
                "falling back to '%s'",
                previous_label,
                selected_action.text(),
            )

        selected_action.setChecked(True)
        selected_selection = SliceSelection.from_payload(selected_action.data())
        if selected_selection is not None:
            self.panel.plot_slice_selection(selected_selection)

    def checked_action(self) -> Any:
        """Return the checked slice QAction, if the menu exists."""
        action_group = self.handles.action_group
        if action_group is None:
            return None
        return action_group.checkedAction()

    def toggle_plot(self, *, reverse: bool = False) -> None:
        """Toggle to the next available slice plot."""
        action_group = self.handles.action_group
        if action_group is None:
            logger.warning("No available slice plot actions to toggle")
            return
        current_action = action_group.checkedAction()
        actions = action_group.actions()
        if not actions:
            logger.warning("No available slice plot actions to toggle")
            return
        if current_action is None:
            self._trigger_action(actions[0])
            return
        try:
            current_idx = actions.index(current_action)
        except ValueError:
            self._trigger_action(actions[0])
            return
        next_idx = (current_idx + (-1 if reverse else 1)) % len(actions)
        self._trigger_action(actions[next_idx])

    @staticmethod
    def _trigger_action(action: Any) -> None:
        action.setChecked(True)
        action.trigger()


@dataclass(frozen=True)
class DesktopSliceDisplay:
    """Own the slice panel and slice menu presentation cluster."""

    panel: SlicePanelPresenter
    menu_presenter: DesktopSliceMenuPresenter
    handles: _SliceMenuHandles

    @classmethod
    def create(
        cls,
        *,
        app: Any,
        ports: DesktopSliceDisplayPorts,
        action_factory: Callable[..., Any] = QtWidgets.QAction,
        action_group_factory: Callable[..., Any] = QtWidgets.QActionGroup,
    ) -> DesktopSliceDisplay:
        """Build the slice display cluster from desktop ports."""
        handles = _SliceMenuHandles()
        panel = SlicePanelPresenter(
            app=app,
            plots=SlicePanelPlots(
                coronal=ports.coronal_plot,
                coronal_layout=ports.coronal_layout,
                histogram_alt=ports.histogram_alt,
                perpendicular=ports.perpendicular_plot,
            ),
            style=SlicePanelStyle(
                dotted_pen=ports.dotted_pen,
                solid_pen=ports.solid_pen,
                reference_line_pen=ports.reference_line_pen,
            ),
            histology_exists=ports.histology_exists,
            action_group_provider=lambda: handles.action_group,
            slice_item=ports.slice_item,
        )
        menu_presenter = DesktopSliceMenuPresenter(
            app=app,
            panel=panel,
            handles=handles,
            action_factory=action_factory,
            action_group_factory=action_group_factory,
        )
        return cls(
            panel=panel,
            menu_presenter=menu_presenter,
            handles=handles,
        )

    @property
    def action_group(self) -> Any:
        """Return the Slice Plots QActionGroup, if menus have been attached."""
        return self.handles.action_group

    def clear(self) -> None:
        """Clear slice-panel plot items and forget desktop handles."""
        self.panel.clear()

    def attach_slice_menu(self, menu_bar: Any, *, parent: Any, offline: bool) -> None:
        """Attach the Slice Plots menu to a desktop menu bar."""
        self.menu_presenter.attach_menu(menu_bar, parent=parent, offline=offline)

    def capture_selection(self) -> SliceSelectionSnapshot:
        """Capture the selected slice menu entry."""
        return self.menu_presenter.capture_selection()

    def restore_selection(
        self,
        slice_menu_state: ActiveSliceMenuState | None,
        previous_selection: SliceSelection | None,
        previous_label: str | None,
    ) -> None:
        """Restore or choose the active slice menu selection after shank redraw."""
        self.menu_presenter.restore_selection(
            slice_menu_state,
            previous_selection,
            previous_label,
        )

    def toggle_slice_plot(self, *, reverse: bool = False) -> None:
        """Toggle to the next available slice plot."""
        self.menu_presenter.toggle_plot(reverse=reverse)

    def plot_slice_selection(self, selection: SliceSelection) -> None:
        """Render a coronal slice selection."""
        self.panel.plot_slice_selection(selection)

    def refresh_perpendicular_histology(self) -> None:
        """Refresh perpendicular slice for the selected scalar slice."""
        self.panel.refresh_perpendicular_histology()

    def plot_channels(self, projection: Any = None) -> None:
        """Render or update channel/tip overlays on the coronal slice."""
        self.panel.plot_channels(projection)

    def toggle_channel_visibility(self) -> None:
        """Toggle channel, tip, trajectory, and perpendicular overlays."""
        self.panel.toggle_channel_visibility()

    def render_export_trajectory_overlay(self, pen: Any) -> None:
        """Render the coronal trajectory overlay used by overview exports."""
        self.panel.render_export_trajectory_overlay(pen)

    def current_channel_locations_ras(self) -> Any | None:
        """Return channel locations for the current slice overlay."""
        return self.panel.current_channel_locations_ras()
