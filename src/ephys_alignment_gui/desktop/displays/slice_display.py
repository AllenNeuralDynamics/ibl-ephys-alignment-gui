"""Desktop slice display composition."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.core.alignment_read_models import ActiveSliceMenuState
from ephys_alignment_gui.core.slice_display_policy import SliceSelection
from ephys_alignment_gui.desktop.displays.slice_panel_view import (
    SlicePanelView,
)
from ephys_alignment_gui.desktop.presenters.slice_panel_presenter import (
    SlicePanelPresenter,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopSliceDisplayConfig:
    """External style/callback dependencies needed to build the slice display."""

    dotted_pen: Any
    solid_pen: Any
    reference_line_pen: Any
    set_axis: Callable[..., Any]
    padding_provider: Callable[[], float]
    histology_exists: Callable[[], bool]


@dataclass(frozen=True)
class SliceSelectionSnapshot:
    """Selected slice menu entry captured before a shank redraw."""

    selection: SliceSelection | None = None
    label: str | None = None


@dataclass
class _SliceMenuHandles:
    menu: Any = None
    action_group: Any = None
    initial_action: Any = None
    parent: Any = None


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
        self.handles.menu = slice_options
        self.handles.parent = parent

        menu_state = self.app.queries.slices.active_slice_menu_state(offline=offline)
        self.render_menu(menu_state)

    def render_menu(self, menu_state: ActiveSliceMenuState | None) -> None:
        """Render Slice Plots menu actions from the current slice-menu state."""
        menu = self.handles.menu
        if menu is None:
            return
        clear = getattr(menu, "clear", None)
        if callable(clear):
            clear()
        action_group = self.action_group_factory(menu)
        action_group.setExclusive(True)
        self.handles.action_group = action_group
        self.handles.initial_action = None

        if menu_state is None:
            self._set_menu_enabled(False)
            return

        selected_selection = menu_state.selection.selection
        selected_action = None
        for item in menu_state.items:
            action = self.action_factory(
                item.label,
                self.handles.parent,
                checkable=True,
                checked=False,
            )
            action.setData(item.selection.to_payload())
            action.triggered.connect(
                lambda _checked=False, selection=item.selection: (
                    self.panel.render_slice_selection(selection)
                )
            )
            menu.addAction(action)
            action_group.addAction(action)
            if item.selection == menu_state.default_selection:
                self.handles.initial_action = action
            if item.selection == selected_selection:
                selected_action = action

        actions = action_group.actions()
        if self.handles.initial_action is None and actions:
            self.handles.initial_action = actions[0]
        if selected_action is None:
            selected_action = self.handles.initial_action
        if selected_action is not None:
            selected_action.setChecked(True)
        self._set_menu_enabled(bool(actions))

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

        self.render_menu(slice_menu_state)
        choice = slice_menu_state.selection
        selected_action = self.action_for_selection(choice.selection)
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
            self.panel.render_slice_selection(selected_selection)

    def _set_menu_enabled(self, enabled: bool) -> None:
        menu = self.handles.menu
        set_enabled = getattr(menu, "setEnabled", None)
        if callable(set_enabled):
            set_enabled(enabled)

    def checked_action(self) -> Any:
        """Return the checked slice QAction, if the menu exists."""
        action_group = self.handles.action_group
        if action_group is None:
            return None
        return action_group.checkedAction()

    def current_selection(self) -> SliceSelection | None:
        """Return the slice selection stored on the checked QAction."""
        action = self.checked_action()
        if action is None:
            return None
        return SliceSelection.from_payload(action.data())

    def action_for_selection(self, selection: SliceSelection) -> Any:
        """Find the QAction that represents a slice selection."""
        action_group = self.handles.action_group
        if action_group is None:
            return None
        for action in action_group.actions():
            action_selection = SliceSelection.from_payload(action.data())
            if action_selection == selection:
                return action
        return None

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
        config: DesktopSliceDisplayConfig,
        action_factory: Callable[..., Any] = QtWidgets.QAction,
        action_group_factory: Callable[..., Any] = QtWidgets.QActionGroup,
        view_factory: Callable[..., SlicePanelView] = SlicePanelView.create,
    ) -> DesktopSliceDisplay:
        """Build the slice display cluster from desktop dependencies."""
        handles = _SliceMenuHandles()
        view = view_factory(
            depth_view=app.queries.workspace.depth_view_settings(),
            padding=config.padding_provider(),
            set_axis=config.set_axis,
            dotted_pen=config.dotted_pen,
            solid_pen=config.solid_pen,
            reference_line_pen=config.reference_line_pen,
            histology_exists=config.histology_exists,
        )
        panel = SlicePanelPresenter(
            app=app,
            view=view,
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

    @property
    def area(self) -> Any:
        """Return the top-level coronal slice panel widget."""
        return self.panel.view.plots.area

    @property
    def coronal_plot(self) -> Any:
        """Return the coronal slice plot handle."""
        return self.panel.view.plots.coronal

    @property
    def perpendicular_plot(self) -> Any:
        """Return the perpendicular slice plot handle."""
        return self.panel.view.plots.perpendicular

    def set_perpendicular_depth_link(self, linked_plot: Any) -> None:
        """Link the perpendicular slice y-axis to the histology depth plot."""
        self.panel.view.set_perpendicular_depth_link(linked_plot)

    def capture_export_geometry(self) -> tuple[float, float, Any]:
        """Capture slice plot geometry for zoomed plot export."""
        return self.panel.view.capture_export_geometry()

    def clear(self) -> None:
        """Clear slice-panel plot items and forget desktop handles."""
        self.panel.clear()

    def attach_slice_menu(self, menu_bar: Any, *, parent: Any, offline: bool) -> None:
        """Attach the Slice Plots menu to a desktop menu bar."""
        self.menu_presenter.attach_menu(menu_bar, parent=parent, offline=offline)

    def render_slice_menu(self, slice_menu_state: ActiveSliceMenuState | None) -> None:
        """Render Slice Plots menu actions from the active shank menu state."""
        self.menu_presenter.render_menu(slice_menu_state)

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
        self.panel.render_slice_selection(selection)

    def refresh_perpendicular_histology(self) -> None:
        """Refresh perpendicular slice for the selected scalar slice."""
        self.panel.refresh_perpendicular_histology(
            self.menu_presenter.current_selection()
        )

    def plot_channels(self, projection: Any = None) -> None:
        """Render or update channel/tip overlays on the coronal slice."""
        self.panel.plot_channels(
            projection,
            selection=self.menu_presenter.current_selection(),
        )

    def toggle_channel_visibility(self) -> None:
        """Toggle channel, tip, trajectory, and perpendicular overlays."""
        self.panel.toggle_channel_visibility()

    def render_export_trajectory_overlay(self, pen: Any) -> None:
        """Render the coronal trajectory overlay used by overview exports."""
        self.panel.render_export_trajectory_overlay(
            pen,
            selection=self.menu_presenter.current_selection(),
        )

    def current_channel_locations_ras(self) -> Any | None:
        """Return channel locations for the current slice overlay."""
        return self.panel.current_channel_locations_ras(
            self.menu_presenter.current_selection()
        )
