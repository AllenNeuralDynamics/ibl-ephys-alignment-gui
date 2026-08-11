"""Desktop QAction coordination for the Slice Plots menu."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.core.alignment_read_models import ActiveSliceMenuState
from ephys_alignment_gui.core.slice_display_policy import SliceSelection
from ephys_alignment_gui.desktop.presenters.slice_panel_presenter import (
    SlicePanelPresenter,
)

logger = logging.getLogger(__name__)


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
class DesktopSliceMenuCoordinator:
    """Own desktop QAction state for the Slice Plots menu."""

    app: Any
    panel: SlicePanelPresenter
    handles: _SliceMenuHandles
    action_factory: Any = QtWidgets.QAction
    action_group_factory: Any = QtWidgets.QActionGroup

    @classmethod
    def create(
        cls,
        *,
        app: Any,
        panel: SlicePanelPresenter,
        action_factory: Any = QtWidgets.QAction,
        action_group_factory: Any = QtWidgets.QActionGroup,
    ) -> DesktopSliceMenuCoordinator:
        """Build a slice-menu coordinator with fresh Qt handle storage."""
        return cls(
            app=app,
            panel=panel,
            handles=_SliceMenuHandles(),
            action_factory=action_factory,
            action_group_factory=action_group_factory,
        )

    @property
    def action_group(self) -> Any:
        """Return the Slice Plots QActionGroup, if menus have been attached."""
        return self.handles.action_group

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

    def _set_menu_enabled(self, enabled: bool) -> None:
        menu = self.handles.menu
        set_enabled = getattr(menu, "setEnabled", None)
        if callable(set_enabled):
            set_enabled(enabled)

    @staticmethod
    def _trigger_action(action: Any) -> None:
        action.setChecked(True)
        action.trigger()
