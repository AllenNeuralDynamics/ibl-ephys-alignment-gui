"""Tests for desktop slice-menu QAction coordination."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.core.alignment_read_models import ActiveSliceMenuState
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.slice_display_policy import (
    SliceMenuItem,
    SliceSelection,
    SliceSelectionDecision,
)
from ephys_alignment_gui.desktop.coordinators.slice_menu_coordinator import (
    DesktopSliceMenuCoordinator,
)


class FakeSignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class FakeAction:
    def __init__(
        self,
        label: str,
        _parent: Any,
        *,
        checkable: bool = False,
        checked: bool = False,
    ) -> None:
        self._text = label
        self.checkable = checkable
        self.checked = checked
        self._data = None
        self.triggered = FakeSignal()

    def setData(self, data: Any) -> None:
        self._data = data

    def data(self) -> Any:
        return self._data

    def setChecked(self, checked: bool) -> None:
        self.checked = checked

    def text(self) -> str:
        return self._text

    def trigger(self) -> None:
        self.triggered.emit(False)


class FakeActionGroup:
    def __init__(self, _parent: Any) -> None:
        self._actions = []
        self.exclusive = False

    def setExclusive(self, exclusive: bool) -> None:
        self.exclusive = exclusive

    def addAction(self, action: FakeAction) -> None:
        self._actions.append(action)
        action.triggered.connect(lambda _checked=False: self._set_checked(action))

    def actions(self) -> list[FakeAction]:
        return list(self._actions)

    def checkedAction(self) -> FakeAction | None:
        return next((action for action in self._actions if action.checked), None)

    def _set_checked(self, selected: FakeAction) -> None:
        for action in self._actions:
            action.setChecked(action is selected)


class FakeMenu:
    def __init__(self, title: str) -> None:
        self.title = title
        self.actions: list[FakeAction] = []
        self.enabled = True

    def clear(self) -> None:
        self.actions.clear()

    def addAction(self, action: FakeAction) -> None:
        self.actions.append(action)

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


class FakeMenuBar:
    def __init__(self) -> None:
        self.menus: dict[str, FakeMenu] = {}

    def addMenu(self, title: str) -> FakeMenu:
        menu = FakeMenu(title)
        self.menus[title] = menu
        return menu


class FakeQueries:
    def __init__(self, menu_state: ActiveSliceMenuState | None) -> None:
        self.menu_state = menu_state
        self.offline_values: list[bool] = []
        self.slices = SimpleNamespace(
            active_slice_menu_state=self.active_slice_menu_state,
        )

    def active_slice_menu_state(
        self,
        *,
        offline: bool,
    ) -> ActiveSliceMenuState | None:
        self.offline_values.append(offline)
        return self.menu_state


class FakePanel:
    def __init__(self) -> None:
        self.calls: list[SliceSelection] = []

    def render_slice_selection(self, selection: SliceSelection) -> None:
        self.calls.append(selection)


def _selection(key: str) -> SliceSelection:
    return SliceSelection("slice_data", key)


def _menu_state(
    *,
    default: SliceSelection,
    selected: SliceSelection,
    used_previous: bool = False,
) -> ActiveSliceMenuState:
    return ActiveSliceMenuState(
        key=AlignmentKey("rec", "stream", 0),
        items=(
            SliceMenuItem("CCF", _selection("ccf")),
            SliceMenuItem("Registration", _selection("histology_registration")),
        ),
        default_selection=default,
        selection=SliceSelectionDecision(selected, used_previous=used_previous),
    )


def _coordinator(
    menu_state: ActiveSliceMenuState | None,
) -> tuple[DesktopSliceMenuCoordinator, FakeQueries, FakePanel]:
    queries = FakeQueries(menu_state)
    panel = FakePanel()
    coordinator = DesktopSliceMenuCoordinator.create(
        app=SimpleNamespace(queries=queries),
        panel=panel,
        action_factory=FakeAction,
        action_group_factory=FakeActionGroup,
    )
    return coordinator, queries, panel


def test_slice_menu_attaches_and_captures_checked_selection() -> None:
    default = _selection("histology_registration")
    coordinator, queries, _panel = _coordinator(
        _menu_state(default=default, selected=default)
    )
    menu_bar = FakeMenuBar()

    coordinator.attach_menu(menu_bar, parent=object(), offline=True)
    actions = coordinator.action_group.actions()
    actions[1].setChecked(True)

    snapshot = coordinator.capture_selection()

    assert queries.offline_values == [True]
    assert menu_bar.menus["Slice Plots"].actions == actions
    assert coordinator.handles.initial_action is actions[1]
    assert snapshot.selection == default
    assert snapshot.label == "Registration"


def test_slice_menu_restores_selection_and_reports_fallback(caplog) -> None:
    caplog.set_level(logging.INFO)
    previous = _selection("missing")
    selected = _selection("ccf")
    coordinator, _queries, panel = _coordinator(
        _menu_state(
            default=_selection("histology_registration"),
            selected=selected,
        )
    )
    coordinator.attach_menu(FakeMenuBar(), parent=object(), offline=False)

    coordinator.restore_selection(
        coordinator.app.queries.menu_state,
        previous,
        "Old",
    )

    assert panel.calls == [selected]
    assert coordinator.action_group.checkedAction().text() == "CCF"
    assert "falling back to 'CCF'" in caplog.text


def test_slice_menu_rerenders_actions_before_restore() -> None:
    previous_state = _menu_state(
        default=_selection("ccf"),
        selected=_selection("ccf"),
    )
    fallback = _selection("histology_registration")
    next_state = _menu_state(
        default=fallback,
        selected=fallback,
    )
    coordinator, queries, panel = _coordinator(previous_state)
    coordinator.attach_menu(FakeMenuBar(), parent=object(), offline=True)
    queries.menu_state = next_state

    coordinator.restore_selection(
        next_state,
        _selection("old_channel"),
        "Old channel",
    )

    assert [action.text() for action in coordinator.action_group.actions()] == [
        "CCF",
        "Registration",
    ]
    assert panel.calls == [fallback]
    assert coordinator.action_group.checkedAction().text() == "Registration"


def test_slice_menu_toggles_slice_actions() -> None:
    coordinator, _queries, panel = _coordinator(
        _menu_state(
            default=_selection("ccf"),
            selected=_selection("ccf"),
        )
    )
    coordinator.attach_menu(FakeMenuBar(), parent=object(), offline=True)

    coordinator.toggle_plot()
    coordinator.toggle_plot()
    coordinator.toggle_plot(reverse=True)

    assert panel.calls == [
        _selection("histology_registration"),
        _selection("ccf"),
        _selection("histology_registration"),
    ]
