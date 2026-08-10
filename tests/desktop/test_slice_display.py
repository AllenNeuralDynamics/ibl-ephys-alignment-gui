"""Tests for desktop slice display composition."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.alignment_read_models import ActiveSliceMenuState
from ephys_alignment_gui.desktop.slice_display import (
    DesktopSliceDisplay,
    DesktopSliceDisplayPorts,
)
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.slice_display_policy import (
    SliceMenuItem,
    SliceSelection,
    SliceSelectionDecision,
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

    def addAction(self, action: FakeAction) -> None:
        self.actions.append(action)


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

    def active_slice_menu_state(self, *, offline: bool) -> ActiveSliceMenuState | None:
        self.offline_values.append(offline)
        return self.menu_state


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


def _display(
    menu_state: ActiveSliceMenuState | None,
) -> tuple[DesktopSliceDisplay, FakeQueries, list[SliceSelection]]:
    queries = FakeQueries(menu_state)
    display = DesktopSliceDisplay.create(
        app=SimpleNamespace(queries=queries),
        ports=DesktopSliceDisplayPorts(
            coronal_plot=None,
            coronal_layout=None,
            histogram_alt=None,
            perpendicular_plot=None,
            dotted_pen=None,
            solid_pen=None,
            reference_line_pen=None,
            histology_exists=lambda: True,
        ),
        action_factory=FakeAction,
        action_group_factory=FakeActionGroup,
    )
    calls: list[SliceSelection] = []
    display.panel.plot_slice_selection = calls.append
    return display, queries, calls


def test_slice_display_attaches_menu_and_captures_checked_selection() -> None:
    default = _selection("histology_registration")
    display, queries, _calls = _display(_menu_state(default=default, selected=default))
    menu_bar = FakeMenuBar()

    display.attach_slice_menu(menu_bar, parent=object(), offline=True)
    actions = display.action_group.actions()
    actions[1].setChecked(True)

    snapshot = display.capture_selection()

    assert queries.offline_values == [True]
    assert menu_bar.menus["Slice Plots"].actions == actions
    assert display.handles.initial_action is actions[1]
    assert snapshot.selection == default
    assert snapshot.label == "Registration"


def test_slice_display_restores_selection_and_reports_fallback(caplog) -> None:
    caplog.set_level(logging.INFO)
    previous = _selection("missing")
    selected = _selection("ccf")
    display, _queries, calls = _display(
        _menu_state(
            default=_selection("histology_registration"),
            selected=selected,
        )
    )
    display.attach_slice_menu(FakeMenuBar(), parent=object(), offline=False)

    display.restore_selection(
        display.menu_presenter.app.queries.menu_state,
        previous,
        "Old",
    )

    assert calls == [selected]
    assert display.action_group.checkedAction().text() == "CCF"
    assert "falling back to 'CCF'" in caplog.text


def test_slice_display_toggles_slice_actions() -> None:
    display, _queries, calls = _display(
        _menu_state(
            default=_selection("ccf"),
            selected=_selection("ccf"),
        )
    )
    display.attach_slice_menu(FakeMenuBar(), parent=object(), offline=True)

    display.toggle_slice_plot()
    display.toggle_slice_plot()
    display.toggle_slice_plot(reverse=True)

    assert calls == [
        _selection("ccf"),
        _selection("histology_registration"),
        _selection("ccf"),
    ]
