"""Tests for desktop ephys plot menu presentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.alignment_read_models import ActiveShankScreenState
from ephys_alignment_gui.desktop_ephys_plot_presenter import (
    DesktopEphysPlotPresenter,
    EphysPlotRenderCallbacks,
)
from ephys_alignment_gui.plot_menu_state import PlotMenuGroupState, PlotMenuState
from ephys_alignment_gui.plot_registry import PlotSpec


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

    def setData(self, data) -> None:
        self._data = data

    def data(self):
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
        self.triggered = FakeSignal()
        self.exclusive = False

    def setExclusive(self, exclusive: bool) -> None:
        self.exclusive = exclusive

    def addAction(self, action: FakeAction) -> None:
        self._actions.append(action)
        action.triggered.connect(lambda _checked=False: self.triggered.emit(action))

    def actions(self) -> list[FakeAction]:
        return list(self._actions)

    def checkedAction(self) -> FakeAction | None:
        return next((action for action in self._actions if action.checked), None)


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
    def __init__(self) -> None:
        self.payload_calls = []
        self.bounds_calls = []
        self.ephys = SimpleNamespace(
            active_plot_payload=self.active_plot_payload,
            active_plot_bounds=self.active_plot_bounds,
        )

    def active_plot_payload(self, spec_key: str, *, raw_image_payloads):
        self.payload_calls.append((spec_key, raw_image_payloads))
        return {"payload": spec_key}

    def active_plot_bounds(self, spec_key: str, *, raw_image_payloads):
        self.bounds_calls.append((spec_key, raw_image_payloads))
        return [1, 2] if spec_key == "probe.depth" else None


class FakeCommands:
    def __init__(self) -> None:
        self.unit_filters = []

    def set_unit_filter(self, unit_filter: str) -> None:
        self.unit_filters.append(unit_filter)


def _spec(
    key: str,
    *,
    label: str | None = None,
    menu: str = "image",
    renderer: str = "image",
    default: bool = False,
) -> PlotSpec:
    return PlotSpec(
        key=key,
        label=label or key,
        menu=menu,
        renderer=renderer,
        source=lambda _plotdata: None,
        default=default,
    )


def _plot_menu_state() -> PlotMenuState:
    return PlotMenuState(
        groups={
            "image": PlotMenuGroupState(
                menu="image",
                specs=(
                    _spec("image.first", label="First", default=True),
                    _spec("image.second", label="Second"),
                ),
                selected_key="image.second",
            ),
            "line": PlotMenuGroupState(
                menu="line",
                specs=(_spec("line.depth", menu="line", renderer="line"),),
                selected_key="line.depth",
            ),
            "probe": PlotMenuGroupState(
                menu="probe",
                specs=(
                    _spec("probe.depth", menu="probe", renderer="probe", default=True),
                ),
                selected_key="probe.depth",
            ),
        }
    )


def _presenter(calls: list[Any]):
    queries = FakeQueries()
    commands = FakeCommands()
    app = SimpleNamespace(queries=queries, commands=commands)
    presenter = DesktopEphysPlotPresenter(
        app=app,
        callbacks=EphysPlotRenderCallbacks(
            raw_image_payloads=lambda: {"raw": "payload"},
            render_image=lambda data: calls.append(("image", data)),
            render_scatter=lambda data: calls.append(("scatter", data)),
            render_line=lambda data: calls.append(("line", data)),
            render_probe=lambda data, bounds: calls.append(("probe", data, bounds)),
        ),
        action_factory=FakeAction,
        action_group_factory=FakeActionGroup,
    )
    presenter.attach_plot_menus(FakeMenuBar())
    presenter.attach_unit_filter_menu(FakeMenuBar(), parent=object())
    presenter.render_menus(_plot_menu_state())
    return presenter, queries, commands


def test_ephys_plot_presenter_renders_menu_state_and_selected_keys() -> None:
    presenter, _queries, _commands = _presenter([])

    assert presenter.has_plot_menus()
    assert presenter.current_plot_keys() == {
        "image": "image.second",
        "line": "line.depth",
        "probe": "probe.depth",
    }


def test_ephys_plot_presenter_toggles_and_dispatches_selected_plot() -> None:
    calls: list[Any] = []
    presenter, queries, _commands = _presenter(calls)

    presenter.toggle_plot("image")

    assert presenter.current_plot_keys()["image"] == "image.first"
    assert calls == [("image", {"payload": "image.first"})]
    assert queries.payload_calls[-1] == ("image.first", {"raw": "payload"})


def test_ephys_plot_presenter_dispatches_probe_plot_with_bounds() -> None:
    calls: list[Any] = []
    presenter, queries, _commands = _presenter(calls)

    presenter.plot_from_spec("probe.depth")

    assert calls == [("probe", {"payload": "probe.depth"}, [1, 2])]
    assert queries.bounds_calls == [("probe.depth", {"raw": "payload"})]


def test_ephys_plot_presenter_applies_unit_filter_and_redraws_current() -> None:
    calls: list[Any] = []
    presenter, _queries, commands = _presenter(calls)

    presenter.filter_unit_pressed("KS good")

    assert commands.unit_filters == ["KS good"]
    assert calls == [
        ("image", {"payload": "image.second"}),
        ("line", {"payload": "line.depth"}),
        ("probe", {"payload": "probe.depth"}, [1, 2]),
    ]


def test_ephys_plot_presenter_renders_defaults_for_new_shank() -> None:
    calls: list[Any] = []
    presenter, _queries, _commands = _presenter(calls)
    state = ActiveShankScreenState(
        shank_idx=0,
        shank_id=1,
        alignment_key=None,
        data_loaded=True,
        preserve_plot_selection=False,
        unit_filter="all",
        plot_menu=_plot_menu_state(),
        slice_menu=None,
    )

    presenter.render_shank_ephys_plots(state)

    assert calls == [
        ("image", {"payload": "image.first"}),
        ("probe", {"payload": "probe.depth"}, [1, 2]),
        ("line", {"payload": "line.depth"}),
    ]
