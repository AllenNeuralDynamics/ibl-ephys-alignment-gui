"""Tests for desktop shank presentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ephys_alignment_gui.application.results import LoadedShankPrepared
from ephys_alignment_gui.application.workflow import Failed
from ephys_alignment_gui.core.alignment_events import ShankChanged
from ephys_alignment_gui.core.alignment_read_models import (
    ActiveShankPlotDataState,
    ActiveShankScreenState,
    PreparedActiveShankScreenState,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.slice_display_policy import SliceSelection
from ephys_alignment_gui.desktop.shank_presenter import (
    DesktopShankPresenter,
    DesktopShankRenderCallbacks,
    DesktopShankSelectionState,
)


def _event(
    *,
    shank_idx: int = 1,
    data_loaded: bool = True,
    preserve_plot_selection: bool | None = None,
) -> ShankChanged:
    return ShankChanged(
        source="test",
        previous_shank_idx=0,
        shank_idx=shank_idx,
        previous_key=AlignmentKey("rec", "stream", 0),
        active_key=AlignmentKey("rec", "stream", shank_idx),
        data_loaded=data_loaded,
        preserve_plot_selection=preserve_plot_selection,
    )


class FakeCommands:
    def __init__(
        self,
        calls: list[Any],
        *,
        prepared: Any | None = None,
    ) -> None:
        self.calls = calls
        self.prepared = prepared or LoadedShankPrepared(
            shank_idx=1,
            n_channels=384,
            histology_available=True,
            alignment_choices=["original"],
        )
        self.loaded_shank = self
        self.edit = self
        self.display = self

    def set_unit_filter(self, unit_filter: str) -> None:
        self.calls.append(("set_unit_filter", unit_filter))

    def prepare_loaded_shank(self, shank_idx: int):
        self.calls.append(("prepare_shank", shank_idx))
        return self.prepared

    def set_probe_limits(self, min_um: float, max_um: float) -> None:
        self.calls.append(("set_probe_limits", min_um, max_um))


class FakeQueries:
    def __init__(
        self,
        calls: list[Any],
        *,
        resolved_preserve: bool = True,
        slice_ready: bool = True,
        plot_ready: bool = True,
    ) -> None:
        self.calls = calls
        self.resolved_preserve = resolved_preserve
        self.slice_ready = slice_ready
        self.plot_ready = plot_ready
        self.plot_data_state = ActiveShankPlotDataState(
            key=AlignmentKey("rec", "stream", 1),
            shank_idx=1,
            unit_filter="all",
            channel_min_um=5.0,
            channel_max_um=100.0,
            in_brain_depths_um=None,
        )
        self.screen_state = ActiveShankScreenState(
            shank_idx=1,
            shank_id=2,
            alignment_key=AlignmentKey("rec", "stream", 1),
            data_loaded=True,
            preserve_plot_selection=resolved_preserve,
            unit_filter="all",
            plot_menu="plot-menu",
            slice_menu="slice-menu",
        )
        self.workspace = SimpleNamespace(
            resolve_shank_preserve_plot_selection=(
                self.resolve_shank_preserve_plot_selection
            ),
        )
        self.active_shank = SimpleNamespace(
            prepare_active_shank_screen_state=self.prepare_active_shank_screen_state,
        )

    def resolve_shank_preserve_plot_selection(self, preserve_plot_selection):
        self.calls.append(("resolve", preserve_plot_selection))
        return self.resolved_preserve

    def prepare_active_shank_screen_state(self, **kwargs):
        self.calls.append(("prepared_screen", kwargs))
        plot_data = self.plot_data_state if self.plot_ready else None
        screen = (
            self.screen_state
            if plot_data is not None
            and (not kwargs["histology_available"] or self.slice_ready)
            else None
        )
        return PreparedActiveShankScreenState(
            plot_data=plot_data,
            screen=screen,
            histology_available=kwargs["histology_available"],
            slice_data_available=self.slice_ready,
        )


def _app(
    calls: list[Any],
    *,
    resolved_preserve: bool = True,
    slice_ready: bool = True,
    plot_ready: bool = True,
    prepared: Any | None = None,
):
    events = EventBus()
    return SimpleNamespace(
        commands=FakeCommands(calls, prepared=prepared),
        queries=FakeQueries(
            calls,
            resolved_preserve=resolved_preserve,
            slice_ready=slice_ready,
            plot_ready=plot_ready,
        ),
        events=events,
    )


def _callbacks(calls: list[Any]) -> DesktopShankRenderCallbacks:
    selection = SliceSelection("slice_data", "ccf")
    selections = DesktopShankSelectionState(
        previous_slice_selection=selection,
        previous_slice_label="CCF",
        previous_ephys_plot_keys={
            "image": "image.rms_ap",
            "line": "line.spikes",
            "probe": "probe.rms_ap",
        },
    )
    return DesktopShankRenderCallbacks(
        capture_plot_selection=lambda preserve: (
            calls.append(("capture", preserve)) or selections
        ),
        clear_reference_lines=lambda: calls.append("clear_lines"),
        render_alignment_choices=lambda choices: calls.append(
            ("alignment_choices", choices)
        ),
        apply_plot_data_state=lambda state: calls.append(("apply_plot_data", state)),
        raw_image_payloads=lambda: calls.append("raw_payloads") or {"raw": "payload"},
        render_plot_menus=lambda state: calls.append(("menus", state)),
        render_ephys_plots=lambda state: calls.append(("ephys", state)),
        render_histology_plots=lambda idx: calls.append(("render_histology", idx)),
        restore_slice_selection=lambda menu, selection, label: calls.append(
            ("slice_selection", menu, selection, label)
        ),
        configure_view=lambda preserve: calls.append(("configure", preserve)),
        offline=lambda: True,
    )


def test_shank_presenter_coordinates_loaded_shank_rendering() -> None:
    calls: list[Any] = []
    app = _app(calls, resolved_preserve=True)
    presenter = DesktopShankPresenter(
        app,
        callbacks=_callbacks(calls),
    )
    subscriptions = presenter.connect_shank_events()

    app.events.emit(_event(shank_idx=2, preserve_plot_selection=None))

    assert len(subscriptions) == 1
    assert all(subscription.active for subscription in subscriptions)
    assert calls == [
        ("resolve", None),
        ("capture", True),
        "clear_lines",
        ("prepare_shank", 2),
        ("alignment_choices", ["original"]),
        "raw_payloads",
        (
            "prepared_screen",
            {
                "histology_available": True,
                "preserve_plot_selection": True,
                "previous_ephys_plot_keys": {
                    "image": "image.rms_ap",
                    "line": "line.spikes",
                    "probe": "probe.rms_ap",
                },
                "raw_image_payloads": {"raw": "payload"},
                "previous_slice_selection": SliceSelection("slice_data", "ccf"),
                "offline": True,
            },
        ),
        ("set_probe_limits", 0.0, 100.0),
        ("apply_plot_data", app.queries.plot_data_state),
        ("menus", "plot-menu"),
        ("ephys", app.queries.screen_state),
        (
            "slice_selection",
            "slice-menu",
            SliceSelection("slice_data", "ccf"),
            "CCF",
        ),
        ("render_histology", 2),
        ("configure", True),
    ]


def test_shank_presenter_does_not_render_before_data_is_loaded() -> None:
    calls: list[Any] = []
    app = _app(calls)
    presenter = DesktopShankPresenter(app, callbacks=_callbacks(calls))
    presenter.connect_shank_events()

    app.events.emit(_event(shank_idx=3, data_loaded=False))

    assert calls == []


def test_shank_presenter_stops_when_shank_preparation_fails() -> None:
    calls: list[Any] = []
    app = _app(calls, resolved_preserve=False, prepared=Failed("not ready"))
    presenter = DesktopShankPresenter(
        app,
        callbacks=_callbacks(calls),
    )

    presenter.render_loaded_shank(shank_idx=1, preserve_plot_selection=False)

    assert calls == [
        ("resolve", False),
        ("capture", False),
        "clear_lines",
        ("prepare_shank", 1),
    ]


def test_shank_presenter_raises_when_required_slice_preparation_fails() -> None:
    calls: list[Any] = []
    app = _app(calls, slice_ready=False)
    presenter = DesktopShankPresenter(
        app,
        callbacks=_callbacks(calls),
    )

    with pytest.raises(RuntimeError, match="Could not build active slice data"):
        presenter.render_loaded_shank(shank_idx=1, preserve_plot_selection=True)

    assert calls == [
        ("resolve", True),
        ("capture", True),
        "clear_lines",
        ("prepare_shank", 1),
        ("alignment_choices", ["original"]),
        "raw_payloads",
        (
            "prepared_screen",
            {
                "histology_available": True,
                "preserve_plot_selection": True,
                "previous_ephys_plot_keys": {
                    "image": "image.rms_ap",
                    "line": "line.spikes",
                    "probe": "probe.rms_ap",
                },
                "raw_image_payloads": {"raw": "payload"},
                "previous_slice_selection": SliceSelection("slice_data", "ccf"),
                "offline": True,
            },
        ),
        ("set_probe_limits", 0.0, 100.0),
        ("apply_plot_data", app.queries.plot_data_state),
    ]


def test_shank_presenter_allows_missing_slice_data_without_histology() -> None:
    calls: list[Any] = []
    app = _app(
        calls,
        resolved_preserve=False,
        slice_ready=False,
        prepared=LoadedShankPrepared(
            shank_idx=1,
            n_channels=384,
            histology_available=False,
            alignment_choices=None,
        ),
    )
    presenter = DesktopShankPresenter(
        app,
        callbacks=_callbacks(calls),
    )

    presenter.render_loaded_shank(shank_idx=1, preserve_plot_selection=False)

    assert ("set_unit_filter", "all") in calls
    assert ("menus", "plot-menu") in calls
    assert ("configure", False) in calls
