"""Tests for desktop shank presentation."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.alignment_events import ShankChanged
from ephys_alignment_gui.desktop_shank_presenter import (
    DesktopShankPresenter,
    DesktopShankRenderCallbacks,
    DesktopShankSelectionState,
)
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.event_bus import EventBus
from ephys_alignment_gui.slice_display_policy import SliceSelection


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


def _callbacks(
    calls: list[Any],
    *,
    resolved_preserve: bool = True,
    histology_ready: bool = True,
    slice_ready: bool = True,
) -> DesktopShankRenderCallbacks:
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
        resolve_preserve_plot_selection=lambda preserve: (
            calls.append(("resolve", preserve)) or resolved_preserve
        ),
        capture_plot_selection=lambda preserve: (
            calls.append(("capture", preserve)) or selections
        ),
        clear_reference_lines=lambda: calls.append("clear_lines"),
        prepare_runtime=lambda idx: calls.append(("runtime", idx)),
        prepare_histology=lambda idx: (
            calls.append(("histology", idx)) or histology_ready
        ),
        prepare_plot_data=lambda idx, preserve: calls.append(
            ("plot_data", idx, preserve)
        ),
        prepare_slice_data=lambda: calls.append("slice_data") or slice_ready,
        refresh_plot_menus=lambda preserve, keys: calls.append(
            ("menus", preserve, keys)
        ),
        render_ephys_plots=lambda preserve: calls.append(("ephys", preserve)),
        render_histology_plots=lambda idx: calls.append(("render_histology", idx)),
        restore_slice_selection=lambda selection, label: calls.append(
            ("slice_selection", selection, label)
        ),
        configure_view=lambda preserve: calls.append(("configure", preserve)),
    )


def test_shank_presenter_coordinates_loaded_shank_rendering() -> None:
    events = EventBus()
    calls: list[Any] = []
    presenter = DesktopShankPresenter(
        events,
        callbacks=_callbacks(calls, resolved_preserve=True),
    )
    subscriptions = presenter.connect_shank_events()

    events.emit(_event(shank_idx=2, preserve_plot_selection=None))

    assert len(subscriptions) == 1
    assert all(subscription.active for subscription in subscriptions)
    assert calls == [
        ("resolve", None),
        ("capture", True),
        "clear_lines",
        ("runtime", 2),
        ("histology", 2),
        ("plot_data", 2, True),
        "slice_data",
        (
            "menus",
            True,
            {
                "image": "image.rms_ap",
                "line": "line.spikes",
                "probe": "probe.rms_ap",
            },
        ),
        ("ephys", True),
        ("render_histology", 2),
        ("slice_selection", SliceSelection("slice_data", "ccf"), "CCF"),
        ("configure", True),
    ]


def test_shank_presenter_does_not_render_before_data_is_loaded() -> None:
    events = EventBus()
    calls: list[Any] = []
    presenter = DesktopShankPresenter(events, callbacks=_callbacks(calls))
    presenter.connect_shank_events()

    events.emit(_event(shank_idx=3, data_loaded=False))

    assert calls == []


def test_shank_presenter_stops_when_histology_preparation_fails() -> None:
    calls: list[Any] = []
    presenter = DesktopShankPresenter(
        EventBus(),
        callbacks=_callbacks(calls, resolved_preserve=False, histology_ready=False),
    )

    presenter.render_loaded_shank(shank_idx=1, preserve_plot_selection=False)

    assert calls == [
        ("resolve", False),
        ("capture", False),
        "clear_lines",
        ("runtime", 1),
        ("histology", 1),
    ]


def test_shank_presenter_stops_when_slice_preparation_fails() -> None:
    calls: list[Any] = []
    presenter = DesktopShankPresenter(
        EventBus(),
        callbacks=_callbacks(calls, slice_ready=False),
    )

    presenter.render_loaded_shank(shank_idx=1, preserve_plot_selection=True)

    assert calls == [
        ("resolve", True),
        ("capture", True),
        "clear_lines",
        ("runtime", 1),
        ("histology", 1),
        ("plot_data", 1, True),
        "slice_data",
    ]
