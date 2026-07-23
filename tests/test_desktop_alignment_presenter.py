"""Tests for desktop alignment presentation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_events import AlignmentEdited, AlignmentEditKind
from ephys_alignment_gui.alignment_read_models import ActiveAlignmentRenderState
from ephys_alignment_gui.desktop_alignment_presenter import (
    DesktopAlignmentPresenter,
    DesktopAlignmentRenderCallbacks,
    desktop_presentation_options_for_edit,
)
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.event_bus import EventBus


class FakeQueries:
    def __init__(self, render_state: ActiveAlignmentRenderState | None) -> None:
        self.render_state = render_state
        self.calls: list[str] = []

    def active_alignment_render_state(self) -> ActiveAlignmentRenderState | None:
        self.calls.append("query")
        return self.render_state


def _render_state() -> ActiveAlignmentRenderState:
    active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    return ActiveAlignmentRenderState(
        key=AlignmentKey("rec", "stream", 1),
        active_alignment=active_alignment,
        histology="histology",
        projection="projection",
    )


def _recording_callbacks(calls: list[Any]) -> DesktopAlignmentRenderCallbacks:
    return DesktopAlignmentRenderCallbacks(
        restore_lin_fit=lambda lin_fit: calls.append(("restore_lin_fit", lin_fit)),
        clear_reference_lines=lambda: calls.append("clear_reference_lines"),
        capture_depth_plot_y_ranges=lambda: calls.append("capture_depth") or {"y": 1},
        restore_depth_plot_y_ranges=lambda ranges: calls.append(
            ("restore_depth", ranges)
        ),
        apply_histology_data=lambda histology: calls.append(("histology", histology)),
        apply_channel_projection=lambda projection: calls.append(
            ("projection", projection)
        ),
        reattach_reference_lines=lambda: calls.append("reattach_lines"),
        plot_histology=lambda: calls.append("plot_histology"),
        plot_scale_factor=lambda: calls.append("plot_scale"),
        plot_fit=lambda: calls.append("plot_fit"),
        plot_channels=lambda projection: calls.append(("plot_channels", projection)),
        refresh_perpendicular_histology=lambda: calls.append("refresh_perp"),
        update_reference_lines_to_alignment=lambda: calls.append("update_lines"),
        create_reference_lines_for_previous_alignment=lambda: calls.append(
            "create_previous_lines"
        ),
        set_default_feature_y_range=lambda: calls.append("set_default_range"),
        update_status=lambda: calls.append("update_status"),
    )


def _configured_presenter(
    render_state: ActiveAlignmentRenderState | None,
    calls: list[Any],
) -> tuple[EventBus, FakeQueries, DesktopAlignmentPresenter]:
    events = EventBus()
    queries = FakeQueries(render_state)
    presenter = DesktopAlignmentPresenter(
        events,
        queries=queries,
        callbacks=_recording_callbacks(calls),
    )
    return events, queries, presenter


def _emit_edit(
    events: EventBus,
    render_state: ActiveAlignmentRenderState,
    edit_kind: AlignmentEditKind,
    *,
    lin_fit: bool | None = None,
) -> None:
    events.emit(
        AlignmentEdited(
            edit_kind=edit_kind,
            active_key=render_state.key,
            active_alignment=render_state.active_alignment,
            lin_fit=lin_fit,
        )
    )


def test_desktop_presentation_options_are_derived_from_edit_kind() -> None:
    fit = desktop_presentation_options_for_edit("fit")
    reset = desktop_presentation_options_for_edit("reset")

    assert fit.line_update == "sync_to_alignment"
    assert fit.preserve_depth_range
    assert not fit.clear_reference_lines
    assert reset.line_update == "reset_to_previous"
    assert reset.reset_histology_range
    assert reset.clear_reference_lines


def test_desktop_presenter_coordinates_alignment_edit_rendering() -> None:
    render_state = _render_state()
    calls: list[Any] = []
    events, queries, presenter = _configured_presenter(render_state, calls)
    subscriptions = presenter.connect_alignment_events()

    _emit_edit(events, render_state, "fit", lin_fit=False)

    assert len(subscriptions) == 1
    assert all(subscription.active for subscription in subscriptions)
    assert queries.calls == ["query"]
    assert calls == [
        ("restore_lin_fit", False),
        "capture_depth",
        ("histology", "histology"),
        ("projection", "projection"),
        "plot_histology",
        "plot_scale",
        "plot_fit",
        ("plot_channels", "projection"),
        "refresh_perp",
        "reattach_lines",
        "update_lines",
        "update_status",
        ("restore_depth", {"y": 1}),
    ]


def test_desktop_presenter_coordinates_offset_rendering() -> None:
    render_state = _render_state()
    calls: list[Any] = []
    events, queries, presenter = _configured_presenter(render_state, calls)
    presenter.connect_alignment_events()

    _emit_edit(events, render_state, "offset", lin_fit=True)

    assert queries.calls == ["query"]
    assert calls == [
        ("restore_lin_fit", True),
        ("histology", "histology"),
        ("projection", "projection"),
        "plot_histology",
        "plot_scale",
        "plot_fit",
        ("plot_channels", "projection"),
        "refresh_perp",
        "reattach_lines",
        "update_lines",
        "update_status",
    ]


def test_desktop_presenter_coordinates_previous_and_next_rendering() -> None:
    for edit_kind in ("previous", "next"):
        render_state = _render_state()
        calls: list[Any] = []
        events, queries, presenter = _configured_presenter(render_state, calls)
        presenter.connect_alignment_events()

        _emit_edit(events, render_state, edit_kind)

        assert queries.calls == ["query"]
        assert calls == [
            ("restore_lin_fit", None),
            ("histology", "histology"),
            ("projection", "projection"),
            "reattach_lines",
            "plot_histology",
            "plot_scale",
            "plot_fit",
            ("plot_channels", "projection"),
            "refresh_perp",
            "reattach_lines",
            "update_status",
        ]


def test_desktop_presenter_coordinates_reset_rendering() -> None:
    render_state = _render_state()
    calls: list[Any] = []
    events, queries, presenter = _configured_presenter(render_state, calls)
    presenter.connect_alignment_events()

    _emit_edit(events, render_state, "reset", lin_fit=False)

    assert queries.calls == ["query"]
    assert calls == [
        ("restore_lin_fit", False),
        "clear_reference_lines",
        ("histology", "histology"),
        ("projection", "projection"),
        "plot_histology",
        "plot_scale",
        "plot_fit",
        ("plot_channels", "projection"),
        "refresh_perp",
        "create_previous_lines",
        "set_default_range",
        "update_status",
    ]
