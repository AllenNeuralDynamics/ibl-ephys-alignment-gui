"""Tests for desktop alignment presentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_events import AlignmentEdited, AlignmentEditKind
from ephys_alignment_gui.core.alignment_read_models import (
    ActiveAlignmentRenderState,
    ActiveReferenceLineRenderState,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.desktop.presenters.alignment_presenter import (
    DesktopAlignmentPresenter,
    DesktopAlignmentRenderCallbacks,
    desktop_presentation_options_for_edit,
)


class FakeQueries:
    def __init__(
        self,
        render_state: ActiveAlignmentRenderState | None,
        line_state: ActiveReferenceLineRenderState | None,
    ) -> None:
        self.render_state = render_state
        self.line_state = line_state
        self.calls: list[Any] = []
        self.alignment_render = SimpleNamespace(
            active_alignment_render_state=self.active_alignment_render_state,
        )
        self.workspace = SimpleNamespace(
            active_alignment_reference_line_state=(
                self.active_alignment_reference_line_state
            )
        )

    def active_alignment_render_state(self) -> ActiveAlignmentRenderState | None:
        self.calls.append("query_alignment")
        return self.render_state

    def active_alignment_reference_line_state(
        self,
        shank_idx: int,
    ) -> ActiveReferenceLineRenderState | None:
        self.calls.append(("query_lines", shank_idx))
        return self.line_state


def _render_state() -> ActiveAlignmentRenderState:
    active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    return ActiveAlignmentRenderState(
        key=AlignmentKey("rec", "stream", 1),
        active_alignment=active_alignment,
        histology=SimpleNamespace(
            scale=SimpleNamespace(region="region", scale="scale")
        ),
        projection="projection",
    )


def _line_state() -> ActiveReferenceLineRenderState:
    return ActiveReferenceLineRenderState(
        feature_positions_um=np.array([1000.0]),
        raw_track_positions_um=np.array([1100.0]),
    )


def _recording_callbacks(calls: list[Any]) -> DesktopAlignmentRenderCallbacks:
    return DesktopAlignmentRenderCallbacks(
        restore_lin_fit=lambda lin_fit: calls.append(("restore_lin_fit", lin_fit)),
        clear_reference_lines=lambda: calls.append("clear_reference_lines"),
        capture_depth_plot_y_ranges=lambda: calls.append("capture_depth") or {"y": 1},
        restore_depth_plot_y_ranges=lambda ranges: calls.append(
            ("restore_depth", ranges)
        ),
        render_histology_alignment=lambda state: calls.append(
            ("render_histology_alignment", state)
        ),
        plot_channels=lambda projection: calls.append(("plot_channels", projection)),
        refresh_perpendicular_histology=lambda: calls.append("refresh_perp"),
        render_reference_lines_from_alignment=lambda state: calls.append(
            ("render_lines", state)
        ),
        set_default_feature_y_range=lambda: calls.append("set_default_range"),
        update_status=lambda: calls.append("update_status"),
    )


def _configured_presenter(
    render_state: ActiveAlignmentRenderState | None,
    calls: list[Any],
) -> tuple[EventBus, FakeQueries, DesktopAlignmentPresenter]:
    events = EventBus()
    queries = FakeQueries(render_state, _line_state())
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
    previous = desktop_presentation_options_for_edit("previous")
    reset = desktop_presentation_options_for_edit("reset")

    assert fit.line_update == "render_from_alignment"
    assert fit.preserve_depth_range
    assert not fit.clear_reference_lines
    assert (
        desktop_presentation_options_for_edit("load_previous").line_update
        == "render_from_alignment"
    )
    assert previous.line_update == "render_from_alignment"
    assert reset.line_update == "none"
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
    assert queries.calls == ["query_alignment", ("query_lines", 1)]
    assert calls == [
        ("restore_lin_fit", False),
        "capture_depth",
        ("render_histology_alignment", render_state),
        ("plot_channels", "projection"),
        "refresh_perp",
        ("render_lines", queries.line_state),
        "update_status",
        ("restore_depth", {"y": 1}),
    ]


def test_desktop_presenter_coordinates_offset_rendering() -> None:
    render_state = _render_state()
    calls: list[Any] = []
    events, queries, presenter = _configured_presenter(render_state, calls)
    presenter.connect_alignment_events()

    _emit_edit(events, render_state, "offset", lin_fit=True)

    assert queries.calls == ["query_alignment", ("query_lines", 1)]
    assert calls == [
        ("restore_lin_fit", True),
        ("render_histology_alignment", render_state),
        ("plot_channels", "projection"),
        "refresh_perp",
        ("render_lines", queries.line_state),
        "update_status",
    ]


def test_desktop_presenter_coordinates_previous_and_next_rendering() -> None:
    for edit_kind in ("load_previous", "previous", "next"):
        render_state = _render_state()
        calls: list[Any] = []
        events, queries, presenter = _configured_presenter(render_state, calls)
        presenter.connect_alignment_events()

        _emit_edit(events, render_state, edit_kind)

        assert queries.calls == ["query_alignment", ("query_lines", 1)]
        assert calls == [
            ("restore_lin_fit", None),
            ("render_histology_alignment", render_state),
            ("plot_channels", "projection"),
            "refresh_perp",
            ("render_lines", queries.line_state),
            "update_status",
        ]


def test_desktop_presenter_coordinates_reset_rendering() -> None:
    render_state = _render_state()
    calls: list[Any] = []
    events, queries, presenter = _configured_presenter(render_state, calls)
    presenter.connect_alignment_events()

    _emit_edit(events, render_state, "reset", lin_fit=False)

    assert queries.calls == ["query_alignment"]
    assert calls == [
        ("restore_lin_fit", False),
        "clear_reference_lines",
        ("render_histology_alignment", render_state),
        ("plot_channels", "projection"),
        "refresh_perp",
        "set_default_range",
        "update_status",
    ]
