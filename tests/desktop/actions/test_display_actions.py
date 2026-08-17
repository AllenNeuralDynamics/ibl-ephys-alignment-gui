"""Tests for desktop display action/event coordination."""

from __future__ import annotations

from types import SimpleNamespace

from ephys_alignment_gui.application.commands.display import DisplayCommandHandler
from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.desktop.actions.display_actions import DesktopDisplayActions


class FakeReferenceLines:
    def __init__(self) -> None:
        self.add_count = 0
        self.remove_count = 0
        self.reattach_count = 0
        self.delete_count = 0

    def add_to_plots(self) -> None:
        self.add_count += 1

    def remove_from_plots(self) -> None:
        self.remove_count += 1

    def reattach(self) -> None:
        self.reattach_count += 1

    def delete_selected(self) -> None:
        self.delete_count += 1


class FakeHistologyDisplay:
    def __init__(self) -> None:
        self.toggle_count = 0
        self.sync_calls: list[str] = []

    def toggle_labels(self) -> None:
        self.toggle_count += 1

    def sync_top_to_tip(self) -> None:
        self.sync_calls.append("top-to-tip")

    def sync_tip_to_top(self) -> None:
        self.sync_calls.append("tip-to-top")


class FakeHistologyPresenter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def render_active_nearby(self) -> bool:
        self.calls.append("nearby")
        return True

    def render_active_reference(self) -> bool:
        self.calls.append("reference")
        return True

    def render_active_aligned(self) -> bool:
        self.calls.append("aligned")
        return True

    def render_active_scale_factor(self) -> bool:
        self.calls.append("scale")
        return True


class FakeSlicePanelPresenter:
    def __init__(self) -> None:
        self.toggle_count = 0

    def toggle_channel_visibility(self) -> None:
        self.toggle_count += 1


def test_reference_line_visibility_updates_from_display_event() -> None:
    state = AlignmentDisplayState()
    reference_lines = FakeReferenceLines()
    actions = _actions(state, reference_lines=reference_lines)
    actions.connect_display_events()

    actions.toggle_reference_lines()
    actions.toggle_reference_lines()

    assert state.reference_lines_visible is True
    assert reference_lines.remove_count == 1
    assert reference_lines.add_count == 1


def test_histology_boundary_toggle_renders_from_display_event() -> None:
    state = AlignmentDisplayState()
    histology_presenter = FakeHistologyPresenter()
    actions = _actions(state, histology_presenter=histology_presenter)
    actions.connect_display_events()

    assert actions.toggle_histology_boundaries()
    assert actions.toggle_histology_boundaries()

    assert state.histology_boundaries_visible is True
    assert histology_presenter.calls == ["nearby", "reference"]


def test_region_annotation_source_refreshes_histology_from_display_event() -> None:
    state = AlignmentDisplayState()
    reference_lines = FakeReferenceLines()
    histology_presenter = FakeHistologyPresenter()
    actions = _actions(
        state,
        reference_lines=reference_lines,
        histology_presenter=histology_presenter,
    )
    actions.connect_display_events()

    actions.toggle_region_annotation_source()

    assert state.region_annotation_source == "FranklinPaxinos"
    assert histology_presenter.calls == ["aligned", "reference", "scale"]
    assert reference_lines.reattach_count == 1


def test_region_annotation_source_refresh_preserves_hidden_reference_lines() -> None:
    state = AlignmentDisplayState(reference_lines_visible=False)
    reference_lines = FakeReferenceLines()
    actions = _actions(state, reference_lines=reference_lines)
    actions.connect_display_events()

    actions.toggle_region_annotation_source()

    assert reference_lines.reattach_count == 0
    assert reference_lines.remove_count == 1


def test_desktop_only_toggles_remain_local_display_actions() -> None:
    histology_display = FakeHistologyDisplay()
    slice_panel = FakeSlicePanelPresenter()
    reference_lines = FakeReferenceLines()
    actions = _actions(
        AlignmentDisplayState(),
        histology_display=histology_display,
        slice_panel_presenter=slice_panel,
        reference_lines=reference_lines,
    )

    actions.toggle_labels()
    actions.toggle_channels()
    actions.delete_selected_reference_line()
    actions.sync_histology_top_to_tip()
    actions.sync_histology_tip_to_top()

    assert histology_display.toggle_count == 1
    assert slice_panel.toggle_count == 1
    assert reference_lines.delete_count == 1
    assert histology_display.sync_calls == ["top-to-tip", "tip-to-top"]


def _actions(
    state: AlignmentDisplayState,
    *,
    reference_lines: FakeReferenceLines | None = None,
    histology_display: FakeHistologyDisplay | None = None,
    histology_presenter: FakeHistologyPresenter | None = None,
    slice_panel_presenter: FakeSlicePanelPresenter | None = None,
) -> DesktopDisplayActions:
    events = EventBus()
    app = SimpleNamespace(
        commands=SimpleNamespace(display=DisplayCommandHandler(state, events)),
        events=events,
        queries=SimpleNamespace(
            workspace=SimpleNamespace(
                reference_lines_visible=lambda: state.reference_lines_visible,
            )
        ),
    )
    return DesktopDisplayActions(
        app=app,
        displays=SimpleNamespace(
            reference_lines=reference_lines or FakeReferenceLines(),
            histology=histology_display or FakeHistologyDisplay(),
            ephys=SimpleNamespace(reset_feature_image_x_range=lambda: None),
        ),
        histology_presenter=histology_presenter or FakeHistologyPresenter(),
        slice_panel_presenter=slice_panel_presenter or FakeSlicePanelPresenter(),
        alignment_screen=SimpleNamespace(set_default_feature_y_range=lambda **_: None),
        fit_alignment=lambda: True,
        histology_available=lambda: True,
    )
