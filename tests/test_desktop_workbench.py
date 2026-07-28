"""Tests for desktop workbench presenter composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop_alignment_presenter import (
    DesktopAlignmentRenderCallbacks,
)
from ephys_alignment_gui.desktop_histology_presenter import (
    DesktopHistologyPresenter,
    DesktopHistologyRenderCallbacks,
)
from ephys_alignment_gui.desktop_shank_presenter import (
    DesktopShankRenderCallbacks,
    DesktopShankSelectionState,
)
from ephys_alignment_gui.desktop_workbench import DesktopWorkbench
from ephys_alignment_gui.event_bus import EventBus


class FakeSubscription:
    def __init__(self) -> None:
        self.disconnect_count = 0

    def disconnect(self) -> None:
        self.disconnect_count += 1


class FakeAlignmentPresenter:
    def __init__(self, subscriptions: list[FakeSubscription]) -> None:
        self.subscriptions = subscriptions
        self.connect_count = 0

    def connect_alignment_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions


class FakeShankPresenter:
    def __init__(self, subscriptions: list[FakeSubscription]) -> None:
        self.subscriptions = subscriptions
        self.connect_count = 0
        self.render_calls: list[tuple[int, bool | None]] = []

    def connect_shank_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions

    def render_loaded_shank(
        self,
        *,
        shank_idx: int,
        preserve_plot_selection: bool | None = None,
    ) -> None:
        self.render_calls.append((shank_idx, preserve_plot_selection))


class FakeHistologyPresenter:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def render_active_aligned(
        self,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> bool:
        self.calls.append(("aligned", fig, movable))
        return True

    def render_active_reference(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        self.calls.append(("reference", fig, movable))
        return True

    def render_active_scale_factor(self) -> bool:
        self.calls.append("scale")
        return True

    def render_active_fit(self) -> bool:
        self.calls.append("fit")
        return True

    def render_active_panels(self) -> bool:
        self.calls.append("panels")
        return True


def _workbench(
    alignment: Any,
    shank: Any,
    histology: Any,
) -> DesktopWorkbench:
    return DesktopWorkbench(
        app=object(),
        alignment_presenter=alignment,
        shank_presenter=shank,
        histology_presenter=histology,
    )


def test_workbench_owns_event_subscription_lifecycle() -> None:
    alignment_sub = FakeSubscription()
    shank_sub = FakeSubscription()
    alignment = FakeAlignmentPresenter([alignment_sub])
    shank = FakeShankPresenter([shank_sub])
    workbench = _workbench(alignment, shank, FakeHistologyPresenter())

    subscriptions = workbench.connect_events()
    second_connect = workbench.connect_events()

    assert subscriptions == [alignment_sub, shank_sub]
    assert second_connect == subscriptions
    assert alignment.connect_count == 1
    assert shank.connect_count == 1

    workbench.disconnect_events()
    workbench.disconnect_events()

    assert alignment_sub.disconnect_count == 1
    assert shank_sub.disconnect_count == 1


def test_workbench_delegates_focused_presenter_entry_points() -> None:
    shank = FakeShankPresenter([])
    histology = FakeHistologyPresenter()
    workbench = _workbench(FakeAlignmentPresenter([]), shank, histology)

    workbench.render_loaded_shank(shank_idx=2, preserve_plot_selection=True)
    workbench.render_active_aligned_histology("fig", movable=False)
    workbench.render_active_reference_histology("ref", movable=True)
    workbench.render_active_scale_factor()
    workbench.render_active_fit()
    workbench.render_active_histology_panels()

    assert shank.render_calls == [(2, True)]
    assert histology.calls == [
        ("aligned", "fig", False),
        ("reference", "ref", True),
        "scale",
        "fit",
        "panels",
    ]


def _alignment_callbacks(histology: DesktopHistologyPresenter):
    return DesktopAlignmentRenderCallbacks(
        restore_lin_fit=lambda _lin_fit: None,
        clear_reference_lines=lambda: None,
        capture_depth_plot_y_ranges=lambda: None,
        restore_depth_plot_y_ranges=lambda _ranges: None,
        reattach_reference_lines=lambda: None,
        render_histology_alignment=histology.render_alignment_edit,
        plot_channels=lambda _projection: None,
        refresh_perpendicular_histology=lambda: None,
        update_reference_lines_to_alignment=lambda: None,
        create_reference_lines_for_previous_alignment=lambda: None,
        set_default_feature_y_range=lambda: None,
        update_status=lambda: None,
    )


def _shank_callbacks() -> DesktopShankRenderCallbacks:
    return DesktopShankRenderCallbacks(
        capture_plot_selection=lambda _preserve: DesktopShankSelectionState(),
        clear_reference_lines=lambda: None,
        prepare_runtime=lambda _shank_idx: None,
        prepare_histology=lambda _shank_idx: True,
        apply_plot_data_state=lambda _state: None,
        raw_image_payloads=dict,
        render_plot_menus=lambda _state: None,
        render_ephys_plots=lambda _state: None,
        render_histology_plots=lambda _shank_idx: None,
        restore_slice_selection=lambda _state, _selection, _label: None,
        configure_view=lambda _preserve: None,
        histology_available=lambda: True,
        offline=lambda: True,
    )


def test_workbench_factory_configures_focused_presenters() -> None:
    callbacks_seen: list[DesktopHistologyPresenter] = []

    def alignment_callbacks_factory(
        histology: DesktopHistologyPresenter,
    ) -> DesktopAlignmentRenderCallbacks:
        callbacks_seen.append(histology)
        return _alignment_callbacks(histology)

    histology_callbacks = DesktopHistologyRenderCallbacks(
        probe_extent_query_kwargs=dict,
        fit_depth_um=lambda: [],
        lin_fit_enabled=lambda: False,
        scale_factor_y_range=lambda: (0.0, 1.0),
    )
    app = SimpleNamespace(events=EventBus(), queries=object())
    panel = object()

    workbench = DesktopWorkbench.create(
        app=app,
        histology_panel=panel,
        histology_callbacks=histology_callbacks,
        alignment_callbacks_factory=alignment_callbacks_factory,
        shank_callbacks=_shank_callbacks(),
    )

    assert isinstance(workbench.histology_presenter, DesktopHistologyPresenter)
    assert workbench.histology_presenter.panel is panel
    assert callbacks_seen == [workbench.histology_presenter]
    assert workbench.alignment_presenter.callbacks is not None
    assert workbench.shank_presenter.callbacks is not None
