"""Tests for active-shank desktop screen view helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.core.alignment_read_models import ActiveShankPlotDataState
from ephys_alignment_gui.desktop.displays import DesktopDisplays
from ephys_alignment_gui.desktop.views.shank_screen_view import DesktopShankScreenView


class FakeSliceDisplay:
    def __init__(self) -> None:
        self.selection = SimpleNamespace(selection="slice-selection", label="slice")

    def capture_selection(self) -> Any:
        return self.selection


class FakeEphysDisplay:
    def __init__(self, *, has_menus: bool = True) -> None:
        self.has_menus = has_menus
        self.current_plot_key_calls = 0
        self.rendered_menus: list[Any] = []

    def has_plot_menus(self) -> bool:
        return self.has_menus

    def current_plot_keys(self) -> dict[str, str]:
        self.current_plot_key_calls += 1
        return {"image": "raw_ap"}

    def render_menus(self, state: Any) -> None:
        self.rendered_menus.append(state)


class FakeDepthPlotView:
    def __init__(self, calls: list[Any]) -> None:
        self.calls = calls

    def set_probe_limits(self, low: float, high: float) -> None:
        self.calls.append(("lims", low, high))


def _view(
    *,
    has_menus: bool = True,
) -> tuple[DesktopShankScreenView, FakeEphysDisplay, list[Any], DesktopDisplays]:
    calls: list[Any] = []
    ephys = FakeEphysDisplay(has_menus=has_menus)
    displays = DesktopDisplays(
        ephys=ephys,
        histology=object(),
        reference_lines=object(),
        slice=FakeSliceDisplay(),
    )
    view = DesktopShankScreenView(
        depth_plots=FakeDepthPlotView(calls),
        init_menubar=lambda: calls.append("init-menubar"),
        set_view=lambda **kwargs: calls.append(("view", kwargs)),
    )
    view.raw_image_payloads = {"raw": "payload"}
    return view, ephys, calls, displays


def test_capture_plot_selection_preserves_ephys_keys_only_when_requested() -> None:
    preserve_view, preserve_ephys, _calls, preserve_displays = _view(has_menus=True)
    no_preserve_view, no_preserve_ephys, _calls, no_preserve_displays = _view(
        has_menus=True
    )

    preserved = preserve_view.capture_plot_selection(True, displays=preserve_displays)
    not_preserved = no_preserve_view.capture_plot_selection(
        False,
        displays=no_preserve_displays,
    )

    assert preserved.previous_slice_selection == "slice-selection"
    assert preserved.previous_slice_label == "slice"
    assert preserved.previous_ephys_plot_keys == {"image": "raw_ap"}
    assert preserve_ephys.current_plot_key_calls == 1
    assert not_preserved.previous_slice_selection == "slice-selection"
    assert not_preserved.previous_ephys_plot_keys is None
    assert no_preserve_ephys.current_plot_key_calls == 0


def test_apply_plot_data_state_sets_depth_limits_and_clears_raw_payloads() -> None:
    view, _ephys, calls, _displays = _view()
    state = ActiveShankPlotDataState(
        key=None,
        shank_idx=0,
        unit_filter="all",
        channel_min_um=25.0,
        channel_max_um=400.0,
        in_brain_depths_um=np.array([10.0, 20.0]),
    )

    view.apply_plot_data_state(state)

    assert calls == [("lims", 0.0, 400.0)]
    assert view.raw_image_payload_mapping() == {}


def test_render_plot_menus_initializes_menubar_when_needed() -> None:
    view, ephys, calls, displays = _view(has_menus=False)

    view.render_plot_menus("menu-state", displays=displays)

    assert calls == ["init-menubar"]
    assert ephys.rendered_menus == ["menu-state"]


def test_configure_view_after_render_only_configures_first_unpreserved_view() -> None:
    view, _ephys, calls, _displays = _view()

    view.configure_view_after_render(False)
    view.configure_view_after_render(False)
    view.configure_view_after_render(True)

    assert calls == [
        ("view", {"view": 1, "configure": True}),
        ("view", {"view": 1, "configure": False}),
        ("view", {"view": 1, "configure": False}),
    ]
