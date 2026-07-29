"""Tests for desktop alignment screen view helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.desktop_alignment_screen_view import (
    DesktopAlignmentScreenView,
)


class FakeDepthPlots:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def capture_y_ranges(self) -> dict[str, tuple[float, float]]:
        self.calls.append("capture")
        return {"image": (1.0, 2.0)}

    def restore_y_ranges(self, ranges: dict[str, tuple[float, float]]) -> None:
        self.calls.append(("restore", ranges))

    def set_default_feature_y_range(self) -> None:
        self.calls.append("default-range")


class FakeCheckbox:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def blockSignals(self, blocked: bool) -> None:
        self.calls.append(("block", blocked))

    def setChecked(self, checked: bool) -> None:
        self.calls.append(("checked", checked))


class FakeReferenceLines:
    def __init__(self) -> None:
        self.previous_features: list[Any] = []

    def create_previous_feature_lines(self, feature_prev: Any) -> None:
        self.previous_features.append(feature_prev)


class FakeLabel:
    def __init__(self) -> None:
        self.text: str | None = None

    def setText(self, text: str) -> None:
        self.text = text


def _view(
    *,
    active_state: Any = None,
) -> tuple[
    DesktopAlignmentScreenView,
    FakeDepthPlots,
    FakeCheckbox,
    FakeReferenceLines,
    FakeLabel,
    FakeLabel,
]:
    depth_plots = FakeDepthPlots()
    checkbox = FakeCheckbox()
    reference_lines = FakeReferenceLines()
    current_label = FakeLabel()
    total_label = FakeLabel()
    display_state = SimpleNamespace(
        edit_settings=SimpleNamespace(
            lin_fit=True,
            set_lin_fit=lambda enabled: setattr(
                display_state.edit_settings, "lin_fit", bool(enabled)
            ),
        )
    )
    view = DesktopAlignmentScreenView(
        depth_plots=depth_plots,
        display_state=display_state,
        reference_lines=reference_lines,
        active_alignment_state=lambda: active_state,
        lin_fit_checkbox=checkbox,
        current_index_label=current_label,
        total_index_label=total_label,
    )
    return view, depth_plots, checkbox, reference_lines, current_label, total_label


def test_restore_lin_fit_updates_display_state_and_checkbox() -> None:
    view, _depth, checkbox, _lines, _current, _total = _view()

    view.restore_lin_fit_from_edit(False)
    view.restore_lin_fit_from_edit(None)

    assert view.display_state.edit_settings.lin_fit is False
    assert checkbox.calls == [
        ("block", True),
        ("checked", False),
        ("block", False),
    ]


def test_depth_range_methods_delegate_to_depth_plot_view() -> None:
    view, depth_plots, _checkbox, _lines, _current, _total = _view()

    assert view.capture_depth_plot_y_ranges() == {"image": (1.0, 2.0)}
    view.restore_depth_plot_y_ranges({"image": (3.0, 4.0)})
    view.set_default_feature_y_range()

    assert depth_plots.calls == [
        "capture",
        ("restore", {"image": (3.0, 4.0)}),
        "default-range",
    ]


def test_create_previous_reference_lines_uses_middle_feature_points_in_um() -> None:
    active_state = SimpleNamespace(feature_prev=np.array([0.0, 0.001, 0.002, 0.003]))
    view, _depth, _checkbox, reference_lines, _current, _total = _view(
        active_state=active_state
    )

    view.create_reference_lines_for_previous_alignment()

    assert len(reference_lines.previous_features) == 1
    np.testing.assert_allclose(reference_lines.previous_features[0], [1000.0, 2000.0])


def test_update_status_uses_active_edit_history() -> None:
    active_state = SimpleNamespace(
        edit_history=SimpleNamespace(current_idx=2, total_idx=5)
    )
    view, _depth, _checkbox, _lines, current_label, total_label = _view(
        active_state=active_state
    )

    view.update_status()

    assert current_label.text == "Current Index = 2"
    assert total_label.text == "Total Index = 5"
