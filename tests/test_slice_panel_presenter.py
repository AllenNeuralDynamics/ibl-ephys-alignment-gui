"""Tests for the desktop slice panel presenter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.alignment_read_models import ActiveSliceDataState
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.slice_display_policy import SliceSelection
from ephys_alignment_gui.slice_panel_presenter import (
    SlicePanelPlots,
    SlicePanelPresenter,
    SlicePanelStyle,
)


class FakeAction:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def data(self) -> Any:
        return self._payload


class FakeActionGroup:
    def __init__(self, actions: list[FakeAction], checked: FakeAction | None) -> None:
        self._actions = actions
        self._checked = checked

    def checkedAction(self) -> FakeAction | None:
        return self._checked

    def actions(self) -> list[FakeAction]:
        return self._actions


class FakeQueries:
    def __init__(self, slice_state: ActiveSliceDataState | None = None) -> None:
        self.slice_state = slice_state
        self.rendered_selections: list[SliceSelection] = []

    def active_slice_render_state(self, selection: SliceSelection) -> Any:
        self.rendered_selections.append(selection)
        return SimpleNamespace(scalar_channel=selection.key)

    def active_slice_data_state(self) -> ActiveSliceDataState | None:
        return self.slice_state


def _presenter(
    queries: FakeQueries,
    action_group: FakeActionGroup | None = None,
) -> SlicePanelPresenter:
    return SlicePanelPresenter(
        app=SimpleNamespace(queries=queries),
        plots=SlicePanelPlots(
            coronal=None,
            coronal_layout=None,
            histogram_alt=None,
            perpendicular=None,
        ),
        style=SlicePanelStyle(
            dotted_pen=None,
            solid_pen=None,
            reference_line_pen=None,
        ),
        session_provider=lambda: SimpleNamespace(),
        histology_exists=lambda: True,
        action_group_provider=lambda: action_group,
    )


def test_slice_panel_reads_current_selection_from_action_group() -> None:
    selection = SliceSelection("slice_data", "histology_registration")
    checked = FakeAction(selection.to_payload())
    other = FakeAction(SliceSelection("slice_data", "ccf").to_payload())
    action_group = FakeActionGroup([other, checked], checked)
    queries = FakeQueries()
    presenter = _presenter(queries, action_group)

    assert presenter.current_slice_selection() == selection
    assert presenter.action_for_selection(selection) is checked
    assert presenter.current_scalar_slice_channel() == "histology_registration"
    assert queries.rendered_selections == [selection]


def test_slice_panel_maps_legacy_slice_payload_by_identity() -> None:
    slice_data = {"ccf": np.array([[1.0]])}
    fp_slice_data = {"label": np.zeros((1, 1, 3))}
    queries = FakeQueries(
        ActiveSliceDataState(
            key=AlignmentKey("rec", "stream", 0),
            slice_data=slice_data,
            fp_slice_data=fp_slice_data,
        )
    )
    presenter = _presenter(queries)
    calls: list[SliceSelection] = []
    presenter.plot_slice_selection = calls.append

    presenter.plot_slice(slice_data, "ccf")
    presenter.plot_slice(fp_slice_data, "label")
    presenter.plot_slice({"ccf": np.array([[1.0]])}, "ccf")

    assert calls == [
        SliceSelection("slice_data", "ccf"),
        SliceSelection("fp_slice_data", "label"),
    ]
