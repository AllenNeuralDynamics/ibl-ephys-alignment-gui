"""Tests for the desktop histology panel presenter."""

from __future__ import annotations

from typing import Any

import numpy as np

from ephys_alignment_gui.histology_panel_presenter import (
    HistologyPanelAxes,
    HistologyPanelPlots,
    HistologyPanelPresenter,
    HistologyPanelStyle,
)


class FakeAxis:
    def __init__(self) -> None:
        self.pen: Any = "initial"
        self.text_pen: Any = "initial"

    def setPen(self, pen: Any) -> None:
        self.pen = pen

    def setTextPen(self, pen: Any) -> None:
        self.text_pen = pen


class FakePlot:
    def __init__(self) -> None:
        self.update_count = 0

    def update(self) -> None:
        self.update_count += 1


def _presenter(
) -> tuple[HistologyPanelPresenter, FakeAxis, FakeAxis, FakePlot, FakePlot]:
    aligned_axis = FakeAxis()
    reference_axis = FakeAxis()
    aligned_plot = FakePlot()
    reference_plot = FakePlot()
    presenter = HistologyPanelPresenter(
        plots=HistologyPanelPlots(
            aligned=aligned_plot,
            reference=reference_plot,
        ),
        axes=HistologyPanelAxes(
            aligned=aligned_axis,
            reference=reference_axis,
        ),
        style=HistologyPanelStyle(dotted_pen=None),
        set_axis=lambda *args, **kwargs: None,
        padding_provider=lambda: 0.0,
    )
    return presenter, aligned_axis, reference_axis, aligned_plot, reference_plot


def test_histology_panel_toggles_label_axis_visibility() -> None:
    presenter, aligned_axis, reference_axis, aligned_plot, reference_plot = _presenter()

    presenter.toggle_labels()

    assert presenter.label_status is False
    assert aligned_axis.pen is None
    assert aligned_axis.text_pen is None
    assert reference_axis.pen is None
    assert reference_axis.text_pen is None
    assert aligned_plot.update_count == 1
    assert reference_plot.update_count == 1

    presenter.toggle_labels()

    assert presenter.label_status is True
    assert aligned_axis.pen == "k"
    assert aligned_axis.text_pen == "k"
    assert reference_axis.pen == "k"
    assert reference_axis.text_pen == "k"
    assert aligned_plot.update_count == 2
    assert reference_plot.update_count == 2


def test_histology_panel_owns_selected_region_lookup() -> None:
    presenter, *_ = _presenter()
    selected = object()
    other = object()
    presenter.hist_regions = np.array([[other], [selected]], dtype=object)
    presenter.hist_ref_regions = np.array([[object()]], dtype=object)

    presenter.select_region(selected)

    assert presenter.selected_region_index() == 1


def test_histology_panel_owns_scale_factor_lookup() -> None:
    presenter, *_ = _presenter()
    selected = object()
    presenter.scale_regions = np.array([[object()], [selected]], dtype=object)
    presenter.scale_factor = np.array([0.75, 1.25])

    assert presenter.scale_factor_for_region_item(selected) == 1.25
