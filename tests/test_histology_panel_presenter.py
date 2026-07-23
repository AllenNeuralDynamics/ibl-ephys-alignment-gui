"""Tests for the desktop histology panel presenter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
    *,
    session: Any | None = None,
    histology_exists: bool = True,
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
        session_provider=lambda: session,
        histology_exists=lambda: histology_exists,
        set_axis=lambda *args, **kwargs: None,
        tip_line_moved=lambda: None,
        top_line_moved=lambda: None,
        padding_provider=lambda: 0.0,
    )
    return presenter, aligned_axis, reference_axis, aligned_plot, reference_plot


def test_histology_panel_toggles_label_axis_visibility() -> None:
    session = SimpleNamespace(label_status=True)
    presenter, aligned_axis, reference_axis, aligned_plot, reference_plot = _presenter(
        session=session
    )

    presenter.toggle_labels()

    assert session.label_status is False
    assert aligned_axis.pen is None
    assert aligned_axis.text_pen is None
    assert reference_axis.pen is None
    assert reference_axis.text_pen is None
    assert aligned_plot.update_count == 1
    assert reference_plot.update_count == 1

    presenter.toggle_labels()

    assert session.label_status is True
    assert aligned_axis.pen == "k"
    assert aligned_axis.text_pen == "k"
    assert reference_axis.pen == "k"
    assert reference_axis.text_pen == "k"
    assert aligned_plot.update_count == 2
    assert reference_plot.update_count == 2


def test_histology_panel_plotting_is_guarded_when_histology_is_absent() -> None:
    def raise_if_session_requested() -> Any:
        raise AssertionError("session should not be requested")

    presenter, *_ = _presenter(histology_exists=False)
    presenter.session_provider = raise_if_session_requested

    presenter.plot_aligned()
    presenter.plot_reference()
    presenter.plot_nearby()
