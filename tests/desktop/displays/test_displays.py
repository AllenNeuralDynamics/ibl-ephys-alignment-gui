"""Tests for desktop display-region composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import ephys_alignment_gui.desktop.displays as module
from ephys_alignment_gui.desktop.displays import DesktopDisplayConfig, DesktopDisplays


class FakePlot:
    def __init__(self, name: str) -> None:
        self.name = name
        self.y_links: list[object] = []

    def setYLink(self, plot: object) -> None:
        self.y_links.append(plot)


def test_desktop_displays_factory_composes_display_regions(monkeypatch) -> None:
    calls: list[tuple[str, Any, Any, Any]] = []
    image_plot = FakePlot("image")
    line_plot = FakePlot("line")
    probe_plot = FakePlot("probe")
    aligned_plot = FakePlot("aligned")
    reference_plot = FakePlot("reference")
    perpendicular_plot = FakePlot("perpendicular")
    ephys = SimpleNamespace(
        panel=SimpleNamespace(
            plots=SimpleNamespace(
                image=image_plot,
                line=line_plot,
                probe=probe_plot,
            ),
            feature_y_range=lambda: (0.0, 1.0),
        ),
    )
    histology = SimpleNamespace(
        aligned_plot=aligned_plot,
        reference_plot=reference_plot,
        fit_plot="fit-plot",
    )
    slice_display = SimpleNamespace(
        perpendicular_plot=perpendicular_plot,
        set_perpendicular_depth_link=lambda plot: perpendicular_plot.setYLink(plot),
    )
    reference_lines = object()

    def create_ephys(*, config: Any) -> Any:
        calls.append(("ephys", None, config, None))
        return ephys

    def create_histology(
        *,
        config: Any,
        perpendicular_plot: Any,
    ) -> Any:
        calls.append(("histology", None, config, perpendicular_plot))
        assert perpendicular_plot is slice_display.perpendicular_plot
        return histology

    def create_reference_lines(*, bindings: Any) -> Any:
        calls.append(("reference_lines", None, bindings, None))
        assert bindings.histology_plot is aligned_plot
        assert bindings.reference_plot is reference_plot
        assert bindings.image_plot is image_plot
        assert bindings.line_plot is line_plot
        assert bindings.probe_plot is probe_plot
        assert bindings.perpendicular_plot is perpendicular_plot
        assert bindings.fit_plot == "fit-plot"
        return reference_lines

    def create_slice(*, config: Any) -> Any:
        calls.append(("slice", None, config, None))
        return slice_display

    monkeypatch.setattr(
        module.DesktopEphysDisplay,
        "create",
        staticmethod(create_ephys),
    )
    monkeypatch.setattr(
        module.DesktopHistologyDisplay,
        "create",
        staticmethod(create_histology),
    )
    monkeypatch.setattr(
        module.DesktopReferenceLineDisplay,
        "create",
        staticmethod(create_reference_lines),
    )
    monkeypatch.setattr(
        module.DesktopSliceDisplay,
        "create",
        staticmethod(create_slice),
    )
    config = DesktopDisplayConfig(
        ephys="ephys-ports",
        histology="histology-ports",
        slice="slice-ports",
    )

    result = DesktopDisplays.create(config=config)

    assert result.ephys is ephys
    assert result.histology is histology
    assert result.reference_lines is reference_lines
    assert result.slice is slice_display
    assert image_plot.y_links == [line_plot, aligned_plot]
    assert line_plot.y_links == [aligned_plot]
    assert probe_plot.y_links == [image_plot]
    assert perpendicular_plot.y_links == [aligned_plot]
    assert calls[:3] == [
        ("ephys", None, "ephys-ports", None),
        ("slice", None, "slice-ports", None),
        ("histology", None, "histology-ports", perpendicular_plot),
    ]
    assert calls[3][0] == "reference_lines"
