"""Tests for desktop ephys plot item ownership."""

from __future__ import annotations

from ephys_alignment_gui.desktop.ephys_plot_items import EphysPlotItems


class FakeFigure:
    def __init__(self) -> None:
        self.removed: list[object] = []

    def removeItem(self, item: object) -> None:
        self.removed.append(item)


def test_ephys_plot_items_detach_removes_and_clears_all_owned_items() -> None:
    items = EphysPlotItems(
        image_plots=["image"],
        line_plots=["line"],
        probe_plots=["probe"],
        image_colorbars=["image-cbar"],
        probe_colorbars=["probe-cbar"],
        probe_bounds=["bound"],
    )
    figures = {
        "img": FakeFigure(),
        "img_cb": FakeFigure(),
        "line": FakeFigure(),
        "probe": FakeFigure(),
        "probe_cb": FakeFigure(),
    }

    items.detach(figures)

    assert figures["img"].removed == ["image"]
    assert figures["img_cb"].removed == ["image-cbar"]
    assert figures["line"].removed == ["line"]
    assert figures["probe"].removed == ["probe", "bound"]
    assert figures["probe_cb"].removed == ["probe-cbar"]
    assert items.image_plots == []
    assert items.image_colorbars == []
    assert items.line_plots == []
    assert items.probe_plots == []
    assert items.probe_bounds == []
    assert items.probe_colorbars == []
