"""Tests for desktop histology display composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop_histology_display import (
    DesktopHistologyDisplay,
    DesktopHistologyDisplayPorts,
)


class FakeLayout:
    def scene(self) -> str:
        return "histology-scene"


class FakeQueries:
    def __init__(self, *, brain_atlas: Any = "atlas") -> None:
        self.nearby_calls: list[dict[str, Any]] = []
        self.nearby_state: Any = "nearby-state"
        self.alignment_render = SimpleNamespace(
            active_nearby_boundary_state=self.active_nearby_boundary_state,
        )
        self.workspace = SimpleNamespace(
            active_brain_atlas=lambda: brain_atlas,
            allen_structure_tree=lambda: "allen",
            depth_view_settings=lambda: SimpleNamespace(
                probe_tip_um=0.0,
                probe_top_um=3840.0,
                probe_extra_um=100.0,
            ),
            fit_depth_um=lambda: [],
            linear_fit_enabled=lambda: False,
        )

    def active_nearby_boundary_state(self, **kwargs: Any) -> Any:
        self.nearby_calls.append(kwargs)
        return self.nearby_state


def _display(
    queries: FakeQueries | None = None,
    *,
    histology_available: bool = True,
    brain_atlas: Any = "atlas",
) -> DesktopHistologyDisplay:
    queries = queries or FakeQueries(brain_atlas=brain_atlas)
    return DesktopHistologyDisplay.create(
        app=SimpleNamespace(queries=queries),
        ports=DesktopHistologyDisplayPorts(
            aligned_plot="aligned",
            reference_plot="reference",
            scale_plot="scale",
            scale_colorbar="scale-colorbar",
            aligned_axis="aligned-axis",
            reference_axis="reference-axis",
            layout=FakeLayout(),
            extra_y_axis="extra-y-axis",
            dotted_pen="dotted",
            fit_curve=SimpleNamespace(setData=lambda **_kwargs: None),
            fit_scatter=SimpleNamespace(setData=lambda **_kwargs: None),
            linear_fit_curve=SimpleNamespace(setData=lambda **_kwargs: None),
            set_axis=lambda *_args, **_kwargs: None,
            padding_provider=lambda: 0.05,
            scale_factor_y_range=lambda: (0.0, 1.0),
            histology_available=lambda: histology_available,
        ),
    )


def test_histology_display_composes_panel_and_presenter() -> None:
    display = _display()

    assert display.presenter.panel is display.panel
    assert display.aligned_plot == "aligned"
    assert display.reference_plot == "reference"
    assert display.extra_y_axis == "extra-y-axis"
    assert display.export_scene() == "histology-scene"


def test_histology_display_renders_nearby_boundaries() -> None:
    queries = FakeQueries()
    display = _display(queries)
    calls: list[Any] = []
    display.panel.render_nearby = lambda state, fig=None, *, movable=False: (
        calls.append((state, fig, movable))
    )

    rendered = display.render_active_nearby("fig", movable=True)

    assert rendered
    assert queries.nearby_calls == [
        {
            "probe_tip_um": 0.0,
            "probe_top_um": 3840.0,
            "probe_extra_um": 100.0,
            "allen": "allen",
            "brain_atlas": "atlas",
        }
    ]
    assert calls == [("nearby-state", "fig", True)]


def test_histology_display_fails_closed_without_histology_or_atlas() -> None:
    queries = FakeQueries()

    assert not _display(queries, histology_available=False).render_active_nearby()
    assert not _display(FakeQueries(brain_atlas=None)).render_active_nearby()
    assert queries.nearby_calls == []
