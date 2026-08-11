"""Tests for desktop histology render choreography."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_read_models import ActiveAlignmentRenderState
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.desktop.presenters.histology_presenter import (
    DesktopHistologyPresenter,
    DesktopHistologyRenderCallbacks,
)


class FakeQueries:
    def __init__(
        self,
        render_state: ActiveAlignmentRenderState | None,
        *,
        probe_extent: Any = "probe_extent",
        fit_state: Any = "fit_state",
        nearby_state: Any = "nearby_state",
    ) -> None:
        self.render_state = render_state
        self.probe_extent = probe_extent
        self.fit_state = fit_state
        self.nearby_state = nearby_state
        self.screen_state = (
            None
            if render_state is None or probe_extent is None or fit_state is None
            else self._screen_state(render_state, nearby=None)
        )
        self.nearby_screen_state = (
            None
            if self.screen_state is None
            else self._screen_state(render_state, nearby=nearby_state)
        )
        self.calls: list[Any] = []
        self.alignment_render = SimpleNamespace(
            active_alignment_render_state=self.active_alignment_render_state,
            active_histology_screen_state=self.active_histology_screen_state,
            active_nearby_boundary_screen_state=(
                self.active_nearby_boundary_screen_state
            ),
            histology_screen_state_for_alignment=(
                self.histology_screen_state_for_alignment
            ),
        )

    def _screen_state(
        self,
        render_state: ActiveAlignmentRenderState,
        *,
        nearby: Any,
    ) -> Any:
        return SimpleNamespace(
            histology=SimpleNamespace(
                key=render_state.key,
                histology=render_state.histology,
                probe_extent=self.probe_extent,
            ),
            scale_factor=SimpleNamespace(
                key=render_state.key,
                region=render_state.histology.scale.region,
                scale=render_state.histology.scale.scale,
                probe_extent=self.probe_extent,
            ),
            fit=self.fit_state,
            nearby=nearby,
        )

    def active_alignment_render_state(self) -> ActiveAlignmentRenderState | None:
        self.calls.append("active_alignment_render_state")
        return self.render_state

    def active_histology_screen_state(self, **kwargs: Any) -> Any:
        self.calls.append(("active_histology_screen", kwargs))
        return self.screen_state

    def active_nearby_boundary_screen_state(self, **kwargs: Any) -> Any:
        self.calls.append(("active_nearby_boundary_screen", kwargs))
        return self.nearby_screen_state

    def histology_screen_state_for_alignment(
        self,
        render_state: ActiveAlignmentRenderState,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(("histology_screen_for_alignment", render_state, kwargs))
        if self.probe_extent is None or self.fit_state is None:
            return None
        return self._screen_state(render_state, nearby=None)


class FakePanel:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def render_aligned(
        self,
        state: Any,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> None:
        self.calls.append(
            ("aligned", state.histology, state.probe_extent, fig, movable)
        )

    def render_reference(
        self,
        state: Any,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> None:
        self.calls.append(
            ("reference", state.histology, state.probe_extent, fig, movable)
        )

    def render_scale_factor(self, state: Any, *, y_range: tuple[float, float]) -> None:
        self.calls.append(("scale", state.region, state.scale, y_range))

    def render_fit(self, state: Any) -> None:
        self.calls.append(("fit", state))

    def set_labels_visible(self, visible: bool) -> None:
        self.calls.append(("labels", visible))


def _render_state() -> ActiveAlignmentRenderState:
    active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    return ActiveAlignmentRenderState(
        key=AlignmentKey("rec", "stream", 1),
        active_alignment=active_alignment,
        histology=SimpleNamespace(
            scale=SimpleNamespace(region="region", scale="scale")
        ),
        projection="projection",
    )


def _presenter(
    queries: FakeQueries,
    panel: FakePanel,
) -> DesktopHistologyPresenter:
    return DesktopHistologyPresenter(
        app=SimpleNamespace(queries=queries),
        panel=panel,
        callbacks=DesktopHistologyRenderCallbacks(
            probe_extent_query_kwargs=lambda: {
                "probe_tip_um": 0.0,
                "probe_top_um": 3840.0,
                "probe_extra_um": 100.0,
            },
            fit_depth_um=lambda: "depth",
            lin_fit_enabled=lambda: False,
            scale_factor_y_range=lambda: (10.0, 20.0),
        ),
    )


def test_render_alignment_edit_updates_histology_scale_and_fit() -> None:
    render_state = _render_state()
    queries = FakeQueries(render_state)
    panel = FakePanel()

    rendered = _presenter(queries, panel).render_alignment_edit(render_state)

    assert rendered is True
    assert queries.calls == [
        (
            "histology_screen_for_alignment",
            render_state,
            {
                "probe_tip_um": 0.0,
                "probe_top_um": 3840.0,
                "probe_extra_um": 100.0,
                "depth_um": "depth",
                "lin_fit": False,
            },
        )
    ]
    assert panel.calls == [
        ("aligned", render_state.histology, "probe_extent", None, True),
        ("scale", "region", "scale", (10.0, 20.0)),
        ("fit", "fit_state"),
    ]


def test_render_active_panels_updates_reference_aligned_scale_and_fit() -> None:
    render_state = _render_state()
    queries = FakeQueries(render_state)
    panel = FakePanel()

    rendered = _presenter(queries, panel).render_active_panels()

    assert rendered is True
    assert queries.calls == [
        (
            "active_histology_screen",
            {
                "probe_tip_um": 0.0,
                "probe_top_um": 3840.0,
                "probe_extra_um": 100.0,
                "depth_um": "depth",
                "lin_fit": False,
            },
        )
    ]
    assert panel.calls == [
        ("reference", render_state.histology, "probe_extent", None, False),
        ("aligned", render_state.histology, "probe_extent", None, True),
        ("labels", True),
        ("scale", "region", "scale", (10.0, 20.0)),
        ("fit", "fit_state"),
    ]


def test_render_active_aligned_noops_without_active_alignment() -> None:
    queries = FakeQueries(None)
    panel = FakePanel()

    rendered = _presenter(queries, panel).render_active_aligned()

    assert rendered is False
    assert queries.calls == [
        (
            "active_histology_screen",
            {
                "probe_tip_um": 0.0,
                "probe_top_um": 3840.0,
                "probe_extra_um": 100.0,
                "depth_um": "depth",
                "lin_fit": False,
            },
        )
    ]
    assert panel.calls == []


def test_render_alignment_edit_noops_without_probe_extent() -> None:
    render_state = _render_state()
    queries = FakeQueries(render_state, probe_extent=None)
    panel = FakePanel()

    rendered = _presenter(queries, panel).render_alignment_edit(render_state)

    assert rendered is False
    assert queries.calls == [
        (
            "histology_screen_for_alignment",
            render_state,
            {
                "probe_tip_um": 0.0,
                "probe_top_um": 3840.0,
                "probe_extra_um": 100.0,
                "depth_um": "depth",
                "lin_fit": False,
            },
        )
    ]
    assert panel.calls == []
