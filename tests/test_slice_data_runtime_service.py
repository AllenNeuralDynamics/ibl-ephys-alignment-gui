"""Tests for slice runtime materialization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentHistologyData,
    HistologyPlotData,
    ScaleFactorData,
)
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.slice_data_runtime_service import SliceDataRuntimeService
from ephys_alignment_gui.slice_runtime import SliceRuntime


class FakeSliceService:
    def __init__(self) -> None:
        self.coronal_calls: list[dict[str, Any]] = []
        self.perpendicular_calls: list[dict[str, Any]] = []

    def build_slice_set(self, **kwargs):
        self.coronal_calls.append(kwargs)
        return {"ccf": np.array([[1.0]]), "scale": [1.0, 1.0], "offset": [0.0, 0.0]}

    def build_perpendicular_slice_image(self, **kwargs):
        self.perpendicular_calls.append(kwargs)
        n_perp_samples = kwargs["n_perp_samples"]
        n_depths = len(kwargs["feature_grid_m"])
        return np.ones((n_perp_samples, n_depths))


def test_coronal_slice_state_uses_keyed_runtime_cache() -> None:
    service = SliceDataRuntimeService()
    key = AlignmentKey("rec", "stream", 0)
    track = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    shank_runtime = SimpleNamespace(
        ephysalign=SimpleNamespace(track_interpolation_ras=track),
        slice_runtime=SliceRuntime(),
    )
    histology_context = SimpleNamespace(
        brain_atlas=object(),
        histology_images={},
        lazy_channel_paths={},
    )
    slice_service = FakeSliceService()

    first = service.ensure_coronal_slice_state(
        key=key,
        shank_runtime=shank_runtime,
        histology_context=histology_context,
        slice_service=slice_service,
    )
    second = service.ensure_coronal_slice_state(
        key=key,
        shank_runtime=shank_runtime,
        histology_context=histology_context,
        slice_service=slice_service,
    )

    assert first is not None
    assert second is not None
    assert second.slice_data is first.slice_data
    assert len(slice_service.coronal_calls) == 1


def test_coronal_slice_state_fails_closed_without_brain_atlas() -> None:
    service = SliceDataRuntimeService()
    key = AlignmentKey("rec", "stream", 0)
    shank_runtime = SimpleNamespace(
        ephysalign=SimpleNamespace(track_interpolation_ras=np.array([[0.0, 0.0, 0.0]])),
        slice_runtime=SliceRuntime(),
    )

    state = service.ensure_coronal_slice_state(
        key=key,
        shank_runtime=shank_runtime,
        histology_context=SimpleNamespace(brain_atlas=None),
        slice_service=FakeSliceService(),
    )

    assert state is None


def test_perpendicular_slice_state_uses_keyed_runtime_cache() -> None:
    service = SliceDataRuntimeService()
    key = AlignmentKey("rec", "stream", 0)
    active_alignment = ActiveAlignment(
        feature=np.array([0.0, 1.0]),
        track=np.array([0.1, 1.1]),
    )
    track_interpolation = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [2.0, 0.0, 4.0]])
    shank_runtime = SimpleNamespace(
        ephysalign=SimpleNamespace(
            track_interpolation_ras=track_interpolation,
            ephys_depths_along_track=np.array([0.0, 1.0, 2.0]),
        ),
        slice_runtime=SliceRuntime(),
        chn_depths=np.array([0.0, 100.0]),
    )
    histology = AlignmentHistologyData(
        histology=HistologyPlotData(
            region=np.array([-200.0, 500.0]),
            axis_label=[],
            colour=[],
        ),
        reference_histology=HistologyPlotData(region=[], axis_label=[], colour=[]),
        scale=ScaleFactorData(region=[], scale=[]),
    )
    histology_context = SimpleNamespace(
        brain_atlas=SimpleNamespace(bc=SimpleNamespace(dxyz=[20e-6, 20e-6, 20e-6])),
        histology_images={},
        lazy_channel_paths={},
    )
    slice_service = FakeSliceService()

    first = service.perpendicular_slice_state(
        key=key,
        active_alignment=active_alignment,
        shank_runtime=shank_runtime,
        histology=histology,
        histology_context=histology_context,
        slice_service=slice_service,
        channel_name="ccf",
    )
    second = service.perpendicular_slice_state(
        key=key,
        active_alignment=active_alignment,
        shank_runtime=shank_runtime,
        histology=histology,
        histology_context=histology_context,
        slice_service=slice_service,
        channel_name="ccf",
    )

    assert first is not None
    assert second is not None
    assert second.image is first.image
    assert first.n_perp_samples == 51
    assert first.n_depths == 36
    assert len(slice_service.perpendicular_calls) == 1
