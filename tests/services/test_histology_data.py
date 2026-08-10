"""Tests for histology runtime data boundaries."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ephys_alignment_gui.services.histology_data import (
    HistologyDataContext,
    HistologyRuntimeData,
)


def test_histology_data_context_stores_runtime_data(tmp_path) -> None:
    image = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.uint8))
    brain_atlas = object()
    histology_data = HistologyRuntimeData(
        brain_atlas=brain_atlas,
        histology_images={"histology_registration": image},
        lazy_channel_paths={"fluor": tmp_path / "fluor.nii.gz"},
    )
    histology_context = HistologyDataContext()

    histology_context.set(histology_data)

    assert histology_context.brain_atlas is brain_atlas
    assert histology_context.histology_images == {"histology_registration": image}
    assert histology_context.lazy_channel_paths == {"fluor": tmp_path / "fluor.nii.gz"}
