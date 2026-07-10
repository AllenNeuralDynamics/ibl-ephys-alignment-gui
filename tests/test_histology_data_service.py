"""Tests for histology runtime data boundaries."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.histology_data_service import (
    HistologyDataContext,
    HistologyRuntimeData,
)
from ephys_alignment_gui.load_data_local import LoadDataLocal


def test_load_data_local_adapts_histology_runtime_data(tmp_path) -> None:
    image = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.uint8))
    brain_atlas = object()
    histology_data = HistologyRuntimeData(
        brain_atlas=brain_atlas,
        histology_images={"histology_registration": image},
        lazy_channel_paths={"fluor": tmp_path / "fluor.nii.gz"},
    )
    histology_context = HistologyDataContext()
    loader = LoadDataLocal(
        data_context=AlignmentDataContext(),
        histology_context=histology_context,
    )

    loader.set_histology_data(histology_data)

    assert loader.brain_atlas is brain_atlas
    assert loader.histology_images == {"histology_registration": image}
    assert histology_context.lazy_channel_paths == {"fluor": tmp_path / "fluor.nii.gz"}
