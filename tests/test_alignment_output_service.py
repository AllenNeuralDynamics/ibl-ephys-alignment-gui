"""Tests for alignment output construction service."""

from __future__ import annotations

import numpy as np
import pytest
from iblutil.util import Bunch

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.alignment_output_service import AlignmentOutputService
from ephys_alignment_gui.histology_data_service import HistologyDataContext


def test_alignment_output_service_creates_channel_dict() -> None:
    brain_regions = Bunch(
        id=np.array([10, 20]),
        xyz=np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]]),
        axial=np.array([20.0, 40.0]),
        lateral=np.array([5.0, 7.0]),
        acronym=np.array(["VISp", "LGd"]),
    )

    channel_dict = AlignmentOutputService.create_channel_dict(brain_regions)

    assert channel_dict["channel_0"] == {
        "x": np.float64(1000.0),
        "y": np.float64(2000.0),
        "z": np.float64(3000.0),
        "axial": np.float64(20.0),
        "lateral": np.float64(5.0),
        "brain_region_id": 10,
        "brain_region": "VISp",
    }
    assert channel_dict["channel_1"]["brain_region"] == "LGd"


def test_alignment_output_service_requires_loaded_histology() -> None:
    service = AlignmentOutputService(
        AlignmentDataContext(),
        HistologyDataContext(),
    )

    with pytest.raises(ValueError, match="Brain atlas"):
        service.get_alignment_results(
            np.zeros((1, 3), dtype=float),
            np.zeros((1, 2), dtype=float),
        )
