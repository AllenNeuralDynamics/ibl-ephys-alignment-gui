"""Tests for PlotData channel geometry derivation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ephys_alignment_gui.ephys_data_service import ChannelTable, EphysStreamData
from ephys_alignment_gui.plot_channel_geometry import build_plot_channel_geometry


def test_plot_geometry_uses_runtime_channel_collection_rows() -> None:
    table = ChannelTable(
        local_coordinates=np.array(
            [
                [0.0, 0.0],
                [250.0, 0.0],
                [0.0, 20.0],
                [250.0, 20.0],
            ]
        ),
        raw_ind=np.array([100, 101, 102, 103]),
        contact_ids=np.array(["s0e0", "s1e0", "s0e1", "s1e1"]),
        shank_indices=np.array([0, 1, 0, 1]),
    )
    stream = EphysStreamData(
        recording_id="rec",
        ephys_collection="stream",
        ephys_dir=Path("/tmp/ephys"),
        channel_table=table,
        alf_data={},
        session_notes="",
    )
    collection = stream.channel_collection(1)

    geometry = build_plot_channel_geometry(
        {"channels": {"localCoordinates": table.local_coordinates}},
        shank_idx=1,
        channel_collection=collection,
    )

    np.testing.assert_array_equal(geometry.chn_rows, np.array([1, 3]))
    np.testing.assert_array_equal(geometry.chn_ind, np.array([1, 3]))
    np.testing.assert_array_equal(
        geometry.chn_coords,
        np.array([[250.0, 0.0], [250.0, 20.0]]),
    )
    np.testing.assert_array_equal(geometry.chn_contact_id_all, table.contact_ids)


def test_plot_geometry_dedupes_and_sorts_selected_rows() -> None:
    data = {
        "channels": {
            "localCoordinates": np.array(
                [
                    [0.0, 40.0],
                    [0.0, 0.0],
                    [0.0, 40.0],
                    [0.0, 20.0],
                ]
            ),
            "rawInd": np.array([10, 11, 12, 13]),
        }
    }

    geometry = build_plot_channel_geometry(data, shank_idx=0)

    np.testing.assert_array_equal(geometry.chn_rows, np.array([1, 3, 0]))
    np.testing.assert_array_equal(
        geometry.chn_coords,
        np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0]]),
    )
    assert geometry.chn_min == 0.0
    assert geometry.chn_max == 40.0
    assert geometry.chn_diff == 20.0
    np.testing.assert_array_equal(geometry.chn_full, np.array([0.0, 20.0, 40.0]))
    np.testing.assert_array_equal(geometry.idx_full, np.array([0, 1, 2]))
