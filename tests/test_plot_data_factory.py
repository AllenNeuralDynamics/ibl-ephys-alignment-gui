"""Tests for building PlotData from runtime channel-collection views."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ephys_alignment_gui.ephys_data_service import ChannelTable, EphysStreamData
from ephys_alignment_gui.plot_data_factory import PlotDataFactory


def _minimal_stream() -> EphysStreamData:
    channel_table = ChannelTable(
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
    return EphysStreamData(
        recording_id="rec1",
        ephys_collection="streamA",
        ephys_dir=Path("/tmp/ephys"),
        channel_table=channel_table,
        alf_data={
            "channels": {
                "exists": True,
                # Deliberately omit shankInd: factory-built PlotData should
                # use ChannelCollectionView.rows, not rediscover rows here.
                "localCoordinates": channel_table.local_coordinates,
                "rawInd": np.array([100, 101, 102, 103]),
                "contactId": channel_table.contact_ids,
            },
            "spikes": {"exists": False},
            "clusters": {"exists": False},
        },
        session_notes="notes",
    )


def test_factory_builds_plot_data_from_channel_collection_rows():
    stream = _minimal_stream()
    collection = stream.channel_collection(1)

    plot_data = PlotDataFactory().build(collection)

    assert plot_data.channel_collection is collection
    np.testing.assert_array_equal(plot_data.chn_rows, np.array([1, 3]))
    np.testing.assert_array_equal(plot_data.chn_ind, np.array([1, 3]))
    np.testing.assert_array_equal(
        plot_data.chn_coords,
        np.array([[250.0, 0.0], [250.0, 20.0]]),
    )
    np.testing.assert_array_equal(
        plot_data.chn_contact_id_all,
        np.array(["s0e0", "s1e0", "s0e1", "s1e1"]),
    )


def test_factory_can_build_from_stream_and_shank_index():
    plot_data = PlotDataFactory().build_for_stream(_minimal_stream(), shank_idx=0)

    np.testing.assert_array_equal(plot_data.chn_rows, np.array([0, 2]))
