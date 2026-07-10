"""Tests for Qt-free probe data loading workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.datapackage_loader import ProbeInfo
from ephys_alignment_gui.ephys_data_service import ChannelTable, EphysStreamData
from ephys_alignment_gui.probe_data_workflow import ProbeDataWorkflow


def _channel_table() -> ChannelTable:
    return ChannelTable(
        local_coordinates=np.array(
            [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
        ),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )


def _probe(tmp_path: Path) -> ProbeInfo:
    return ProbeInfo(
        probe_id="rec1:streamA",
        probe_name="probeA",
        recording_id="rec1",
        logical_probe="probeA",
        ephys_collection="streamA",
        num_shanks=2,
        ephys_dir=tmp_path,
        channel_table=None,
        xyz_picks=(),
    )


def _stream(tmp_path: Path, table: ChannelTable) -> EphysStreamData:
    return EphysStreamData(
        recording_id="rec1",
        ephys_collection="streamA",
        ephys_dir=tmp_path,
        channel_table=table,
        alf_data={"channels": {"exists": True}},
        session_notes="notes",
    )


def test_probe_data_workflow_loads_stream_and_active_collection(tmp_path) -> None:
    probe = _probe(tmp_path)
    table = _channel_table()
    stream = _stream(tmp_path, table)
    context = AlignmentDataContext(probe_info=probe)
    context.attach_channel_table(table)

    class FakeEphysDataService:
        loaded_probe = None
        loaded_table = None

        def load_stream_data(self, selected_probe, channel_table=None):
            self.loaded_probe = selected_probe
            self.loaded_table = channel_table
            return stream

    service = FakeEphysDataService()
    workflow = ProbeDataWorkflow(context, service)

    loaded = workflow.load(1)

    assert loaded.stream is stream
    assert loaded.channel_collection.rows.tolist() == [2, 3]
    assert loaded.ephys_dir == tmp_path
    assert loaded.depths.tolist() == [0.0, 20.0]
    assert loaded.session_notes == "notes"
    assert loaded.alf_data is stream.alf_data
    assert service.loaded_probe is probe
    assert service.loaded_table is table


def test_probe_data_workflow_requires_channel_table(tmp_path) -> None:
    context = AlignmentDataContext(probe_info=_probe(tmp_path))
    workflow = ProbeDataWorkflow(context, ephys_data_service=object())

    with pytest.raises(RuntimeError, match="Channel info"):
        workflow.load(0)


def test_probe_data_workflow_validates_cached_stream(tmp_path) -> None:
    probe = _probe(tmp_path)
    table = _channel_table()
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection="other",
        ephys_dir=tmp_path,
        channel_table=table,
        alf_data={},
        session_notes="notes",
    )
    context = AlignmentDataContext(probe_info=probe)
    workflow = ProbeDataWorkflow(context, ephys_data_service=object())

    with pytest.raises(ValueError, match="collection"):
        workflow.from_stream(stream, 0)
