"""Tests for ephys runtime models and loading service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.datapackage_loader import ChannelTablePaths, ProbeInfo
from ephys_alignment_gui.ephys_data_service import (
    ChannelTable,
    EphysDataService,
    EphysStreamData,
)
from ephys_alignment_gui.histology_data_service import HistologyDataContext
from ephys_alignment_gui.load_data_local import LoadDataLocal


def _probe_info(ephys_dir: Path, channel_table: ChannelTablePaths) -> ProbeInfo:
    return ProbeInfo(
        probe_id="rec1:streamA",
        probe_name="probeA",
        recording_id="rec1",
        logical_probe="probeA",
        ephys_collection="streamA",
        num_shanks=2,
        ephys_dir=ephys_dir,
        channel_table=channel_table,
        xyz_picks=(),
    )


def _write_channel_table_files(
    ephys_dir: Path,
    *,
    local_coordinates: np.ndarray,
    raw_ind: np.ndarray | None = None,
    contact_ids: np.ndarray | None = None,
    shank_indices: np.ndarray | None = None,
) -> ChannelTablePaths:
    ephys_dir.mkdir()
    local_path = ephys_dir / "channels.localCoordinates.npy"
    raw_path = ephys_dir / "channels.rawInd.npy"
    contact_path = ephys_dir / "channels.contactId.npy"
    shank_path = ephys_dir / "channels.shankInd.npy"
    np.save(local_path, local_coordinates)
    np.save(
        raw_path, raw_ind if raw_ind is not None else np.arange(len(local_coordinates))
    )
    if contact_ids is not None:
        np.save(contact_path, contact_ids)
    if shank_indices is not None:
        np.save(shank_path, shank_indices)
    return ChannelTablePaths(
        local_coordinates=local_path,
        raw_ind=raw_path,
        contact_id=contact_path if contact_ids is not None else None,
        shank_ind=shank_path,
    )


def test_channel_collection_is_row_view_into_stream():
    table = ChannelTable(
        local_coordinates=np.array(
            [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
        ),
        raw_ind=np.array([10, 11, 12, 13]),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection="streamA",
        ephys_dir=Path("/tmp/ephys"),
        channel_table=table,
        alf_data={"channels": {"exists": True}},
        session_notes="notes",
    )

    collection = stream.channel_collection(1)

    assert collection.stream is stream
    assert collection.rows.tolist() == [2, 3]
    assert collection.local_coordinates.tolist() == [[250.0, 0.0], [250.0, 20.0]]
    assert collection.depths.tolist() == [0.0, 20.0]
    assert collection.raw_ind.tolist() == [12, 13]
    assert collection.contact_ids.tolist() == ["s1e0", "s1e1"]


def test_channel_table_rejects_invalid_shank_index():
    table = ChannelTable(
        local_coordinates=np.array([[0.0, 0.0], [0.0, 20.0]]),
        shank_indices=np.array([0, 0]),
    )

    with pytest.raises(IndexError):
        table.rows_for_shank(1)


def test_ephys_data_service_loads_channel_table_from_probe_paths(tmp_path):
    local_coordinates = np.array([[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]])
    channel_paths = _write_channel_table_files(
        tmp_path / "ephys",
        local_coordinates=local_coordinates,
        raw_ind=np.array([0, 1, 2, 3]),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )

    table = EphysDataService().load_channel_table(
        _probe_info(tmp_path / "ephys", channel_paths)
    )

    assert table.n_shanks == 2
    assert table.rows_for_shank(1).tolist() == [2, 3]
    assert table.contact_ids.tolist() == ["s0e0", "s0e1", "s1e0", "s1e1"]


def test_ephys_data_service_loads_stream_data(monkeypatch, tmp_path):
    local_coordinates = np.array([[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]])
    channel_paths = _write_channel_table_files(
        tmp_path / "ephys",
        local_coordinates=local_coordinates,
        raw_ind=np.array([10, 11, 12, 13]),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )
    np.save(tmp_path / "ephys" / "spike_shank_indices.npy", np.array([0, 1, 1]))
    (tmp_path / "ephys" / "session_notes.txt").write_text("session notes")

    def fake_load_object(_folder, _object_name):
        return {}

    monkeypatch.setattr(
        "ephys_alignment_gui.ephys_data_service.alfio.load_object",
        fake_load_object,
    )

    stream = EphysDataService().load_stream_data(
        _probe_info(tmp_path / "ephys", channel_paths)
    )

    assert stream.stream_key == ("rec1", "streamA")
    assert stream.session_notes == "session notes"
    assert stream.alf_data["channels"]["localCoordinates"] is (
        stream.channel_table.local_coordinates
    )
    assert stream.alf_data["channels"]["rawInd"].tolist() == [10, 11, 12, 13]
    assert stream.alf_data["channels"]["contactId"].tolist() == [
        "s0e0",
        "s0e1",
        "s1e0",
        "s1e1",
    ]
    assert stream.alf_data["channels"]["shankInd"].tolist() == [0, 0, 1, 1]
    assert stream.alf_data["rms_AP"]["xaxis"] == "Time (s)"
    assert stream.alf_data["spike_shanks"].tolist() == [0, 1, 1]
    assert stream.channel_collection(1).depths.tolist() == [0.0, 20.0]


def test_load_data_local_keeps_legacy_channel_adapter(tmp_path):
    table = ChannelTable(
        local_coordinates=np.array(
            [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
        ),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )
    probe = ProbeInfo(
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
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection="streamA",
        ephys_dir=tmp_path,
        channel_table=table,
        alf_data={"channels": {"exists": True}},
        session_notes="notes",
    )

    context = AlignmentDataContext(probe_info=probe)
    context.attach_channel_table(table)
    loader = LoadDataLocal(
        data_context=context,
        histology_context=HistologyDataContext(),
    )

    assert loader.set_channels_for_shank(1).tolist() == [0.0, 20.0]
    assert loader.chn_contact_id_all.tolist() == ["s0e0", "s0e1", "s1e0", "s1e1"]
    assert loader.channel_collection is not None
    assert loader.channel_collection.rows.tolist() == [2, 3]

    loader.set_channel_collection(stream.channel_collection(1))
    assert loader.ephys_stream is stream
    assert loader.channel_collection is not None
    assert loader.channel_collection.stream is stream


def test_load_data_local_restores_cached_stream_without_service_reload(tmp_path):
    table = ChannelTable(
        local_coordinates=np.array(
            [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
        ),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )
    probe = ProbeInfo(
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
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection="streamA",
        ephys_dir=tmp_path,
        channel_table=table,
        alf_data={"channels": {"exists": True}},
        session_notes="cached notes",
    )

    context = AlignmentDataContext(probe_info=probe)
    loader = LoadDataLocal(
        data_context=context,
        histology_context=HistologyDataContext(),
    )
    loader.set_channel_collection(stream.channel_collection(1))

    assert loader.ephys_stream is stream
    assert context.channel_table is table
    assert context.n_shanks == 2
    assert loader.chn_coords_all is table.local_coordinates
    assert loader.chn_contact_id_all is table.contact_ids
    assert loader.chn_coords.tolist() == [[250.0, 0.0], [250.0, 20.0]]

    assert loader.set_channels_for_shank(1).tolist() == [0.0, 20.0]
    assert loader.channel_collection is not None
    assert loader.channel_collection.stream is stream


def test_load_data_local_adapts_context_channel_collection_without_mirror_state(
    tmp_path,
):
    table = ChannelTable(
        local_coordinates=np.array(
            [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
        ),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )
    probe = ProbeInfo(
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
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection="streamA",
        ephys_dir=tmp_path,
        channel_table=table,
        alf_data={"channels": {"exists": True}},
        session_notes="context notes",
    )
    context = AlignmentDataContext(probe_info=probe)
    context.attach_channel_table(table)
    loader = LoadDataLocal(
        data_context=context,
        histology_context=HistologyDataContext(),
    )

    loader.set_channel_collection(stream.channel_collection(1))

    assert context.channel_table is table
    assert loader.ephys_stream is stream
    assert loader.chn_coords_all is table.local_coordinates
    assert loader.chn_contact_id_all is table.contact_ids
    assert loader.channel_collection is not None
    assert loader.channel_collection.stream is stream
