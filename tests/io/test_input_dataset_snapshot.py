"""Tests for lightweight input dataset snapshots."""

from __future__ import annotations

from pathlib import Path

from ephys_alignment_gui.io.datapackage_loader import (
    ChannelTablePaths,
    MouseRoot,
    ProbeInfo,
)
from ephys_alignment_gui.io.input_dataset_snapshot import InputDatasetSnapshot


def test_input_dataset_snapshot_normalizes_probe_lookup_and_sessions(
    tmp_path,
) -> None:
    probe_a = _probe(
        tmp_path,
        recording_id="rec-a",
        probe_name="probe-a",
        ephys_collection="stream-a",
        num_shanks=1,
    )
    probe_b = _probe(
        tmp_path,
        recording_id="rec-b",
        probe_name="probe-b",
        ephys_collection="stream-b",
        num_shanks=4,
    )
    mouse_root = _mouse_root(
        tmp_path,
        probes={
            "rec-b": {"probe-b": probe_b},
            "rec-a": {"probe-a": probe_a},
        },
    )

    snapshot = InputDatasetSnapshot.from_mouse_root(mouse_root)

    assert snapshot.root == tmp_path
    assert snapshot.schema_version == "4.1.0"
    assert snapshot.mouse_id == "mouse"
    assert snapshot.transforms is mouse_root.transforms
    assert snapshot.histology is mouse_root.histology
    assert snapshot.sessions == ("rec-a", "rec-b")
    assert snapshot.stream_keys == (("rec-a", "stream-a"), ("rec-b", "stream-b"))
    assert snapshot.probes_for_session("rec-b") == ("probe-b",)
    stream_probe = snapshot.probe_for_stream_key("rec-b", "stream-b")
    assert stream_probe.probe_name == "probe-b"
    assert stream_probe.logical_probe == "logical-probe-b"
    assert stream_probe.num_shanks == 4


def test_input_dataset_snapshot_reports_missing_save_critical_paths(tmp_path) -> None:
    present_paths = _channel_paths(tmp_path / "present", touch=True)
    missing_paths = _channel_paths(tmp_path / "missing", touch=False)
    mouse_root = _mouse_root(
        tmp_path,
        probes={
            "rec": {
                "present": _probe(
                    tmp_path,
                    probe_name="present",
                    ephys_collection="present",
                    channel_table=present_paths,
                ),
                "missing": _probe(
                    tmp_path,
                    probe_name="missing",
                    ephys_collection="missing",
                    channel_table=missing_paths,
                ),
                "no-table": _probe(
                    tmp_path,
                    probe_name="no-table",
                    ephys_collection="no-table",
                    channel_table=None,
                ),
            }
        },
    )

    snapshot = InputDatasetSnapshot.from_mouse_root(mouse_root)
    missing = snapshot.missing_save_critical_paths()

    assert {
        (item.ephys_collection, item.role, item.path)
        for item in missing
    } == {
        (
            "missing",
            "channel_table.local_coordinates",
            missing_paths.local_coordinates,
        ),
        ("missing", "channel_table.raw_ind", missing_paths.raw_ind),
        ("missing", "channel_table.shank_ind", missing_paths.shank_ind),
        ("no-table", "channel_table", None),
    }


def _mouse_root(
    root: Path,
    *,
    probes: dict[str, dict[str, ProbeInfo]],
) -> MouseRoot:
    return MouseRoot(
        root=root,
        schema_version="4.1.0",
        mouse_id="mouse",
        transforms=None,
        histology=None,
        probes=probes,
    )


def _probe(
    tmp_path: Path,
    *,
    recording_id: str = "rec",
    probe_name: str = "probe",
    ephys_collection: str = "stream",
    num_shanks: int = 1,
    channel_table: ChannelTablePaths | None | object = ...,
) -> ProbeInfo:
    if channel_table is ...:
        channel_table = _channel_paths(tmp_path / recording_id / probe_name)
    return ProbeInfo(
        probe_id=f"id-{probe_name}",
        probe_name=probe_name,
        recording_id=recording_id,
        logical_probe=f"logical-{probe_name}",
        ephys_collection=ephys_collection,
        num_shanks=num_shanks,
        ephys_dir=tmp_path / recording_id / probe_name,
        channel_table=channel_table,
        xyz_picks=(),
    )


def _channel_paths(root: Path, *, touch: bool = True) -> ChannelTablePaths:
    root.mkdir(parents=True, exist_ok=True)
    paths = ChannelTablePaths(
        local_coordinates=root / "channels.localCoordinates.npy",
        raw_ind=root / "channels.rawInd.npy",
        contact_id=root / "channels.contactId.npy",
        shank_ind=root / "channels.shankInd.npy",
    )
    if touch:
        paths.local_coordinates.touch()
        paths.raw_ind.touch()
        paths.shank_ind.touch()
    return paths
