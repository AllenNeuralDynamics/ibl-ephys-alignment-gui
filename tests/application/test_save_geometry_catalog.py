"""Tests for lightweight save geometry catalog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.application.save_geometry_catalog import (
    SaveGeometryCatalog,
    SaveGeometryError,
)
from ephys_alignment_gui.application.workspace import AlignmentWorkspace
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.io.datapackage_loader import (
    ChannelTablePaths,
    MouseRoot,
    ProbeInfo,
)
from ephys_alignment_gui.io.input_dataset_snapshot import InputDatasetSnapshot


def test_save_geometry_catalog_loads_shank_geometry_and_identity(tmp_path) -> None:
    channel_paths = _write_channel_table(
        tmp_path / "rec" / "stream",
        local_coordinates=np.array(
            [
                [0.0, 0.0],
                [0.0, 20.0],
                [250.0, 0.0],
                [250.0, 20.0],
            ]
        ),
        raw_ind=np.array([10, 11, 20, 21]),
        contact_id=np.array([100, 101, 200, 201]),
        shank_ind=np.array([0, 0, 1, 1]),
    )
    snapshot = _input_dataset(tmp_path, channel_paths=channel_paths)
    catalog = SaveGeometryCatalog(snapshot)
    key = AlignmentKey("rec", "stream", 1)

    geometry = catalog.geometry_for_key(key)

    np.testing.assert_array_equal(
        geometry.channel_coordinates,
        [[250.0, 0.0], [250.0, 20.0]],
    )
    np.testing.assert_array_equal(geometry.channel_depths_um, [0.0, 20.0])
    np.testing.assert_array_equal(geometry.channel_identity.raw_ind, [20, 21])
    np.testing.assert_array_equal(geometry.channel_identity.contact_id, [200, 201])
    np.testing.assert_array_equal(geometry.channel_identity.shank_idx, [1, 1])
    assert geometry.multi_shank
    assert geometry.output_metadata.recording_id == "rec"
    assert geometry.output_metadata.ephys_collection == "stream"
    assert geometry.output_metadata.logical_probe == "logical-stream"
    assert geometry.output_metadata.shank_idx == 1
    assert geometry.output_metadata.n_shanks == 2
    assert geometry.output_metadata.probe_id == "probe-id"
    assert catalog.geometry_for_key(key) is geometry


def test_save_geometry_catalog_reports_missing_channel_paths(tmp_path) -> None:
    channel_paths = ChannelTablePaths(
        local_coordinates=tmp_path / "missing" / "channels.localCoordinates.npy",
        raw_ind=tmp_path / "missing" / "channels.rawInd.npy",
        contact_id=None,
        shank_ind=tmp_path / "missing" / "channels.shankInd.npy",
    )
    catalog = SaveGeometryCatalog(_input_dataset(tmp_path, channel_paths=channel_paths))

    with pytest.raises(SaveGeometryError, match="Missing save-critical"):
        catalog.geometry_for_key(AlignmentKey("rec", "stream", 0))


def test_save_geometry_catalog_requires_input_dataset() -> None:
    catalog = SaveGeometryCatalog()

    with pytest.raises(SaveGeometryError, match="No input dataset snapshot"):
        catalog.geometry_for_key(AlignmentKey("rec", "stream", 0))


def test_mouse_root_selection_refreshes_workspace_save_geometry_catalog(
    monkeypatch,
    tmp_path,
) -> None:
    root = tmp_path / "mouse"
    root.mkdir()
    channel_paths = _write_channel_table(
        root / "rec" / "stream",
        local_coordinates=np.array([[0.0, 0.0], [0.0, 20.0]]),
        raw_ind=np.array([10, 11]),
        contact_id=np.array([100, 101]),
        shank_ind=np.array([0, 0]),
    )
    mouse_root = _mouse_root(tmp_path, channel_paths=channel_paths)
    monkeypatch.setattr(
        "ephys_alignment_gui.io.alignment_data_context.load_mouse_root",
        lambda _root: mouse_root,
    )
    workspace = AlignmentWorkspace()

    workspace.app.commands.metadata.set_mouse_root(root)

    assert workspace.save_geometry_catalog.input_dataset is (
        workspace.data_context.input_dataset
    )
    geometry = workspace.save_geometry_catalog.geometry_for_key(
        AlignmentKey("rec", "stream", 0)
    )
    np.testing.assert_array_equal(geometry.channel_identity.raw_ind, [10, 11])


def _input_dataset(
    tmp_path: Path,
    *,
    channel_paths: ChannelTablePaths,
) -> InputDatasetSnapshot:
    return InputDatasetSnapshot.from_mouse_root(
        _mouse_root(tmp_path, channel_paths=channel_paths)
    )


def _mouse_root(
    tmp_path: Path,
    *,
    channel_paths: ChannelTablePaths,
) -> MouseRoot:
    return MouseRoot(
        root=tmp_path,
        schema_version="4.1.0",
        mouse_id="mouse",
        transforms=None,
        histology=None,
        probes={
            "rec": {
                "stream": ProbeInfo(
                    probe_id="probe-id",
                    probe_name="stream",
                    recording_id="rec",
                    logical_probe="logical-stream",
                    ephys_collection="stream",
                    num_shanks=2,
                    ephys_dir=tmp_path / "rec" / "stream",
                    channel_table=channel_paths,
                    xyz_picks=(),
                )
            }
        },
    )


def _write_channel_table(
    root: Path,
    *,
    local_coordinates: np.ndarray,
    raw_ind: np.ndarray,
    contact_id: np.ndarray,
    shank_ind: np.ndarray,
) -> ChannelTablePaths:
    root.mkdir(parents=True, exist_ok=True)
    paths = ChannelTablePaths(
        local_coordinates=root / "channels.localCoordinates.npy",
        raw_ind=root / "channels.rawInd.npy",
        contact_id=root / "channels.contactId.npy",
        shank_ind=root / "channels.shankInd.npy",
    )
    np.save(paths.local_coordinates, local_coordinates)
    np.save(paths.raw_ind, raw_ind)
    assert paths.contact_id is not None
    np.save(paths.contact_id, contact_id)
    np.save(paths.shank_ind, shank_ind)
    return paths
