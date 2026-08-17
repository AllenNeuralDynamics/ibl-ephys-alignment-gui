"""Tests for alignment save input factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.application.alignment_save_input_factory import (
    AlignmentSaveInputFactory,
    AlignmentSaveInputFactoryError,
)
from ephys_alignment_gui.application.save_geometry_catalog import SaveGeometryCatalog
from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_state import AlignmentState
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.io.datapackage_loader import (
    ChannelTablePaths,
    MouseRoot,
    ProbeInfo,
)
from ephys_alignment_gui.io.input_dataset_snapshot import InputDatasetSnapshot


def test_alignment_save_input_factory_builds_output_input_from_catalog(
    tmp_path,
) -> None:
    key = AlignmentKey("rec", "stream", 0)
    state = AlignmentState()
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    channel_locations_ras = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output_directory = tmp_path / "out"
    factory = AlignmentSaveInputFactory(SaveGeometryCatalog(_input_dataset(tmp_path)))

    save_input = factory.build(
        key=key,
        state=state,
        channel_locations_ras=channel_locations_ras,
        output_directory=output_directory,
    )

    assert save_input.state is state
    assert save_input.output_directory == output_directory
    assert save_input.multi_shank
    np.testing.assert_array_equal(
        save_input.output_input.channel_locations_ras,
        channel_locations_ras,
    )
    np.testing.assert_array_equal(
        save_input.output_input.channel_coordinates,
        [[0.0, 0.0], [0.0, 20.0]],
    )
    np.testing.assert_array_equal(
        save_input.output_input.channel_identity.raw_ind, [10, 11]
    )
    np.testing.assert_array_equal(
        save_input.output_input.channel_identity.contact_id,
        [100, 101],
    )
    np.testing.assert_array_equal(
        save_input.output_input.channel_identity.shank_idx, [0, 0]
    )
    assert save_input.output_metadata.recording_id == "rec"
    assert save_input.output_metadata.ephys_collection == "stream"
    assert save_input.output_metadata.logical_probe == "logical-stream"
    assert save_input.output_metadata.shank_idx == 0
    assert save_input.output_metadata.n_shanks == 2
    assert save_input.output_metadata.probe_id == "probe-id"


def test_alignment_save_input_factory_requires_active_alignment(tmp_path) -> None:
    factory = AlignmentSaveInputFactory(SaveGeometryCatalog(_input_dataset(tmp_path)))

    with pytest.raises(AlignmentSaveInputFactoryError, match="active alignment"):
        factory.build(
            key=AlignmentKey("rec", "stream", 0),
            state=AlignmentState(),
            channel_locations_ras=np.empty((0, 3)),
            output_directory=tmp_path / "out",
        )


def test_alignment_save_input_factory_wraps_geometry_errors() -> None:
    factory = AlignmentSaveInputFactory(SaveGeometryCatalog())
    state = AlignmentState()
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )

    with pytest.raises(AlignmentSaveInputFactoryError, match="No input dataset"):
        factory.build(
            key=AlignmentKey("rec", "stream", 0),
            state=state,
            channel_locations_ras=np.empty((0, 3)),
            output_directory=Path("/tmp/out"),
        )


def _input_dataset(tmp_path: Path) -> InputDatasetSnapshot:
    channel_paths = _write_channel_table(tmp_path / "rec" / "stream")
    mouse_root = MouseRoot(
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
    return InputDatasetSnapshot.from_mouse_root(mouse_root)


def _write_channel_table(root: Path) -> ChannelTablePaths:
    root.mkdir(parents=True, exist_ok=True)
    paths = ChannelTablePaths(
        local_coordinates=root / "channels.localCoordinates.npy",
        raw_ind=root / "channels.rawInd.npy",
        contact_id=root / "channels.contactId.npy",
        shank_ind=root / "channels.shankInd.npy",
    )
    np.save(
        paths.local_coordinates,
        np.array(
            [
                [0.0, 0.0],
                [0.0, 20.0],
                [250.0, 0.0],
                [250.0, 20.0],
            ]
        ),
    )
    np.save(paths.raw_ind, np.array([10, 11, 20, 21]))
    assert paths.contact_id is not None
    np.save(paths.contact_id, np.array([100, 101, 200, 201]))
    np.save(paths.shank_ind, np.array([0, 0, 1, 1]))
    return paths
