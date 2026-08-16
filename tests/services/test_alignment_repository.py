"""Tests for alignment persistence helpers."""

from __future__ import annotations

import json

from ephys_alignment_gui.core.alignment_output import AlignmentOutputMetadata
from ephys_alignment_gui.services.alignment_repository import AlignmentRepository


def test_load_previous_alignments_local_uses_shank_suffix(tmp_path):
    repo = AlignmentRepository()
    folder = tmp_path / "alignments"
    folder.mkdir()
    expected = {"saved": [[1.0, 2.0], [3.0, 4.0]]}
    with open(folder / "prev_alignments_shank2.json", "w") as f:
        json.dump(expected, f)

    loaded = repo.load_previous_alignments(
        folder=folder,
        recording_id="rec1",
        probe_name="probeA",
        shank_idx=1,
        n_shanks=2,
        use_docdb=False,
    )

    assert loaded is not None
    assert loaded.alignments == expected


def test_load_previous_alignments_local_single_shank_uses_base_name(tmp_path):
    repo = AlignmentRepository()
    folder = tmp_path / "alignments"
    folder.mkdir()
    expected = {"saved": [[1.0], [2.0]]}
    with open(folder / "prev_alignments.json", "w") as f:
        json.dump(expected, f)

    loaded = repo.load_previous_alignments(
        folder=folder,
        recording_id="rec1",
        probe_name="probeA",
        shank_idx=0,
        n_shanks=1,
        use_docdb=False,
    )

    assert loaded is not None
    assert loaded.alignments == expected


def test_load_previous_alignment_package_scans_recording_probe_dirs(tmp_path):
    repo = AlignmentRepository()
    package = tmp_path / "ibl_annotations_mouse_2026-08-16_14-32-05"
    probe_a = package / "rec1" / "probeA"
    probe_b = package / "rec1" / "probeB"
    probe_a.mkdir(parents=True)
    probe_b.mkdir(parents=True)
    align_a = {"a": [[1.0], [2.0]]}
    align_b = {"b": [[3.0], [4.0]]}
    with open(probe_a / "prev_alignments.json", "w") as f:
        json.dump(align_a, f)
    with open(probe_b / "prev_alignments_shank2.json", "w") as f:
        json.dump(align_b, f)

    loaded = repo.load_previous_alignment_package(folder=package)

    assert loaded.histories[("rec1", "probeA", 0)].alignments == align_a
    assert loaded.histories[("rec1", "probeB", 1)].alignments == align_b


def test_save_alignment_outputs_writes_expected_files(tmp_path):
    repo = AlignmentRepository()
    output_dir = tmp_path / "rec1" / "probeA"
    output_dir.mkdir(parents=True)

    saved = repo.save_alignment_outputs(
        output_directory=output_dir,
        shank_idx=1,
        multi_shank=True,
        channel_results={"channel": {"x": 1}},
        previous_alignments={"saved": [[1.0], [2.0]]},
        ccf_channel_results={"channel": {"ccf_x": 2}},
        metadata=AlignmentOutputMetadata(
            recording_id="rec1",
            ephys_collection="probeA",
            logical_probe="logicalA",
            probe_id="probe-id",
            shank_idx=1,
            n_shanks=2,
        ),
        use_docdb=False,
    )

    assert saved.channel_results_path == output_dir / "channel_locations_shank2.json"
    assert saved.previous_alignments_path == output_dir / "prev_alignments_shank2.json"
    assert (
        saved.ccf_channel_results_path
        == output_dir / "ccf_channel_locations_shank2.json"
    )
    assert (
        saved.metadata_path == output_dir / "alignment_output_metadata_shank2.json"
    )
    with open(saved.previous_alignments_path) as f:
        assert json.load(f) == {"saved": [[1.0], [2.0]]}
    with open(saved.metadata_path) as f:
        metadata = json.load(f)
    assert metadata == {
        "schema_version": "1.0.0",
        "recording_id": "rec1",
        "ephys_collection": "probeA",
        "logical_probe": "logicalA",
        "probe_id": "probe-id",
        "shank_idx": 1,
        "shank_id": 2,
        "n_shanks": 2,
        "files": {
            "channel_locations": "channel_locations_shank2.json",
            "prev_alignments": "prev_alignments_shank2.json",
            "ccf_channel_locations": "ccf_channel_locations_shank2.json",
        },
    }
    assert saved.docdb_probe_name is None
