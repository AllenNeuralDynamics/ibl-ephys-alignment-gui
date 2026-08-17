"""Tests for saved alignment output package manifests."""

from __future__ import annotations

import json
from importlib.resources import files

from jsonschema import Draft202012Validator

from ephys_alignment_gui.core.alignment_output import AlignmentOutputMetadata
from ephys_alignment_gui.services.alignment_output_package import (
    ALIGNMENT_OUTPUT_DATAPACKAGE_FILENAME,
    ALIGNMENT_OUTPUT_SCHEMA_NAME,
    ALIGNMENT_OUTPUT_SCHEMA_VERSION,
    load_previous_alignment_package_manifest,
    upsert_alignment_output_datapackage,
)


def test_load_previous_alignment_package_manifest_uses_manifest_keys(tmp_path):
    package = tmp_path / "ibl_annotations_mouse_2026-08-16_14-32-05"
    history_path = package / "custom" / "nested" / "prev_alignments.json"
    history_path.parent.mkdir(parents=True)
    expected = {"saved": [[1.0, 2.0], [3.0, 4.0]]}
    history_path.write_text(json.dumps(expected))

    manifest = {
        "schema_name": ALIGNMENT_OUTPUT_SCHEMA_NAME,
        "schema_version": ALIGNMENT_OUTPUT_SCHEMA_VERSION,
        "generated_by": {
            "name": "ibl-ephys-alignment-gui",
            "version": "test",
        },
        "mouse_id": "mouse",
        "recordings": {
            "recording-key": {
                "recording_id": "recording-from-manifest",
                "probes": {
                    "probe-key": {
                        "ephys_collection": "collection-from-manifest",
                        "logical_probe": "logical-probe",
                        "probe_id": "probe-id",
                        "n_shanks": 1,
                        "shanks": {
                            "1": {
                                "shank_idx": 0,
                                "shank_id": 1,
                                "files": {
                                    "metadata": "missing-metadata.json",
                                    "channel_locations": "missing-channel.json",
                                    "prev_alignments": (
                                        "custom/nested/prev_alignments.json"
                                    ),
                                    "ccf_channel_locations": "missing-ccf.json",
                                },
                            }
                        },
                    }
                },
            }
        },
    }
    (package / ALIGNMENT_OUTPUT_DATAPACKAGE_FILENAME).write_text(
        json.dumps(manifest)
    )

    loaded = load_previous_alignment_package_manifest(package)

    assert loaded is not None
    assert loaded[("recording-from-manifest", "collection-from-manifest", 0)] == (
        expected
    )


def test_upsert_alignment_output_datapackage_writes_valid_manifest(tmp_path):
    package = tmp_path / "ibl_annotations_mouse_2026-08-16_14-32-05"
    output_dir = package / "rec1" / "probeA"
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / "alignment_output_metadata_shank2.json"
    channel_path = output_dir / "channel_locations_shank2.json"
    history_path = output_dir / "prev_alignments_shank2.json"
    ccf_path = output_dir / "ccf_channel_locations_shank2.json"
    for path in (metadata_path, channel_path, history_path, ccf_path):
        path.write_text("{}")

    path = upsert_alignment_output_datapackage(
        output_package_directory=package,
        metadata=AlignmentOutputMetadata(
            recording_id="rec1",
            ephys_collection="probeA",
            logical_probe="logicalA",
            probe_id="probe-id",
            shank_idx=1,
            n_shanks=2,
        ),
        mouse_id="mouse",
        channel_results_path=channel_path,
        previous_alignments_path=history_path,
        ccf_channel_results_path=ccf_path,
        metadata_path=metadata_path,
    )

    assert path == package / ALIGNMENT_OUTPUT_DATAPACKAGE_FILENAME
    manifest = json.loads(path.read_text())
    _validate_alignment_output_datapackage(manifest)
    assert manifest["mouse_id"] == "mouse"
    assert manifest["recordings"]["rec1"]["probes"]["probeA"] == {
        "ephys_collection": "probeA",
        "logical_probe": "logicalA",
        "probe_id": "probe-id",
        "n_shanks": 2,
        "shanks": {
            "2": {
                "shank_idx": 1,
                "shank_id": 2,
                "files": {
                    "metadata": (
                        "rec1/probeA/alignment_output_metadata_shank2.json"
                    ),
                    "channel_locations": (
                        "rec1/probeA/channel_locations_shank2.json"
                    ),
                    "prev_alignments": "rec1/probeA/prev_alignments_shank2.json",
                    "ccf_channel_locations": (
                        "rec1/probeA/ccf_channel_locations_shank2.json"
                    ),
                },
            }
        },
    }


def test_alignment_output_schema_matches_pydantic_authoring_model():
    from scripts.generate_alignment_output_schema import AlignmentOutputPackage

    schema = _load_alignment_output_schema()
    assert schema == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **AlignmentOutputPackage.model_json_schema(),
    }


def _validate_alignment_output_datapackage(datapackage: dict) -> None:
    Draft202012Validator(_load_alignment_output_schema()).validate(datapackage)


def _load_alignment_output_schema() -> dict:
    resource = files("ephys_alignment_gui.io")
    for part in (
        "schemas",
        ALIGNMENT_OUTPUT_SCHEMA_NAME,
        ALIGNMENT_OUTPUT_SCHEMA_VERSION,
        "datapackage.schema.json",
    ):
        resource = resource.joinpath(part)
    loaded = json.loads(resource.read_text())
    assert isinstance(loaded, dict)
    return loaded
