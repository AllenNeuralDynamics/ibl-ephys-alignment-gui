"""Saved alignment output package manifests."""

from __future__ import annotations

import json
import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ephys_alignment_gui import __version__
from ephys_alignment_gui.core.alignment_output import AlignmentOutputMetadata

logger = logging.getLogger(__name__)


AlignmentHistory = dict[str, list[list[float]]]
ALIGNMENT_OUTPUT_SCHEMA_NAME = "aind-ibl-ephys-alignment-output"
ALIGNMENT_OUTPUT_SCHEMA_VERSION = "1.0.0"
ALIGNMENT_OUTPUT_DATAPACKAGE_FILENAME = "datapackage.json"


def upsert_alignment_output_datapackage(
    *,
    output_package_directory: Path,
    metadata: AlignmentOutputMetadata,
    mouse_id: str | None,
    channel_results_path: Path,
    previous_alignments_path: Path,
    ccf_channel_results_path: Path,
    metadata_path: Path,
) -> Path:
    """Create or update the package-level manifest for GUI outputs."""
    output_package_directory = Path(output_package_directory)
    datapackage_path = output_package_directory / ALIGNMENT_OUTPUT_DATAPACKAGE_FILENAME
    manifest = _existing_output_datapackage(datapackage_path)
    if mouse_id is not None:
        manifest["mouse_id"] = mouse_id

    recordings = manifest.setdefault("recordings", {})
    recording = recordings.setdefault(metadata.recording_id, {"probes": {}})
    recording["recording_id"] = metadata.recording_id
    probes = recording.setdefault("probes", {})
    probe = probes.setdefault(metadata.ephys_collection, {"shanks": {}})
    probe.update(
        {
            "ephys_collection": metadata.ephys_collection,
            "logical_probe": metadata.logical_probe,
            "probe_id": metadata.probe_id,
            "n_shanks": metadata.n_shanks,
        }
    )
    shanks = probe.setdefault("shanks", {})
    shank_id = metadata.shank_idx + 1
    shanks[str(shank_id)] = {
        "shank_idx": metadata.shank_idx,
        "shank_id": shank_id,
        "files": {
            "metadata": _relative_manifest_path(
                output_package_directory,
                metadata_path,
            ),
            "channel_locations": _relative_manifest_path(
                output_package_directory,
                channel_results_path,
            ),
            "prev_alignments": _relative_manifest_path(
                output_package_directory,
                previous_alignments_path,
            ),
            "ccf_channel_locations": _relative_manifest_path(
                output_package_directory,
                ccf_channel_results_path,
            ),
        },
    }
    _validate_output_datapackage(manifest)
    _write_dict_to_json(datapackage_path, manifest)
    return datapackage_path


def load_previous_alignment_package_manifest(
    folder: Path,
) -> dict[tuple[str, str, int], AlignmentHistory] | None:
    """Load previous alignments from a package manifest when available."""
    folder = Path(folder)
    datapackage_path = folder / ALIGNMENT_OUTPUT_DATAPACKAGE_FILENAME
    if not datapackage_path.exists():
        return None
    with open(datapackage_path) as f:
        manifest = json.load(f)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_name") != ALIGNMENT_OUTPUT_SCHEMA_NAME
    ):
        return None
    _validate_output_datapackage(manifest)

    histories: dict[tuple[str, str, int], AlignmentHistory] = {}
    recordings = manifest.get("recordings", {})
    if not isinstance(recordings, dict):
        return histories
    for recording_key, recording in recordings.items():
        if not isinstance(recording, dict):
            continue
        recording_id = str(recording.get("recording_id") or recording_key)
        _load_recording_manifest_histories(
            folder,
            recording_id=recording_id,
            recording=recording,
            histories=histories,
        )
    return histories


def _load_recording_manifest_histories(
    folder: Path,
    *,
    recording_id: str,
    recording: dict[str, Any],
    histories: dict[tuple[str, str, int], AlignmentHistory],
) -> None:
    probes = recording.get("probes", {})
    if not isinstance(probes, dict):
        return
    for probe_key, probe in probes.items():
        if not isinstance(probe, dict):
            continue
        ephys_collection = str(probe.get("ephys_collection") or probe_key)
        _load_probe_manifest_histories(
            folder,
            recording_id=recording_id,
            ephys_collection=ephys_collection,
            probe=probe,
            histories=histories,
        )


def _load_probe_manifest_histories(
    folder: Path,
    *,
    recording_id: str,
    ephys_collection: str,
    probe: dict[str, Any],
    histories: dict[tuple[str, str, int], AlignmentHistory],
) -> None:
    shanks = probe.get("shanks", {})
    if not isinstance(shanks, dict):
        return
    for shank_key, shank in shanks.items():
        history = _load_manifest_shank_history(
            folder,
            recording_id=recording_id,
            ephys_collection=ephys_collection,
            shank_key=shank_key,
            shank=shank,
        )
        if history is not None:
            key, alignments = history
            histories[key] = alignments


def _load_manifest_shank_history(
    folder: Path,
    *,
    recording_id: str,
    ephys_collection: str,
    shank_key: str,
    shank: object,
) -> tuple[tuple[str, str, int], AlignmentHistory] | None:
    if not isinstance(shank, dict):
        return None
    shank_idx = _manifest_shank_idx(shank_key, shank)
    if shank_idx is None:
        return None
    files = shank.get("files", {})
    if not isinstance(files, dict):
        return None
    prev_alignments_path = files.get("prev_alignments")
    if not isinstance(prev_alignments_path, str):
        return None
    path = folder / prev_alignments_path
    if not path.exists():
        logger.warning(
            "Skipping manifest alignment entry with missing file: %s",
            path,
        )
        return None
    with open(path) as f:
        alignments: AlignmentHistory = json.load(f)
    return (recording_id, ephys_collection, shank_idx), alignments


def _existing_output_datapackage(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path) as f:
            loaded = json.load(f)
        if (
            isinstance(loaded, dict)
            and loaded.get("schema_name") == ALIGNMENT_OUTPUT_SCHEMA_NAME
        ):
            _validate_output_datapackage(loaded)
            return loaded
        logger.warning(
            "Ignoring non-GUI alignment output datapackage at %s",
            path,
        )
    return {
        "schema_name": ALIGNMENT_OUTPUT_SCHEMA_NAME,
        "schema_version": ALIGNMENT_OUTPUT_SCHEMA_VERSION,
        "generated_by": {
            "name": "ibl-ephys-alignment-gui",
            "version": __version__,
        },
        "mouse_id": None,
        "recordings": {},
    }


def _validate_output_datapackage(manifest: Any) -> None:
    schema = _load_output_datapackage_schema()
    try:
        Draft202012Validator(schema).validate(manifest)
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ValueError(
            "Alignment output datapackage does not match schema "
            f"{ALIGNMENT_OUTPUT_SCHEMA_VERSION} at {path}: {exc.message}"
        ) from exc


def _load_output_datapackage_schema() -> dict[str, Any]:
    resource = files("ephys_alignment_gui.io")
    for part in (
        "schemas",
        ALIGNMENT_OUTPUT_SCHEMA_NAME,
        ALIGNMENT_OUTPUT_SCHEMA_VERSION,
        "datapackage.schema.json",
    ):
        resource = resource.joinpath(part)
    loaded = json.loads(resource.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("Alignment output datapackage schema must be a JSON object")
    return loaded


def _relative_manifest_path(package_directory: Path, file_path: Path) -> str:
    try:
        return file_path.relative_to(package_directory).as_posix()
    except ValueError:
        return file_path.as_posix()


def _manifest_shank_idx(shank_key: str, shank: dict[str, Any]) -> int | None:
    raw_idx = shank.get("shank_idx")
    if raw_idx is not None:
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            return None
        return idx if idx >= 0 else None
    try:
        shank_id = int(shank_key)
    except ValueError:
        return None
    return max(0, shank_id - 1)


def _write_dict_to_json(file_path: Path, data_dict: dict[str, Any]) -> None:
    with open(file_path, "w") as fp:
        json.dump(data_dict, fp, indent=2, separators=(",", ": "))
