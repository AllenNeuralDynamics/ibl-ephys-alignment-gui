"""Persistence helpers for alignment histories and output files."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from aind_data_access_api.helpers.data_schema import get_quality_control_by_id

from ephys_alignment_gui.core.alignment_output import AlignmentOutputMetadata
from ephys_alignment_gui.io.docdb import (
    _default_doc_db_api_client,
    query_docdb_id,
    write_output_to_docdb,
)
from ephys_alignment_gui.services.alignment_output_package import (
    load_previous_alignment_package_manifest,
    upsert_alignment_output_datapackage,
)

logger = logging.getLogger(__name__)


AlignmentHistory = dict[str, list[list[float]]]


@dataclass(frozen=True)
class LoadedAlignmentHistory:
    """Alignment history loaded from DocDB or local files."""

    alignments: AlignmentHistory


@dataclass(frozen=True)
class LoadedAlignmentPackage:
    """Alignment histories loaded from a GUI output package directory."""

    histories: dict[tuple[str, str, int], LoadedAlignmentHistory]


@dataclass(frozen=True)
class SavedAlignmentOutputs:
    """Paths and status from persisting alignment outputs."""

    channel_results_path: Path
    previous_alignments_path: Path
    ccf_channel_results_path: Path
    metadata_path: Path
    datapackage_path: Path | None = None
    docdb_probe_name: str | None = None
    docdb_error: str | None = None


class AlignmentRepository:
    """Load and save alignment histories outside the Qt layer."""

    def load_previous_alignments(
        self,
        *,
        folder: Path | None,
        recording_id: str,
        probe_name: str,
        shank_idx: int,
        n_shanks: int,
        use_docdb: bool,
    ) -> LoadedAlignmentHistory | None:
        """Load prior alignments from DocDB, falling back to local files."""
        maybe_alignments = None
        load_local = not use_docdb
        if use_docdb:
            logger.debug("Using DocDB to get previous alignments")
            try:
                maybe_alignments = self._load_previous_alignments_docdb(
                    recording_id=recording_id,
                    probe_name=probe_name,
                    shank_idx=shank_idx,
                )
                if maybe_alignments is None:
                    load_local = True
            except ValueError as exc:
                logger.warning(
                    "Failed to load previous alignments from DocDB with "
                    "exception %s. Falling back to local file.",
                    exc,
                )
                load_local = True

        if load_local and folder is not None:
            maybe_alignments = self._load_previous_alignments_local(
                folder=folder,
                shank_idx=shank_idx,
                n_shanks=n_shanks,
            )

        if maybe_alignments is None:
            return None
        return LoadedAlignmentHistory(maybe_alignments)

    def _load_previous_alignments_docdb(
        self,
        *,
        recording_id: str,
        probe_name: str,
        shank_idx: int,
    ) -> AlignmentHistory | None:
        """Fetch alignment history from DocDB keyed by recording/probe/shank."""
        docdb_id = query_docdb_id(recording_id)[0]
        quality_control = get_quality_control_by_id(
            _default_doc_db_api_client(), docdb_id
        )

        if quality_control is None:
            return None

        evaluation_name = f"{recording_id}_{probe_name}_{shank_idx}"
        alignment_evaluations = [
            evaluation
            for evaluation in quality_control.evaluations
            if evaluation.name == f"Probe Alignment for {evaluation_name}"
        ]

        if len(alignment_evaluations) == 0:
            logger.info("No alignment found in DocDB for %s", evaluation_name)
            return None

        logger.info("Found existing record for %s. Loading alignment.", evaluation_name)
        latest_alignment_evaluation = max(
            alignment_evaluations, key=lambda evaluation: evaluation.created
        )
        curation_metric = latest_alignment_evaluation.metrics[0].value["curations"]
        alignments: AlignmentHistory = json.loads(curation_metric[0])[
            "previous_alignments"
        ]
        return alignments

    def _load_previous_alignments_local(
        self,
        *,
        folder: Path,
        shank_idx: int,
        n_shanks: int,
    ) -> AlignmentHistory | None:
        """Load ``prev_alignments.json`` or the shank-specific variant."""
        suffix = f"_shank{shank_idx + 1}" if n_shanks > 1 else ""
        path = folder / f"prev_alignments{suffix}.json"
        if not path.exists():
            return None
        with open(path) as f:
            alignments: AlignmentHistory = json.load(f)
        return alignments

    def load_previous_alignment_package(
        self,
        *,
        folder: Path,
    ) -> LoadedAlignmentPackage:
        """Load all previous-alignment histories in an output package."""
        manifest_histories = load_previous_alignment_package_manifest(
            Path(folder),
        )
        if manifest_histories is not None:
            return LoadedAlignmentPackage(
                {
                    key: LoadedAlignmentHistory(alignments)
                    for key, alignments in manifest_histories.items()
                }
            )

        histories: dict[tuple[str, str, int], LoadedAlignmentHistory] = {}
        for path in sorted(Path(folder).glob("*/*/prev_alignments*.json")):
            shank_idx = self._shank_idx_from_previous_alignment_path(path)
            if shank_idx is None:
                continue
            with open(path) as f:
                alignments: AlignmentHistory = json.load(f)
            probe_dir = path.parent
            recording_dir = probe_dir.parent
            histories[(recording_dir.name, probe_dir.name, shank_idx)] = (
                LoadedAlignmentHistory(alignments)
            )
        return LoadedAlignmentPackage(histories)

    def save_alignment_outputs(
        self,
        *,
        output_directory: Path,
        shank_idx: int,
        multi_shank: bool,
        channel_results: dict,
        previous_alignments: AlignmentHistory,
        ccf_channel_results: dict,
        metadata: AlignmentOutputMetadata,
        use_docdb: bool,
        output_package_directory: Path | None = None,
        mouse_id: str | None = None,
    ) -> SavedAlignmentOutputs:
        """Persist alignment output JSON files and optionally DocDB output."""
        suffix = f"_shank{shank_idx + 1}" if multi_shank else ""
        channel_results_path = output_directory / f"channel_locations{suffix}.json"
        prev_alignments_path = output_directory / f"prev_alignments{suffix}.json"
        ccf_channel_results_path = (
            output_directory / f"ccf_channel_locations{suffix}.json"
        )
        metadata_path = output_directory / f"alignment_output_metadata{suffix}.json"

        self._write_dict_to_json(channel_results_path, channel_results)
        self._write_dict_to_json(prev_alignments_path, previous_alignments)
        self._write_dict_to_json(ccf_channel_results_path, ccf_channel_results)
        self._write_dict_to_json(
            metadata_path,
            self._metadata_dict(
                metadata,
                channel_results_path=channel_results_path,
                previous_alignments_path=prev_alignments_path,
                ccf_channel_results_path=ccf_channel_results_path,
            ),
        )

        datapackage_path = None
        if output_package_directory is not None:
            datapackage_path = upsert_alignment_output_datapackage(
                output_package_directory=output_package_directory,
                metadata=metadata,
                mouse_id=mouse_id,
                channel_results_path=channel_results_path,
                previous_alignments_path=prev_alignments_path,
                ccf_channel_results_path=ccf_channel_results_path,
                metadata_path=metadata_path,
            )

        docdb_probe_name = None
        docdb_error = None
        if use_docdb:
            docdb_probe_name = f"{output_directory.stem}_{shank_idx}"
            try:
                write_output_to_docdb(
                    output_directory.parent.stem,
                    docdb_probe_name,
                    channel_results,
                    previous_alignments,
                    ccf_channel_results,
                )
            except ValueError as exc:
                docdb_error = str(exc)

        return SavedAlignmentOutputs(
            channel_results_path=channel_results_path,
            previous_alignments_path=prev_alignments_path,
            ccf_channel_results_path=ccf_channel_results_path,
            metadata_path=metadata_path,
            datapackage_path=datapackage_path,
            docdb_probe_name=docdb_probe_name,
            docdb_error=docdb_error,
        )

    @staticmethod
    def _metadata_dict(
        metadata: AlignmentOutputMetadata,
        *,
        channel_results_path: Path,
        previous_alignments_path: Path,
        ccf_channel_results_path: Path,
    ) -> dict:
        metadata_dict = {
            "schema_version": "1.1.0" if metadata.ccf_export is not None else "1.0.0",
            "recording_id": metadata.recording_id,
            "ephys_collection": metadata.ephys_collection,
            "logical_probe": metadata.logical_probe,
            "probe_id": metadata.probe_id,
            "shank_idx": metadata.shank_idx,
            "shank_id": metadata.shank_idx + 1,
            "n_shanks": metadata.n_shanks,
            "files": {
                "channel_locations": channel_results_path.name,
                "prev_alignments": previous_alignments_path.name,
                "ccf_channel_locations": ccf_channel_results_path.name,
            },
        }
        if metadata.ccf_export is not None:
            metadata_dict["ccf_export"] = asdict(metadata.ccf_export)
        return metadata_dict

    @staticmethod
    def _write_dict_to_json(file_path: Path, data_dict: dict) -> None:
        """Write a dict as stable indented JSON."""
        with open(file_path, "w") as fp:
            json.dump(data_dict, fp, indent=2, separators=(",", ": "))

    @staticmethod
    def _shank_idx_from_previous_alignment_path(path: Path) -> int | None:
        match = re.fullmatch(r"prev_alignments(?:_shank(\d+))?\.json", path.name)
        if match is None:
            return None
        shank_id = match.group(1)
        if shank_id is None:
            return 0
        return max(0, int(shank_id) - 1)
