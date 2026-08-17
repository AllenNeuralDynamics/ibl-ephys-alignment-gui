"""Lightweight save geometry catalog keyed by alignment document keys."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.core.alignment_output import (
    AlignmentOutputMetadata,
    ChannelOutputIdentity,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.io.input_dataset_snapshot import (
    InputDatasetSnapshot,
    InputProbeSnapshot,
    MissingInputPath,
    StreamKey,
)
from ephys_alignment_gui.services.ephys_data import ChannelTable


class SaveGeometryError(RuntimeError):
    """Raised when lightweight save geometry cannot be resolved."""


@dataclass(frozen=True)
class SaveGeometry:
    """Channel geometry and identity needed to build one saved output."""

    key: AlignmentKey
    probe: InputProbeSnapshot
    channel_coordinates: NDArray[Any]
    channel_depths_um: NDArray[Any]
    channel_identity: ChannelOutputIdentity
    output_metadata: AlignmentOutputMetadata
    multi_shank: bool


@dataclass
class SaveGeometryCatalog:
    """Load and cache save-critical channel geometry without stream runtimes."""

    input_dataset: InputDatasetSnapshot | None = None
    _channel_table_by_stream: dict[StreamKey, ChannelTable] = field(
        default_factory=dict
    )
    _geometry_by_key: dict[AlignmentKey, SaveGeometry] = field(default_factory=dict)

    def set_input_dataset(self, input_dataset: InputDatasetSnapshot | None) -> None:
        """Replace the input dataset snapshot and clear cached geometry."""
        if input_dataset is self.input_dataset:
            return
        self.input_dataset = input_dataset
        self.clear()

    def clear(self) -> None:
        """Clear all cached channel tables and shank geometry."""
        self._channel_table_by_stream.clear()
        self._geometry_by_key.clear()

    def geometry_for_key(self, key: AlignmentKey) -> SaveGeometry:
        """Return cached save geometry for one alignment key."""
        if key in self._geometry_by_key:
            return self._geometry_by_key[key]
        input_dataset = self._require_input_dataset()
        probe = self._probe_for_key(input_dataset, key)
        self._raise_for_missing_paths(probe.missing_save_critical_paths())
        channel_table = self._channel_table_for_probe(probe)
        try:
            rows = channel_table.rows_for_shank(key.shank_idx)
        except Exception as exc:
            raise SaveGeometryError(
                "Failed to select save geometry rows for "
                f"{key.recording_id}/{key.ephys_collection} shank "
                f"{key.shank_idx + 1}: {exc}"
            ) from exc

        geometry = SaveGeometry(
            key=key,
            probe=probe,
            channel_coordinates=channel_table.local_coordinates_for_rows(rows),
            channel_depths_um=channel_table.depths_for_rows(rows),
            channel_identity=self._channel_identity(
                channel_table,
                rows,
                default_shank_idx=key.shank_idx,
            ),
            output_metadata=self._output_metadata(
                key,
                probe=probe,
                n_shanks=channel_table.n_shanks,
            ),
            multi_shank=channel_table.n_shanks > 1,
        )
        self._geometry_by_key[key] = geometry
        return geometry

    def _require_input_dataset(self) -> InputDatasetSnapshot:
        if self.input_dataset is None:
            raise SaveGeometryError("No input dataset snapshot is loaded.")
        return self.input_dataset

    @staticmethod
    def _probe_for_key(
        input_dataset: InputDatasetSnapshot,
        key: AlignmentKey,
    ) -> InputProbeSnapshot:
        try:
            return input_dataset.probe_for_stream_key(
                key.recording_id,
                key.ephys_collection,
            )
        except KeyError as exc:
            raise SaveGeometryError(str(exc)) from exc

    @staticmethod
    def _raise_for_missing_paths(
        missing_paths: tuple[MissingInputPath, ...],
    ) -> None:
        if not missing_paths:
            return
        details = ", ".join(
            f"{item.role}={_path_label(item.path)}" for item in missing_paths
        )
        first = missing_paths[0]
        raise SaveGeometryError(
            "Missing save-critical channel geometry for "
            f"{first.recording_id}/{first.ephys_collection}: {details}"
        )

    def _channel_table_for_probe(self, probe: InputProbeSnapshot) -> ChannelTable:
        stream_key = probe.stream_key
        if stream_key not in self._channel_table_by_stream:
            self._channel_table_by_stream[stream_key] = self._load_channel_table(probe)
        return self._channel_table_by_stream[stream_key]

    @staticmethod
    def _load_channel_table(probe: InputProbeSnapshot) -> ChannelTable:
        paths = probe.channel_table
        if paths is None:
            raise SaveGeometryError(
                "Missing save-critical channel geometry for "
                f"{probe.recording_id}/{probe.ephys_collection}: channel_table=None"
            )
        try:
            return ChannelTable(
                local_coordinates=np.load(paths.local_coordinates, allow_pickle=False),
                raw_ind=np.load(paths.raw_ind, allow_pickle=False),
                contact_ids=_load_optional_vector(paths.contact_id),
                shank_indices=np.load(paths.shank_ind, allow_pickle=False),
            )
        except Exception as exc:
            raise SaveGeometryError(
                "Failed to load save-critical channel geometry for "
                f"{probe.recording_id}/{probe.ephys_collection}: {exc}"
            ) from exc

    @staticmethod
    def _channel_identity(
        channel_table: ChannelTable,
        rows: NDArray[Any],
        *,
        default_shank_idx: int,
    ) -> ChannelOutputIdentity:
        raw_ind = (
            channel_table.raw_ind[rows]
            if channel_table.raw_ind is not None
            else np.asarray(rows, dtype=int).copy()
        )
        shank_idx = (
            channel_table.shank_indices[rows]
            if channel_table.shank_indices is not None
            else np.full(np.asarray(rows).shape, default_shank_idx, dtype=int)
        )
        contact_id = (
            channel_table.contact_ids[rows]
            if channel_table.contact_ids is not None
            else None
        )
        return ChannelOutputIdentity(
            raw_ind=raw_ind,
            contact_id=contact_id,
            shank_idx=shank_idx,
        )

    @staticmethod
    def _output_metadata(
        key: AlignmentKey,
        *,
        probe: InputProbeSnapshot,
        n_shanks: int,
    ) -> AlignmentOutputMetadata:
        return AlignmentOutputMetadata(
            recording_id=key.recording_id,
            ephys_collection=key.ephys_collection,
            logical_probe=probe.logical_probe or probe.probe_name,
            shank_idx=key.shank_idx,
            n_shanks=int(n_shanks),
            probe_id=probe.probe_id,
        )


def _load_optional_vector(path: Path | None) -> NDArray[Any] | None:
    if path is None or not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def _path_label(path: Path | None) -> str:
    return "<missing channel table>" if path is None else str(path)
