"""Runtime ephys stream models and loading service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import one.alf.io as alfio
from numpy.typing import NDArray
from one import alf

from ephys_alignment_gui.channel_geometry import (
    n_shanks_from_geometry,
    rows_for_shank,
    valid_shank_indices,
)
from ephys_alignment_gui.datapackage_loader import (
    ChannelTablePaths,
    DataPackageError,
    ProbeInfo,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelTable:
    """Stream-level channel geometry loaded from producer-owned files."""

    local_coordinates: NDArray
    raw_ind: NDArray | None = None
    contact_ids: NDArray | None = None
    shank_indices: NDArray | None = None

    def __post_init__(self) -> None:
        local_coordinates = np.asarray(self.local_coordinates)
        if local_coordinates.ndim != 2 or local_coordinates.shape[1] < 2:
            raise ValueError(
                "local_coordinates must be a 2D array with lateral/axial columns"
            )
        object.__setattr__(self, "local_coordinates", local_coordinates)

        n_channels = local_coordinates.shape[0]
        object.__setattr__(
            self,
            "raw_ind",
            self._validate_optional_vector("raw_ind", self.raw_ind, n_channels),
        )
        object.__setattr__(
            self,
            "contact_ids",
            self._validate_optional_vector("contact_ids", self.contact_ids, n_channels),
        )
        if self.shank_indices is not None:
            shank_indices = valid_shank_indices(self.shank_indices, n_channels)
            if shank_indices is None:
                raise ValueError(
                    "shank_indices must be a 1D vector with one entry per channel"
                )
            object.__setattr__(self, "shank_indices", shank_indices)

    @property
    def n_channels(self) -> int:
        """Number of rows in the channel table."""
        return int(self.local_coordinates.shape[0])

    @property
    def n_shanks(self) -> int:
        """Number of ephys shanks represented by this stream."""
        return n_shanks_from_geometry(self.local_coordinates, self.shank_indices)

    def rows_for_shank(self, shank_idx: int) -> NDArray[np.integer[Any]]:
        """Return channel-table rows for a 0-based ephys shank."""
        if shank_idx < 0 or shank_idx >= self.n_shanks:
            raise IndexError(
                f"shank_idx {shank_idx} is outside valid range 0..{self.n_shanks - 1}"
            )
        return rows_for_shank(
            self.local_coordinates,
            self.shank_indices,
            shank_idx,
            self.n_shanks,
        )

    def local_coordinates_for_rows(self, rows: NDArray) -> NDArray:
        """Return local coordinates for channel-table row positions."""
        return self.local_coordinates[rows, :]

    def depths_for_rows(self, rows: NDArray) -> NDArray:
        """Return axial depths for channel-table row positions."""
        return self.local_coordinates_for_rows(rows)[:, 1]

    @staticmethod
    def _validate_optional_vector(
        name: str, value: NDArray | None, n_channels: int
    ) -> NDArray | None:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim != 1 or arr.shape[0] != n_channels:
            raise ValueError(f"{name} must be a 1D vector with one entry per channel")
        return arr


@dataclass(frozen=True)
class EphysStreamData:
    """Runtime data loaded for one ephys stream / collection."""

    recording_id: str
    ephys_collection: str
    ephys_dir: Path
    channel_table: ChannelTable
    alf_data: dict[str, Any]
    session_notes: str
    probe_id: str | None = None
    probe_name: str | None = None
    logical_probe: str | None = None

    @property
    def stream_key(self) -> tuple[str, str]:
        """Stable runtime key for this recording stream."""
        return self.recording_id, self.ephys_collection

    @property
    def n_shanks(self) -> int:
        """Number of ephys shanks represented by this stream."""
        return self.channel_table.n_shanks

    def channel_collection(self, shank_idx: int) -> ChannelCollectionView:
        """Return a shank/channel-collection view over this stream."""
        return ChannelCollectionView(
            stream=self,
            shank_idx=shank_idx,
            rows=self.channel_table.rows_for_shank(shank_idx),
        )


@dataclass(frozen=True)
class ChannelCollectionView:
    """Shank-scoped row view into an :class:`EphysStreamData`.

    This object owns only the row selection. Stream-level arrays remain owned by
    :class:`EphysStreamData`.
    """

    stream: EphysStreamData
    shank_idx: int
    rows: NDArray[np.integer[Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", np.asarray(self.rows, dtype=int))

    @property
    def channel_table(self) -> ChannelTable:
        """The stream-level channel table this view indexes."""
        return self.stream.channel_table

    @property
    def local_coordinates(self) -> NDArray:
        """Local channel coordinates for this collection."""
        return self.channel_table.local_coordinates_for_rows(self.rows)

    @property
    def depths(self) -> NDArray:
        """Axial channel depths for this collection."""
        return self.channel_table.depths_for_rows(self.rows)

    @property
    def contact_ids(self) -> NDArray | None:
        """Contact IDs for this collection, if the producer supplied them."""
        if self.channel_table.contact_ids is None:
            return None
        return self.channel_table.contact_ids[self.rows]

    @property
    def raw_ind(self) -> NDArray | None:
        """Raw channel indices for this collection, if supplied."""
        if self.channel_table.raw_ind is None:
            return None
        return self.channel_table.raw_ind[self.rows]

    @property
    def shank_indices(self) -> NDArray | None:
        """Producer shank indices for this collection, if supplied."""
        if self.channel_table.shank_indices is None:
            return None
        return self.channel_table.shank_indices[self.rows]


class EphysDataService:
    """Load runtime ephys stream data from resolved datapackage paths."""

    _ALF_OBJECTS: tuple[tuple[str, str], ...] = (
        ("spikes", "spikes"),
        ("clusters", "clusters"),
        ("channels", "channels"),
        ("rms_AP", "ephysTimeRmsAP"),
        ("rms_LF", "ephysTimeRmsLF"),
        ("rms_AP_main", "ephysTimeRmsAPMain"),
        ("rms_LF_main", "ephysTimeRmsLFMain"),
        ("psd_lf", "ephysSpectralDensityLF"),
        ("psd_lf_main", "ephysSpectralDensityLFMain"),
    )

    def load_channel_table(self, probe: ProbeInfo) -> ChannelTable:
        """Load the stream-level channel table for a selected probe entry."""
        paths = self._channel_table_paths(probe)
        local_coordinates = np.load(paths.local_coordinates)
        raw_ind = self._load_optional_vector(paths.raw_ind)
        contact_ids = self._load_optional_vector(paths.contact_id)

        raw_shank_indices = self._load_optional_vector(paths.shank_ind)
        shank_indices = valid_shank_indices(
            raw_shank_indices, local_coordinates.shape[0]
        )
        if raw_shank_indices is not None and shank_indices is None:
            logger.warning(
                "Ignoring invalid channels.shankInd.npy for %s: expected "
                "shape (%d,), got %s",
                probe.probe_name,
                local_coordinates.shape[0],
                np.asarray(raw_shank_indices).shape,
            )

        return ChannelTable(
            local_coordinates=local_coordinates,
            raw_ind=raw_ind,
            contact_ids=contact_ids,
            shank_indices=shank_indices,
        )

    def load_stream_data(
        self,
        probe: ProbeInfo,
        channel_table: ChannelTable | None = None,
    ) -> EphysStreamData:
        """Load ALF data for one ephys stream / collection."""
        if probe.ephys_dir is None:
            raise DataPackageError(f"Probe {probe.probe_name!r} has no ephys dir")

        channel_table = channel_table or self.load_channel_table(probe)
        ephys_dir = probe.ephys_dir
        logger.info(
            "Loading ephys stream data from %s, ephys_collection=%s",
            ephys_dir,
            probe.ephys_collection,
        )

        data = self._load_alf_objects(ephys_dir)
        self._attach_channel_table(data, channel_table)
        self._attach_optional_shank_arrays(data, ephys_dir)

        return EphysStreamData(
            recording_id=probe.recording_id,
            ephys_collection=probe.ephys_collection,
            ephys_dir=ephys_dir,
            channel_table=channel_table,
            alf_data=data,
            session_notes=self._load_session_notes(ephys_dir),
            probe_id=probe.probe_id,
            probe_name=probe.probe_name,
            logical_probe=probe.logical_probe,
        )

    def _channel_table_paths(self, probe: ProbeInfo) -> ChannelTablePaths:
        if probe.channel_table is not None:
            return probe.channel_table
        if probe.ephys_dir is None:
            raise DataPackageError(
                f"Probe {probe.probe_name!r} has no channel table or ephys dir"
            )
        return ChannelTablePaths(
            local_coordinates=probe.ephys_dir / "channels.localCoordinates.npy",
            raw_ind=probe.ephys_dir / "channels.rawInd.npy",
            contact_id=probe.ephys_dir / "channels.contactId.npy",
            shank_ind=probe.ephys_dir / "channels.shankInd.npy",
        )

    @staticmethod
    def _load_optional_vector(path: Path | None) -> NDArray | None:
        if path is None or not path.is_file():
            return None
        return np.load(path, allow_pickle=False)

    def _load_alf_objects(self, ephys_dir: Path) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for value_key, object_name in self._ALF_OBJECTS:
            try:
                data[value_key] = alfio.load_object(ephys_dir, object_name)
                data[value_key]["exists"] = True
                if "rms" in value_key:
                    data[value_key]["xaxis"] = "Time (s)"
            except alf.exceptions.ALFObjectNotFound:
                logger.warning(
                    "%s data was not found, some plots will not display", value_key
                )
                data[value_key] = {"exists": False}

        data["rf_map"] = {"exists": False}
        data["pass_stim"] = {"exists": False}
        data["gabor"] = {"exists": False}
        return data

    @staticmethod
    def _attach_channel_table(
        data: dict[str, Any],
        channel_table: ChannelTable,
    ) -> None:
        channels = data.get("channels")
        if not channels or not channels.get("exists"):
            channels = {"exists": True}
            data["channels"] = channels

        channels["localCoordinates"] = channel_table.local_coordinates
        if channel_table.raw_ind is not None:
            channels["rawInd"] = channel_table.raw_ind
        if channel_table.shank_indices is not None:
            channels["shankInd"] = channel_table.shank_indices
        if channel_table.contact_ids is not None:
            channels["contactId"] = channel_table.contact_ids

    @staticmethod
    def _attach_optional_shank_arrays(data: dict[str, Any], ephys_dir: Path) -> None:
        shank_indices_file = ephys_dir / "spike_shank_indices.npy"
        if shank_indices_file.exists():
            data["spike_shanks"] = np.load(shank_indices_file)

        unit_shank_indices_file = ephys_dir / "unit_shank_indices.npy"
        if unit_shank_indices_file.exists():
            data["unit_shank_indices"] = np.load(unit_shank_indices_file)

    @staticmethod
    def _load_session_notes(ephys_dir: Path) -> str:
        notes_file = ephys_dir / "session_notes.txt"
        if notes_file.exists():
            return notes_file.read_text()
        return "No notes for this session"
