"""Lightweight immutable view of input-dataset facts needed for saving."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ephys_alignment_gui.io.datapackage_loader import (
    ChannelTablePaths,
    DataPackageError,
    HistologyImagePaths,
    MouseRoot,
    ProbeInfo,
    TransformPaths,
    XyzPicks,
)

StreamKey = tuple[str, str]


@dataclass(frozen=True)
class MissingInputPath:
    """A save-critical input path is absent from the selected dataset."""

    recording_id: str
    ephys_collection: str
    role: str
    path: Path | None


@dataclass(frozen=True)
class InputProbeSnapshot:
    """Normalized lightweight metadata for one selectable ephys stream."""

    probe_id: str
    probe_name: str
    recording_id: str
    logical_probe: str
    ephys_collection: str
    num_shanks: int
    ephys_dir: Path | None
    channel_table: ChannelTablePaths | None
    xyz_picks: tuple[XyzPicks, ...]

    @classmethod
    def from_probe(cls, probe: ProbeInfo) -> InputProbeSnapshot:
        """Return a snapshot of one resolved datapackage probe entry."""
        return cls(
            probe_id=probe.probe_id,
            probe_name=probe.probe_name,
            recording_id=probe.recording_id,
            logical_probe=probe.logical_probe,
            ephys_collection=probe.ephys_collection,
            num_shanks=probe.num_shanks,
            ephys_dir=probe.ephys_dir,
            channel_table=probe.channel_table,
            xyz_picks=probe.xyz_picks,
        )

    @property
    def stream_key(self) -> StreamKey:
        """Return the stable ephys stream key for this probe."""
        return self.recording_id, self.ephys_collection

    def picks_for_shank(self, shank_idx: int) -> XyzPicks:
        """Return the xyz-picks entry for a given 0-based shank index."""
        if self.num_shanks == 1:
            return self.xyz_picks[0]

        for field_name in ("ephys_shank", "shank"):
            picks_by_index = self._picks_by_normalized_shank_field(field_name)
            if picks_by_index and shank_idx in picks_by_index:
                return picks_by_index[shank_idx]

        want = shank_idx + 1
        raise DataPackageError(
            f"Probe {self.probe_name!r} has no shank {want} "
            f"(shanks available: {[pk.shank for pk in self.xyz_picks]})"
        )

    def _picks_by_normalized_shank_field(
        self,
        field_name: str,
    ) -> dict[int, XyzPicks]:
        values = [getattr(pk, field_name) for pk in self.xyz_picks]
        if any(value is None for value in values):
            return {}
        unique_values = sorted(set(int(value) for value in values))
        value_to_local = {value: idx for idx, value in enumerate(unique_values)}
        picks_by_index: dict[int, XyzPicks] = {}
        for pick, value in zip(self.xyz_picks, values, strict=True):
            local_idx = value_to_local[int(value)]
            if local_idx in picks_by_index:
                raise DataPackageError(
                    f"Probe {self.probe_name!r} has duplicate {field_name} "
                    f"value {value!r} in xyz_picks"
                )
            picks_by_index[local_idx] = pick
        return picks_by_index

    def missing_save_critical_paths(self) -> tuple[MissingInputPath, ...]:
        """Return absent channel-table paths needed for lightweight saving."""
        if self.channel_table is None:
            return (
                MissingInputPath(
                    recording_id=self.recording_id,
                    ephys_collection=self.ephys_collection,
                    role="channel_table",
                    path=None,
                ),
            )

        checks = {
            "channel_table.local_coordinates": self.channel_table.local_coordinates,
            "channel_table.raw_ind": self.channel_table.raw_ind,
            "channel_table.shank_ind": self.channel_table.shank_ind,
        }
        return tuple(
            MissingInputPath(
                recording_id=self.recording_id,
                ephys_collection=self.ephys_collection,
                role=role,
                path=path,
            )
            for role, path in checks.items()
            if not path.exists()
        )


@dataclass(frozen=True)
class InputDatasetSnapshot:
    """Normalized mouse-root datapackage facts, independent of GUI state."""

    root: Path
    schema_version: str
    mouse_id: str
    transforms: TransformPaths | None
    histology: HistologyImagePaths | None
    probes: tuple[InputProbeSnapshot, ...]

    @classmethod
    def from_mouse_root(cls, mouse_root: MouseRoot) -> InputDatasetSnapshot:
        """Return a snapshot for a loaded mouse-root datapackage."""
        probes = tuple(
            InputProbeSnapshot.from_probe(probe)
            for _recording_id, probes_for_recording in sorted(mouse_root.probes.items())
            for _probe_name, probe in sorted(probes_for_recording.items())
        )
        return cls(
            root=mouse_root.root,
            schema_version=mouse_root.schema_version,
            mouse_id=mouse_root.mouse_id,
            transforms=mouse_root.transforms,
            histology=mouse_root.histology,
            probes=probes,
        )

    @property
    def sessions(self) -> tuple[str, ...]:
        """Return recording IDs represented by the snapshot."""
        return tuple(sorted({probe.recording_id for probe in self.probes}))

    @property
    def stream_keys(self) -> tuple[StreamKey, ...]:
        """Return all stable ephys stream keys represented by the snapshot."""
        return tuple(probe.stream_key for probe in self.probes)

    def probes_for_session(self, recording_id: str) -> tuple[str, ...]:
        """Return selectable probe names for one recording."""
        return tuple(
            probe.probe_name
            for probe in self.probes
            if probe.recording_id == recording_id
        )

    def probe_for_stream_key(
        self,
        recording_id: str,
        ephys_collection: str,
    ) -> InputProbeSnapshot:
        """Return the unique probe snapshot for one ephys stream key."""
        matches = [
            probe
            for probe in self.probes
            if probe.recording_id == recording_id
            and probe.ephys_collection == ephys_collection
        ]
        if not matches:
            raise KeyError(
                "No input probe metadata found for stream "
                f"{recording_id!r}/{ephys_collection!r}"
            )
        if len(matches) > 1:
            probe_names = sorted(probe.probe_name for probe in matches)
            raise KeyError(
                "Multiple input probes map to stream "
                f"{recording_id!r}/{ephys_collection!r}: {probe_names}"
            )
        return matches[0]

    def missing_save_critical_paths(self) -> tuple[MissingInputPath, ...]:
        """Return all absent save-critical paths for the mouse root."""
        return tuple(
            missing
            for probe in self.probes
            for missing in probe.missing_save_critical_paths()
        )
