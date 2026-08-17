"""Serializable document snapshots for cheap alignment recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_state import (
    AlignmentSaveState,
    AlignmentState,
    PendingReferenceLines,
)
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AlignmentKeySnapshot:
    """JSON-serializable form of an ``AlignmentKey``."""

    recording_id: str
    ephys_collection: str
    shank_idx: int

    @classmethod
    def from_key(cls, key: AlignmentKey) -> AlignmentKeySnapshot:
        """Return a snapshot for one alignment key."""
        return cls(
            recording_id=key.recording_id,
            ephys_collection=key.ephys_collection,
            shank_idx=key.shank_idx,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentKeySnapshot:
        """Return a snapshot from a JSON dictionary."""
        return cls(
            recording_id=str(data["recording_id"]),
            ephys_collection=str(data["ephys_collection"]),
            shank_idx=int(data["shank_idx"]),
        )

    def to_key(self) -> AlignmentKey:
        """Return the core alignment key."""
        return AlignmentKey(
            self.recording_id,
            self.ephys_collection,
            self.shank_idx,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "recording_id": self.recording_id,
            "ephys_collection": self.ephys_collection,
            "shank_idx": self.shank_idx,
        }


@dataclass(frozen=True)
class ActiveAlignmentSnapshot:
    """JSON-serializable form of the active alignment control points."""

    feature: list[float]
    track: list[float]
    lin_fit: bool

    @classmethod
    def from_alignment(
        cls,
        alignment: ActiveAlignment,
    ) -> ActiveAlignmentSnapshot:
        """Return a snapshot for one active alignment."""
        return cls(
            feature=_float_list(alignment.feature),
            track=_float_list(alignment.track),
            lin_fit=bool(alignment.lin_fit),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActiveAlignmentSnapshot:
        """Return a snapshot from a JSON dictionary."""
        return cls(
            feature=_float_list(data["feature"]),
            track=_float_list(data["track"]),
            lin_fit=bool(data.get("lin_fit", False)),
        )

    def to_alignment(self) -> ActiveAlignment:
        """Return the core active alignment value object."""
        return ActiveAlignment(
            np.asarray(self.feature, dtype=float),
            np.asarray(self.track, dtype=float),
            lin_fit=self.lin_fit,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "feature": list(self.feature),
            "track": list(self.track),
            "lin_fit": self.lin_fit,
        }


@dataclass(frozen=True)
class PendingReferenceLinesSnapshot:
    """JSON-serializable pending reference-line positions."""

    feature_positions_um: list[float]
    warped_positions_um: list[float]

    @classmethod
    def from_lines(
        cls,
        lines: PendingReferenceLines,
    ) -> PendingReferenceLinesSnapshot:
        """Return a snapshot for pending reference lines."""
        return cls(
            feature_positions_um=_float_list(lines.feature_positions_um),
            warped_positions_um=_float_list(lines.warped_positions_um),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingReferenceLinesSnapshot:
        """Return a snapshot from a JSON dictionary."""
        return cls(
            feature_positions_um=_float_list(data["feature_positions_um"]),
            warped_positions_um=_float_list(data["warped_positions_um"]),
        )

    def to_lines(self) -> PendingReferenceLines:
        """Return core pending reference lines."""
        return PendingReferenceLines(
            np.asarray(self.feature_positions_um, dtype=float),
            np.asarray(self.warped_positions_um, dtype=float),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "feature_positions_um": list(self.feature_positions_um),
            "warped_positions_um": list(self.warped_positions_um),
        }


@dataclass(frozen=True)
class AlignmentSaveStateSnapshot:
    """JSON-serializable save revision metadata."""

    revision: int
    saved_revision: int
    saved_signature: dict[str, Any] | None

    @classmethod
    def from_save_state(
        cls,
        save_state: AlignmentSaveState,
    ) -> AlignmentSaveStateSnapshot:
        """Return a snapshot for save revision state."""
        return cls(
            revision=int(save_state.revision),
            saved_revision=int(save_state.saved_revision),
            saved_signature=_signature_to_dict(save_state.saved_signature),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentSaveStateSnapshot:
        """Return a snapshot from a JSON dictionary."""
        return cls(
            revision=int(data.get("revision", 0)),
            saved_revision=int(data.get("saved_revision", 0)),
            saved_signature=data.get("saved_signature"),
        )

    def to_save_state(self) -> AlignmentSaveState:
        """Return core save revision state."""
        return AlignmentSaveState(
            revision=self.revision,
            saved_revision=self.saved_revision,
            saved_signature=_signature_from_dict(self.saved_signature),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "revision": self.revision,
            "saved_revision": self.saved_revision,
            "saved_signature": self.saved_signature,
        }


@dataclass(frozen=True)
class AlignmentStateSnapshot:
    """JSON-serializable per-key alignment state."""

    key: AlignmentKeySnapshot
    max_idx: int
    alignments: dict[str, list[list[float]]]
    feature_prev: list[float] | None
    track_prev: list[float] | None
    active_alignment: ActiveAlignmentSnapshot | None
    pending_reference_lines: PendingReferenceLinesSnapshot | None
    save_state: AlignmentSaveStateSnapshot

    @classmethod
    def from_state(
        cls,
        key: AlignmentKey,
        state: AlignmentState,
    ) -> AlignmentStateSnapshot:
        """Return a snapshot for one document alignment state."""
        active_alignment = state.active_alignment
        pending_reference_lines = state.pending_reference_lines
        return cls(
            key=AlignmentKeySnapshot.from_key(key),
            max_idx=int(state.max_idx),
            alignments=_alignment_history_to_lists(state.alignments),
            feature_prev=_optional_float_list(state.feature_prev),
            track_prev=_optional_float_list(state.track_prev),
            active_alignment=(
                ActiveAlignmentSnapshot.from_alignment(active_alignment)
                if active_alignment is not None
                else None
            ),
            pending_reference_lines=(
                PendingReferenceLinesSnapshot.from_lines(pending_reference_lines)
                if pending_reference_lines is not None
                else None
            ),
            save_state=AlignmentSaveStateSnapshot.from_save_state(state.save_state),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentStateSnapshot:
        """Return a snapshot from a JSON dictionary."""
        active_data = data.get("active_alignment")
        lines_data = data.get("pending_reference_lines")
        return cls(
            key=AlignmentKeySnapshot.from_dict(data["key"]),
            max_idx=int(data.get("max_idx", 10)),
            alignments=_alignment_history_to_lists(data.get("alignments", {})),
            feature_prev=_optional_float_list(data.get("feature_prev")),
            track_prev=_optional_float_list(data.get("track_prev")),
            active_alignment=(
                ActiveAlignmentSnapshot.from_dict(active_data)
                if active_data is not None
                else None
            ),
            pending_reference_lines=(
                PendingReferenceLinesSnapshot.from_dict(lines_data)
                if lines_data is not None
                else None
            ),
            save_state=AlignmentSaveStateSnapshot.from_dict(
                data.get("save_state", {}),
            ),
        )

    def to_state(self) -> AlignmentState:
        """Return core alignment state."""
        state = AlignmentState(max_idx=self.max_idx)
        state.set_alignments(self.alignments)
        state.feature_prev = _optional_array(self.feature_prev)
        state.track_prev = _optional_array(self.track_prev)
        state.save_state = self.save_state.to_save_state()
        if self.active_alignment is not None:
            state.active_alignment = self.active_alignment.to_alignment()
        if self.pending_reference_lines is not None:
            state.set_pending_reference_lines(
                self.pending_reference_lines.to_lines()
            )
        return state

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "key": self.key.to_dict(),
            "max_idx": self.max_idx,
            "alignments": _alignment_history_to_lists(self.alignments),
            "feature_prev": self.feature_prev,
            "track_prev": self.track_prev,
            "active_alignment": (
                self.active_alignment.to_dict()
                if self.active_alignment is not None
                else None
            ),
            "pending_reference_lines": (
                self.pending_reference_lines.to_dict()
                if self.pending_reference_lines is not None
                else None
            ),
            "save_state": self.save_state.to_dict(),
        }


@dataclass(frozen=True)
class AlignmentDocumentSnapshot:
    """JSON-serializable document snapshot for autosave/checkpoint recovery."""

    schema_version: int
    mouse_root: str | None
    mouse_id: str | None
    selected_recording: str | None
    selected_probe: str | None
    selected_shank: int
    selected_alignment_key: AlignmentKeySnapshot | None
    output_root: str | None
    output_package_directory: str | None
    output_directory: str | None
    channel_info_loaded: bool
    data_loaded: bool
    dirty: bool
    alignment_states: tuple[AlignmentStateSnapshot, ...]

    @classmethod
    def from_document(
        cls,
        document: AlignmentDocument,
    ) -> AlignmentDocumentSnapshot:
        """Return a serializable snapshot of document-owned state."""
        return cls(
            schema_version=SNAPSHOT_SCHEMA_VERSION,
            mouse_root=_path_to_str(document.mouse_root),
            mouse_id=document.mouse_id,
            selected_recording=document.selected_recording,
            selected_probe=document.selected_probe,
            selected_shank=int(document.selected_shank),
            selected_alignment_key=(
                AlignmentKeySnapshot.from_key(document.selected_alignment_key)
                if document.selected_alignment_key is not None
                else None
            ),
            output_root=_path_to_str(document.output_root),
            output_package_directory=_path_to_str(
                document.output_package_directory
            ),
            output_directory=_path_to_str(document.output_directory),
            channel_info_loaded=bool(document.channel_info_loaded),
            data_loaded=bool(document.data_loaded),
            dirty=bool(document.dirty),
            alignment_states=tuple(
                AlignmentStateSnapshot.from_state(key, state)
                for key, state in sorted(
                    document.alignment_states.items(),
                    key=lambda item: (
                        item[0].recording_id,
                        item[0].ephys_collection,
                        item[0].shank_idx,
                    ),
                )
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentDocumentSnapshot:
        """Return a snapshot from a JSON dictionary."""
        schema_version = int(data.get("schema_version", 0))
        if schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported alignment document snapshot schema "
                f"{schema_version}"
            )
        selected_key_data = data.get("selected_alignment_key")
        return cls(
            schema_version=schema_version,
            mouse_root=data.get("mouse_root"),
            mouse_id=data.get("mouse_id"),
            selected_recording=data.get("selected_recording"),
            selected_probe=data.get("selected_probe"),
            selected_shank=int(data.get("selected_shank", 0)),
            selected_alignment_key=(
                AlignmentKeySnapshot.from_dict(selected_key_data)
                if selected_key_data is not None
                else None
            ),
            output_root=data.get("output_root"),
            output_package_directory=data.get("output_package_directory"),
            output_directory=data.get("output_directory"),
            channel_info_loaded=bool(data.get("channel_info_loaded", False)),
            data_loaded=bool(data.get("data_loaded", False)),
            dirty=bool(data.get("dirty", False)),
            alignment_states=tuple(
                AlignmentStateSnapshot.from_dict(item)
                for item in data.get("alignment_states", ())
            ),
        )

    @classmethod
    def read_json(cls, path: Path) -> AlignmentDocumentSnapshot:
        """Read a document snapshot from one JSON file."""
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("Alignment document snapshot must be a JSON object")
        return cls.from_dict(data)

    def restore_document(self) -> AlignmentDocument:
        """Return a new document restored from this snapshot."""
        document = AlignmentDocument(
            mouse_root=_optional_path(self.mouse_root),
            mouse_id=self.mouse_id,
            selected_recording=self.selected_recording,
            selected_probe=self.selected_probe,
            selected_shank=self.selected_shank,
            selected_alignment_key=(
                self.selected_alignment_key.to_key()
                if self.selected_alignment_key is not None
                else None
            ),
            output_root=_optional_path(self.output_root),
            output_package_directory=_optional_path(
                self.output_package_directory
            ),
            output_directory=_optional_path(self.output_directory),
            channel_info_loaded=self.channel_info_loaded,
            data_loaded=self.data_loaded,
            dirty=self.dirty,
        )
        document.alignment_states = {
            state_snapshot.key.to_key(): state_snapshot.to_state()
            for state_snapshot in self.alignment_states
        }
        return document

    def write_json(self, path: Path) -> None:
        """Atomically write this snapshot to one JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        tmp_path.replace(path)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "mouse_root": self.mouse_root,
            "mouse_id": self.mouse_id,
            "selected_recording": self.selected_recording,
            "selected_probe": self.selected_probe,
            "selected_shank": self.selected_shank,
            "selected_alignment_key": (
                self.selected_alignment_key.to_dict()
                if self.selected_alignment_key is not None
                else None
            ),
            "output_root": self.output_root,
            "output_package_directory": self.output_package_directory,
            "output_directory": self.output_directory,
            "channel_info_loaded": self.channel_info_loaded,
            "data_loaded": self.data_loaded,
            "dirty": self.dirty,
            "alignment_states": [
                state.to_dict() for state in self.alignment_states
            ],
        }


def _alignment_history_to_lists(
    alignments: dict[str, Any],
) -> dict[str, list[list[float]]]:
    return {
        str(label): [
            _float_list(values[0]),
            _float_list(values[1]),
        ]
        for label, values in alignments.items()
    }


def _float_list(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).tolist()]


def _optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    return _float_list(value)


def _optional_array(value: list[float] | None) -> np.ndarray[Any, Any] | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _path_to_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value is not None else None


def _signature_to_dict(signature: Any | None) -> dict[str, Any] | None:
    if signature is None:
        return None
    feature, track, lin_fit = signature
    return {
        "feature": _float_list(feature),
        "track": _float_list(track),
        "lin_fit": bool(lin_fit),
    }


def _signature_from_dict(data: dict[str, Any] | None) -> Any | None:
    if data is None:
        return None
    return (
        tuple(_float_list(data["feature"])),
        tuple(_float_list(data["track"])),
        bool(data.get("lin_fit", False)),
    )
