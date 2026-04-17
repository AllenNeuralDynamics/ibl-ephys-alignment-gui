"""Resolve paths from a preprocessed mouse-root directory.

The GUI takes a mouse-root directory containing ``datapackage.json`` (produced
by ``aind-ibl-ephys-alignment-preprocessing`` v1.1.0+) and reads every path it
needs from there. No directory-structure assumptions, no platform-specific
literals, no globbing of sibling assets.

Transform paths in the datapackage are relative to the mouse root and may use
``..`` to reach sibling assets (e.g. the SmartSPIM asset mounted next to the
preprocessed output).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum datapackage schema this loader can read.
MIN_SCHEMA_MAJOR = 1
MIN_SCHEMA_MINOR = 1


class DataPackageError(RuntimeError):
    """Raised when a mouse root is missing or the datapackage is malformed."""


@dataclass(frozen=True)
class TransformPaths:
    """Absolute paths to the 4 ANTs transforms in the chain."""

    image_to_template_affine: Path
    image_to_template_warp: Path
    template_to_ccf_affine: Path
    template_to_ccf_warp: Path


@dataclass(frozen=True)
class HistologyImagePaths:
    """Absolute paths to image-space histology volumes."""

    registration: Path
    registration_pipeline: Path
    ccf_template: Path
    labels: Path
    additional_channels: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class XyzPicks:
    """Absolute paths to xyz-picks JSON files for one shank (or whole probe)."""

    image_space: Path
    ccf: Path
    shank: int | None


@dataclass(frozen=True)
class ProbeInfo:
    """Resolved metadata and paths for a single probe."""

    probe_id: str
    probe_name: str
    recording_id: str
    num_shanks: int
    ephys_dir: Path | None
    xyz_picks: tuple[XyzPicks, ...]

    def picks_for_shank(self, shank_idx: int) -> XyzPicks:
        """Return the xyz-picks entry for a given 0-based shank index."""
        if self.num_shanks == 1:
            return self.xyz_picks[0]
        want = shank_idx + 1  # datapackage uses 1-based `shank` field
        for pk in self.xyz_picks:
            if pk.shank == want:
                return pk
        raise DataPackageError(
            f"Probe {self.probe_name!r} has no shank {want} "
            f"(shanks available: {[pk.shank for pk in self.xyz_picks]})"
        )


@dataclass(frozen=True)
class MouseRoot:
    """Resolved view of a preprocessed mouse output directory."""

    root: Path
    schema_version: str
    mouse_id: str
    transforms: TransformPaths
    histology: HistologyImagePaths
    probes: dict[str, ProbeInfo]

    @property
    def sessions(self) -> list[str]:
        """All recording IDs represented in this mouse root (sorted)."""
        return sorted({p.recording_id for p in self.probes.values()})

    def probes_for_session(self, recording_id: str) -> list[str]:
        """Probe names for a given recording ID (sorted)."""
        names = [
            name for name, p in self.probes.items() if p.recording_id == recording_id
        ]
        return sorted(names)

    def get_probe(self, recording_id: str, probe_name: str) -> ProbeInfo:
        """Look up a probe by (recording_id, probe_name)."""
        if probe_name not in self.probes:
            raise DataPackageError(f"No probe named {probe_name!r} in datapackage")
        probe = self.probes[probe_name]
        if probe.recording_id != recording_id:
            raise DataPackageError(
                f"Probe {probe_name!r} belongs to recording "
                f"{probe.recording_id!r}, not {recording_id!r}"
            )
        return probe


def load_mouse_root(mouse_root: Path) -> MouseRoot:
    """Load and validate a mouse root directory.

    Parameters
    ----------
    mouse_root : Path
        Directory containing ``datapackage.json``.

    Returns
    -------
    MouseRoot
        Resolved view with absolute paths for transforms, histology, and probes.

    Raises
    ------
    DataPackageError
        If ``datapackage.json`` is missing, malformed, or from an incompatible
        schema version.
    """
    mouse_root = Path(mouse_root)
    dp_path = mouse_root / "datapackage.json"
    if not dp_path.is_file():
        raise DataPackageError(
            f"No datapackage.json in {mouse_root}. "
            "Expected output of aind-ibl-ephys-alignment-preprocessing v1.1.0+."
        )

    try:
        raw = json.loads(dp_path.read_text())
    except json.JSONDecodeError as e:
        raise DataPackageError(f"Malformed {dp_path}: {e}") from e

    _check_schema_version(raw.get("schema_version", ""))

    try:
        transforms = _parse_transforms(raw["transforms"], mouse_root)
        histology = _parse_histology(raw["histology"], mouse_root)
        probes = _parse_probes(raw["probes"], mouse_root)
    except KeyError as e:
        raise DataPackageError(f"datapackage.json missing required key: {e}") from e

    return MouseRoot(
        root=mouse_root,
        schema_version=raw["schema_version"],
        mouse_id=raw["mouse_id"],
        transforms=transforms,
        histology=histology,
        probes=probes,
    )


def _check_schema_version(version: str) -> None:
    if not version:
        raise DataPackageError("datapackage.json has no schema_version")
    try:
        major, minor = (int(p) for p in version.split(".")[:2])
    except ValueError as e:
        raise DataPackageError(f"Invalid schema_version {version!r}: {e}") from e
    if major != MIN_SCHEMA_MAJOR or minor < MIN_SCHEMA_MINOR:
        raise DataPackageError(
            f"Unsupported datapackage schema {version}. "
            f"GUI requires {MIN_SCHEMA_MAJOR}.{MIN_SCHEMA_MINOR}.x or newer "
            "(re-run preprocessing)."
        )


def _resolve(rel: str, root: Path) -> Path:
    """Resolve a datapackage-relative POSIX path to an absolute Path."""
    return (root / rel).resolve()


def _parse_transforms(d: dict[str, str], root: Path) -> TransformPaths:
    return TransformPaths(
        image_to_template_affine=_resolve(d["image_to_template_affine"], root),
        image_to_template_warp=_resolve(d["image_to_template_warp"], root),
        template_to_ccf_affine=_resolve(d["template_to_ccf_affine"], root),
        template_to_ccf_warp=_resolve(d["template_to_ccf_warp"], root),
    )


def _parse_histology(d: dict, root: Path) -> HistologyImagePaths:
    img = d["image_space"]
    additional = {
        Path(rel).stem: _resolve(rel, root)
        for rel in img.get("additional_channels", [])
    }
    return HistologyImagePaths(
        registration=_resolve(img["registration"], root),
        registration_pipeline=_resolve(img["registration_pipeline"], root),
        ccf_template=_resolve(img["ccf_template"], root),
        labels=_resolve(img["labels"], root),
        additional_channels=additional,
    )


def _parse_probes(d: dict, root: Path) -> dict[str, ProbeInfo]:
    probes: dict[str, ProbeInfo] = {}
    for probe_name, entry in d.items():
        picks = tuple(
            XyzPicks(
                image_space=_resolve(p["image_space"], root),
                ccf=_resolve(p["ccf"], root),
                shank=p.get("shank"),
            )
            for p in entry["xyz_picks"]
        )
        ephys_rel = entry.get("ephys")
        probes[probe_name] = ProbeInfo(
            probe_id=entry["probe_id"],
            probe_name=probe_name,
            recording_id=entry["recording_id"],
            num_shanks=entry["num_shanks"],
            ephys_dir=_resolve(ephys_rel, root) if ephys_rel else None,
            xyz_picks=picks,
        )
    return probes
