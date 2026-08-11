"""Resolve paths from a preprocessed mouse-root directory.

The GUI takes a mouse-root directory containing ``datapackage.json`` (produced
by ``aind-ibl-ephys-alignment-preprocessing`` v1.1.0+) and reads every path it
needs from there. No directory-structure assumptions, no platform-specific
literals, no globbing of sibling assets.

Datapackage schema 3.x stores each path as ``{asset, path}``. ``asset=None``
means the path is relative to the datapackage directory; non-null assets are
looked up in the datapackage's ``external_assets`` registry and resolved via
runtime configuration supplied by the deployment/user.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ephys_alignment_gui.io.datapackage_schema import (
    DatapackageContractError,
    validate_datapackage_contract,
)


class DataPackageError(RuntimeError):
    """Raised when a mouse root is missing or the datapackage is malformed."""


class AssetNotFound(DataPackageError):
    """Raised when an external asset cannot be located on this machine."""


class ReferenceNotFound(DataPackageError):
    """Raised when a located asset does not contain the requested path."""


@dataclass(frozen=True)
class AssetRef:
    """External asset registry entry plus its logical key."""

    key: str
    role: str
    name: str
    id: str | None = None
    uri: str | None = None
    checksum: str | None = None


@dataclass(frozen=True)
class PathRef:
    """Datapackage path reference before local resolution."""

    asset: str | None
    path: str


class RootSearchResolver:
    """Resolve datapackage path references against local roots/overrides."""

    def __init__(
        self,
        datapackage_dir: Path,
        asset_roots: Sequence[Path] = (),
        asset_overrides: Mapping[str, Path] | None = None,
    ) -> None:
        self.datapackage_dir = Path(datapackage_dir)
        self.asset_roots = [Path(p) for p in asset_roots]
        self.asset_overrides = {
            str(k): Path(v) for k, v in (asset_overrides or {}).items()
        }

    def resolve(self, asset: AssetRef | None, within: str) -> Path:
        """Resolve *within* inside *asset* or the datapackage directory."""
        if asset is None:
            return (self.datapackage_dir / within).resolve()

        for override_key in (asset.key, asset.name):
            if override_key and override_key in self.asset_overrides:
                path = self.asset_overrides[override_key] / within
                if path.exists():
                    return path.resolve()
                raise ReferenceNotFound(
                    f"Asset override {override_key!r} for {asset.name!r} exists, "
                    f"but requested path {within!r} was not found at {path}."
                )

        searched: list[str] = []
        for root in self.asset_roots:
            for key in (asset.name, asset.id):
                if key:
                    candidate = root / key / within
                    searched.append(str(candidate))
                    if candidate.exists():
                        return candidate.resolve()

        identity = ", ".join(
            part
            for part in (
                f"key={asset.key!r}",
                f"role={asset.role!r}",
                f"name={asset.name!r}",
                f"id={asset.id!r}" if asset.id else "",
                f"uri={asset.uri!r}" if asset.uri else "",
            )
            if part
        )
        raise AssetNotFound(
            f"Could not resolve external asset ({identity}) for path {within!r}. "
            f"Searched: {searched or ['<no asset roots configured>']}. "
            "Set IBL_ASSET_ROOTS or IBL_ASSET_OVERRIDES."
        )


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
    registration_pipeline: Path | None
    registration_pipeline_geometry: Path | None
    ccf_template: Path
    labels: Path
    additional_channels: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class XyzPicks:
    """Absolute paths to xyz-picks JSON files for one shank (or whole probe).

    ``ccf`` is a QC-only output (written only when the pipeline ran with
    ``emit_qc``); the GUI never reads it, so it is ``None`` when absent.
    """

    image_space: Path
    ccf: Path | None = None
    histology_track_id: str | None = None
    histology_shank: int | None = None
    ephys_shank: int | None = None
    shank: int | None = None


@dataclass(frozen=True)
class ChannelTablePaths:
    """Absolute paths to producer-owned channel geometry files."""

    local_coordinates: Path
    raw_ind: Path
    contact_id: Path | None
    shank_ind: Path


@dataclass(frozen=True)
class ProbeInfo:
    """Resolved metadata and paths for a single probe."""

    probe_id: str
    probe_name: str
    recording_id: str
    logical_probe: str
    ephys_collection: str
    num_shanks: int
    ephys_dir: Path | None
    channel_table: ChannelTablePaths | None
    xyz_picks: tuple[XyzPicks, ...]

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


@dataclass(frozen=True)
class MouseRoot:
    """Resolved view of a preprocessed mouse output directory.

    Probes are nested by ``recording_id`` then GUI-selectable ephys collection.
    ``ProbeInfo.probe_name`` is currently the selectable collection key for
    compatibility with older app code; ``ProbeInfo.logical_probe`` carries the
    histology/logical probe label and may repeat across split streams.
    """

    root: Path
    schema_version: str
    mouse_id: str
    transforms: TransformPaths
    histology: HistologyImagePaths
    probes: dict[str, dict[str, ProbeInfo]]

    @property
    def sessions(self) -> list[str]:
        """All recording IDs represented in this mouse root (sorted)."""
        return sorted(self.probes.keys())

    def probes_for_session(self, recording_id: str) -> list[str]:
        """Ephys collection labels for a given recording ID (sorted)."""
        return sorted(self.probes.get(recording_id, {}).keys())

    def get_probe(self, recording_id: str, probe_name: str) -> ProbeInfo:
        """Look up a probe by selected ``(recording_id, ephys_collection)``."""
        if recording_id not in self.probes:
            raise DataPackageError(
                f"No recording {recording_id!r} in datapackage "
                f"(available: {sorted(self.probes.keys())})"
            )
        probes_for_rec = self.probes[recording_id]
        if probe_name not in probes_for_rec:
            raise DataPackageError(
                f"No probe {probe_name!r} in recording {recording_id!r} "
                f"(available: {sorted(probes_for_rec.keys())})"
            )
        return probes_for_rec[probe_name]


def load_mouse_root(
    mouse_root: Path,
    *,
    asset_roots: Sequence[Path] | None = None,
    asset_overrides: Mapping[str, Path] | None = None,
) -> MouseRoot:
    """Load and validate a mouse root directory.

    Parameters
    ----------
    mouse_root : Path
        Directory containing ``datapackage.json``.
    asset_roots : sequence of Path, optional
        Roots under which external assets may be found by name or id. When not
        supplied, ``IBL_ASSET_ROOTS`` is used.
    asset_overrides : mapping, optional
        Explicit mapping from logical asset key or asset name to a local asset
        directory. Merged with ``IBL_ASSET_OVERRIDES``; explicit arguments win.

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

    try:
        validate_datapackage_contract(raw)
    except DatapackageContractError as e:
        raise DataPackageError(str(e)) from e

    external_assets = _parse_external_assets(raw.get("external_assets", {}))
    runtime_config = _load_asset_config_file()
    resolver = RootSearchResolver(
        mouse_root,
        asset_roots=_load_asset_roots(asset_roots, runtime_config),
        asset_overrides=_load_asset_overrides(asset_overrides, runtime_config),
    )

    try:
        transforms = _parse_transforms(raw["transforms"], resolver, external_assets)
        histology = _parse_histology(raw["histology"], resolver, external_assets)
        probes = _parse_probes(raw["probes"], resolver, external_assets)
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


def _load_asset_config_file() -> dict[str, object]:
    config_path = os.environ.get("IBL_ASSET_CONFIG", "")
    if not config_path:
        return {}
    path = Path(config_path)
    try:
        parsed = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise DataPackageError(f"Malformed IBL_ASSET_CONFIG JSON in {path}: {e}") from e
    if not isinstance(parsed, dict):
        raise DataPackageError("IBL_ASSET_CONFIG must contain a JSON object")
    return parsed


def _load_asset_roots(
    asset_roots: Sequence[Path] | None,
    runtime_config: Mapping[str, object],
) -> list[Path]:
    if asset_roots is not None:
        return [Path(p) for p in asset_roots]
    raw = os.environ.get("IBL_ASSET_ROOTS", "")
    if raw:
        return [Path(p) for p in raw.split(os.pathsep) if p]
    config_roots = runtime_config.get("asset_roots", [])
    if not isinstance(config_roots, list):
        raise DataPackageError("IBL_ASSET_CONFIG asset_roots must be a list")
    return [Path(p) for p in config_roots]


def _load_asset_overrides(
    asset_overrides: Mapping[str, Path] | None,
    runtime_config: Mapping[str, object],
) -> dict[str, Path]:
    loaded: dict[str, Path] = {}
    config_overrides = runtime_config.get("asset_overrides", {})
    if config_overrides:
        if not isinstance(config_overrides, dict):
            raise DataPackageError(
                "IBL_ASSET_CONFIG asset_overrides must be a JSON object"
            )
        loaded.update({str(k): Path(v) for k, v in config_overrides.items()})
    raw = os.environ.get("IBL_ASSET_OVERRIDES", "")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise DataPackageError(f"Malformed IBL_ASSET_OVERRIDES JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise DataPackageError("IBL_ASSET_OVERRIDES must be a JSON object")
        loaded.update({str(k): Path(v) for k, v in parsed.items()})
    if asset_overrides:
        loaded.update({str(k): Path(v) for k, v in asset_overrides.items()})
    return loaded


def _parse_external_assets(d: dict) -> dict[str, AssetRef]:
    assets: dict[str, AssetRef] = {}
    for key, entry in d.items():
        assets[key] = AssetRef(
            key=key,
            role=entry["role"],
            name=entry["name"],
            id=entry.get("id"),
            uri=entry.get("uri"),
            checksum=entry.get("checksum"),
        )
    return assets


def _parse_path_ref(value: object) -> PathRef:
    if isinstance(value, str):
        raise DataPackageError(
            f"Legacy string path {value!r} found in schema 3 datapackage. "
            "Regenerate datapackage.json with aind-ibl-preprocess --datapackage-only."
        )
    if not isinstance(value, dict):
        raise DataPackageError(
            f"Expected path reference object, got {type(value).__name__}"
        )
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise DataPackageError(f"Path reference missing non-empty 'path': {value!r}")
    asset = value.get("asset")
    if asset is not None and not isinstance(asset, str):
        raise DataPackageError(
            f"Path reference 'asset' must be string or null: {value!r}"
        )
    return PathRef(asset=asset, path=path)


def _resolve_ref(
    value: object, resolver: RootSearchResolver, assets: Mapping[str, AssetRef]
) -> Path:
    ref = _parse_path_ref(value)
    if ref.asset is None:
        return resolver.resolve(None, ref.path)
    try:
        asset = assets[ref.asset]
    except KeyError as e:
        raise DataPackageError(
            f"Unknown external asset key {ref.asset!r} for path {ref.path!r}"
        ) from e
    return resolver.resolve(asset, ref.path)


def _resolve_ephys_dir(
    value: object, resolver: RootSearchResolver, assets: Mapping[str, AssetRef]
) -> Path:
    """Resolve a probe's ephys collection directory without legacy healing."""
    return _resolve_ref(value, resolver, assets)


def _parse_transforms(
    d: dict[str, object], resolver: RootSearchResolver, assets: Mapping[str, AssetRef]
) -> TransformPaths:
    return TransformPaths(
        image_to_template_affine=_resolve_ref(
            d["image_to_template_affine"], resolver, assets
        ),
        image_to_template_warp=_resolve_ref(
            d["image_to_template_warp"], resolver, assets
        ),
        template_to_ccf_affine=_resolve_ref(
            d["template_to_ccf_affine"], resolver, assets
        ),
        template_to_ccf_warp=_resolve_ref(d["template_to_ccf_warp"], resolver, assets),
    )


def _parse_histology(
    d: dict, resolver: RootSearchResolver, assets: Mapping[str, AssetRef]
) -> HistologyImagePaths:
    img = d["image_space"]
    additional = {
        Path(_parse_path_ref(ref).path).stem: _resolve_ref(ref, resolver, assets)
        for ref in img.get("additional_channels", [])
    }
    return HistologyImagePaths(
        registration=_resolve_ref(img["registration"], resolver, assets),
        registration_pipeline=_resolve_ref(
            img["registration_pipeline"], resolver, assets
        )
        if img.get("registration_pipeline") is not None
        else None,
        registration_pipeline_geometry=_resolve_ref(
            img["registration_pipeline_geometry"], resolver, assets
        )
        if img.get("registration_pipeline_geometry") is not None
        else None,
        ccf_template=_resolve_ref(img["ccf_template"], resolver, assets),
        labels=_resolve_ref(img["labels"], resolver, assets),
        additional_channels=additional,
    )


def _parse_probes(
    d: dict,
    resolver: RootSearchResolver,
    assets: Mapping[str, AssetRef],
) -> dict[str, dict[str, ProbeInfo]]:
    """Parse the nested ``recording_id -> ephys_collection -> entry`` JSON."""
    probes: dict[str, dict[str, ProbeInfo]] = {}
    for recording_id, recording_probes in d.items():
        if not isinstance(recording_probes, dict):
            raise DataPackageError(
                f"Expected nested dict under recording {recording_id!r}, got "
                f"{type(recording_probes).__name__}. Datapackage may be from a "
                "pre-2.0.0 schema; re-run preprocessing."
            )
        for collection_key, entry in recording_probes.items():
            ephys_collection = str(entry.get("ephys_collection") or collection_key)
            logical_probe = str(entry.get("logical_probe") or ephys_collection)
            recording_entries = probes.setdefault(recording_id, {})
            if ephys_collection in recording_entries:
                raise DataPackageError(
                    "Duplicate ephys collection in datapackage probes for "
                    f"{recording_id!r}: {ephys_collection!r}"
                )
            picks = tuple(
                XyzPicks(
                    image_space=_resolve_ref(p["image_space"], resolver, assets),
                    # CCF picks are a QC-only output (emit_qc); the GUI never
                    # reads them (it recomputes CCF from image_space + the
                    # transforms), so ``ccf`` is null unless QC was emitted.
                    ccf=_resolve_ref(p["ccf"], resolver, assets)
                    if p.get("ccf") is not None
                    else None,
                    histology_track_id=p.get("histology_track_id"),
                    histology_shank=p.get("histology_shank"),
                    ephys_shank=p.get("ephys_shank"),
                    shank=p.get("shank"),
                )
                for p in entry["xyz_picks"]
            )
            ephys_rel = entry.get("ephys")
            channel_table = _parse_channel_table(
                entry.get("channel_table"), resolver, assets
            )
            recording_entries[ephys_collection] = ProbeInfo(
                probe_id=entry["probe_id"],
                probe_name=ephys_collection,
                recording_id=recording_id,
                logical_probe=logical_probe,
                ephys_collection=ephys_collection,
                num_shanks=entry["num_shanks"],
                ephys_dir=_resolve_ephys_dir(ephys_rel, resolver, assets)
                if ephys_rel
                else None,
                channel_table=channel_table,
                xyz_picks=picks,
            )
    return probes


def _parse_channel_table(
    d: dict | None,
    resolver: RootSearchResolver,
    assets: Mapping[str, AssetRef],
) -> ChannelTablePaths | None:
    if not d:
        return None
    contact_id = d.get("contact_id")
    return ChannelTablePaths(
        local_coordinates=_resolve_ref(d["local_coordinates"], resolver, assets),
        raw_ind=_resolve_ref(d["raw_ind"], resolver, assets),
        contact_id=_resolve_ref(contact_id, resolver, assets)
        if contact_id is not None
        else None,
        shank_ind=_resolve_ref(d["shank_ind"], resolver, assets),
    )
