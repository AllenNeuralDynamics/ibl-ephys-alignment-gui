"""Tests for the mouse-root datapackage loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ephys_alignment_gui.io.datapackage_loader import (
    AssetNotFound,
    DataPackageError,
    load_mouse_root,
)


def _ref(path: str, asset: str | None = None) -> dict[str, str | None]:
    return {"asset": asset, "path": path}


def _load(root: Path):
    return load_mouse_root(root, asset_roots=[root.parent.parent])


def _make_mouse_root(
    tmp_path: Path,
    *,
    schema_version: str = "3.1.0",
    extra_probes: dict[str, dict[str, dict]] | None = None,
    ephys: str | None = "rec1/probeA",
    channels_rel: str | None = None,
    ccf_null: bool = False,
    pipeline_volume: bool = True,
    pipeline_geometry: bool = False,
) -> Path:
    """Create a minimal mouse-root directory with a datapackage.json.

    Also touches the target files on disk so resolver checks can find them.

    ``extra_probes`` is shaped ``{recording_id: {probe_name: entry}}`` to
    match the nested probes schema.
    """
    mouse_root = tmp_path / "results" / "mouse42"
    mouse_root.mkdir(parents=True)

    # Histology files (inside mouse root).
    img_dir = mouse_root / "image_space_histology"
    img_dir.mkdir()
    histology_files = [
        "histology_registration.nrrd",
        "ccf_in_mouse.nrrd",
        "labels_in_mouse.nrrd",
        "Ex_561_Em_600.nrrd",
    ]
    if pipeline_volume:
        histology_files.append("histology_registration_pipeline.nrrd")
    if pipeline_geometry:
        histology_files.append("histology_registration_pipeline.json")
    for name in histology_files:
        (img_dir / name).touch()

    # Transforms live in a sibling SmartSPIM asset (outside mouse root).
    smartspim = (
        tmp_path / "SmartSPIM_mouse42_123" / "image_atlas_alignment" / "Ex_561_Em_600"
    )
    smartspim.mkdir(parents=True)
    for name in (
        "ls_to_template_SyN_0GenericAffine.mat",
        "ls_to_template_SyN_1InverseWarp.nii.gz",
    ):
        (smartspim / name).touch()
    template_ccf = tmp_path / "spim_template_to_ccf"
    template_ccf.mkdir()
    for name in ("syn_0GenericAffine.mat", "syn_1InverseWarp.nii.gz"):
        (template_ccf / name).touch()

    # Probe tree.
    probe_dir = mouse_root / "rec1" / "probeA"
    probe_dir.mkdir(parents=True)
    (probe_dir / "xyz_picks.json").touch()
    (probe_dir / "xyz_picks_image_space.json").touch()
    (probe_dir / "spikes").mkdir()
    if channels_rel is not None:
        chan_dir = mouse_root / channels_rel
        chan_dir.mkdir(parents=True, exist_ok=True)
        (chan_dir / "channels.localCoordinates.npy").touch()

    probes: dict[str, dict[str, dict]] = {
        "rec1": {
            "probeA": {
                "probe_id": "p-1",
                "num_shanks": 1,
                "ephys": _ref(ephys) if ephys else None,
                "xyz_picks": [
                    {
                        "ccf": None if ccf_null else _ref("rec1/probeA/xyz_picks.json"),
                        "image_space": _ref("rec1/probeA/xyz_picks_image_space.json"),
                    }
                ],
            }
        }
    }
    if extra_probes:
        for rec_id, rec_probes in extra_probes.items():
            probes.setdefault(rec_id, {}).update(rec_probes)

    img_rel = "image_space_histology"
    image_space = {
        "registration": _ref(f"{img_rel}/histology_registration.nrrd"),
        "ccf_template": _ref(f"{img_rel}/ccf_in_mouse.nrrd"),
        "labels": _ref(f"{img_rel}/labels_in_mouse.nrrd"),
        "additional_channels": [_ref(f"{img_rel}/Ex_561_Em_600.nrrd")],
    }
    if pipeline_volume:
        image_space["registration_pipeline"] = _ref(
            f"{img_rel}/histology_registration_pipeline.nrrd"
        )
    if pipeline_geometry:
        image_space["registration_pipeline_geometry"] = _ref(
            f"{img_rel}/histology_registration_pipeline.json"
        )

    dp = {
        "schema_version": schema_version,
        "mouse_id": "mouse42",
        "platform": "local",
        "external_assets": {
            "smartspim": {
                "role": "smartspim_registration",
                "name": "SmartSPIM_mouse42_123",
                "id": None,
                "uri": None,
                "checksum": None,
            },
            "spim_template_to_ccf": {
                "role": "template_to_ccf",
                "name": "spim_template_to_ccf",
                "id": None,
                "uri": None,
                "checksum": None,
            },
        },
        "transforms": {
            "image_to_template_affine": _ref(
                "image_atlas_alignment/Ex_561_Em_600/ls_to_template_SyN_0GenericAffine.mat",
                "smartspim",
            ),
            "image_to_template_warp": _ref(
                "image_atlas_alignment/Ex_561_Em_600/ls_to_template_SyN_1InverseWarp.nii.gz",
                "smartspim",
            ),
            "template_to_ccf_affine": _ref(
                "syn_0GenericAffine.mat", "spim_template_to_ccf"
            ),
            "template_to_ccf_warp": _ref(
                "syn_1InverseWarp.nii.gz", "spim_template_to_ccf"
            ),
        },
        "histology": {
            "image_space": image_space,
            "ccf_space": {
                "registration": _ref("ccf_space_histology/histology_registration.nrrd"),
            },
        },
        "probes": probes,
    }
    (mouse_root / "datapackage.json").write_text(json.dumps(dp))
    return mouse_root


def test_missing_datapackage_raises(tmp_path):
    with pytest.raises(DataPackageError, match="No datapackage.json"):
        load_mouse_root(tmp_path)


def test_loads_basic_mouse_root(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = _load(root)
    assert mr.mouse_id == "mouse42"
    assert mr.schema_version == "3.1.0"
    assert mr.sessions == ["rec1"]
    assert mr.probes_for_session("rec1") == ["probeA"]


def test_transforms_resolve_via_asset_root_search(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = _load(root)
    # The transform lives in an external asset outside mouse_root and resolves
    # through the configured asset root.
    assert mr.transforms.image_to_template_affine.is_absolute()
    assert mr.transforms.image_to_template_affine.is_file()
    assert mr.transforms.template_to_ccf_warp.is_file()


def test_external_asset_missing_without_roots(tmp_path):
    root = _make_mouse_root(tmp_path)
    with pytest.raises(AssetNotFound, match="IBL_ASSET_ROOTS"):
        load_mouse_root(root)


def test_external_asset_override_by_name(tmp_path):
    root = _make_mouse_root(tmp_path)
    moved = tmp_path / "renamed_histology_asset"
    source = tmp_path / "SmartSPIM_mouse42_123"
    source.rename(moved)

    mr = load_mouse_root(
        root,
        asset_roots=[tmp_path],
        asset_overrides={"SmartSPIM_mouse42_123": moved},
    )

    assert mr.transforms.image_to_template_affine.is_file()


def test_external_asset_config_file(tmp_path, monkeypatch):
    root = _make_mouse_root(tmp_path)
    moved = tmp_path / "renamed_histology_asset"
    (tmp_path / "SmartSPIM_mouse42_123").rename(moved)
    config = {
        "asset_roots": [str(tmp_path)],
        "asset_overrides": {
            "SmartSPIM_mouse42_123": str(moved),
        },
    }
    config_path = tmp_path / "asset_config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv("IBL_ASSET_CONFIG", str(config_path))

    mr = load_mouse_root(root)

    assert mr.transforms.image_to_template_affine.is_file()


def test_histology_paths_are_absolute(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = _load(root)
    for p in (
        mr.histology.registration,
        mr.histology.registration_pipeline,
        mr.histology.ccf_template,
        mr.histology.labels,
    ):
        assert p.is_absolute()
        assert p.is_file()
    assert "Ex_561_Em_600" in mr.histology.additional_channels
    assert mr.histology.additional_channels["Ex_561_Em_600"].is_file()


def test_loads_3_2_pipeline_geometry_sidecar(tmp_path):
    root = _make_mouse_root(
        tmp_path,
        schema_version="3.2.0",
        pipeline_geometry=True,
    )
    sidecar = root / "image_space_histology" / "histology_registration_pipeline.json"

    mr = _load(root)

    assert mr.schema_version == "3.2.0"
    assert mr.histology.registration_pipeline_geometry == sidecar
    assert mr.histology.registration_pipeline is not None
    assert mr.histology.registration_pipeline.is_file()


def test_loads_4_0_pipeline_geometry_without_volume(tmp_path):
    root = _make_mouse_root(
        tmp_path,
        schema_version="4.0.0",
        pipeline_volume=False,
        pipeline_geometry=True,
    )
    sidecar = root / "image_space_histology" / "histology_registration_pipeline.json"

    mr = _load(root)

    assert mr.schema_version == "4.0.0"
    assert mr.histology.registration_pipeline is None
    assert mr.histology.registration_pipeline_geometry == sidecar


def test_probe_info_resolves_paths(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = _load(root)
    probe = mr.get_probe("rec1", "probeA")
    assert probe.probe_id == "p-1"
    assert probe.logical_probe == "probeA"
    assert probe.ephys_collection == "probeA"
    assert probe.num_shanks == 1
    assert probe.ephys_dir is not None and probe.ephys_dir.is_dir()
    assert len(probe.xyz_picks) == 1
    pk = probe.picks_for_shank(0)
    assert pk.image_space.is_file()
    assert pk.ccf.is_file()
    assert pk.shank is None


def test_null_ccf_pick_is_tolerated(tmp_path):
    # CCF picks are a QC-only output (emit_qc); the producer writes
    # ``ccf: null`` when QC is off. The GUI never reads them, so loading
    # must not choke on the null.
    root = _make_mouse_root(tmp_path, ccf_null=True)
    mr = _load(root)
    probe = mr.get_probe("rec1", "probeA")
    pk = probe.picks_for_shank(0)
    assert pk.ccf is None
    assert pk.image_space.is_file()


def test_loads_explicit_ephys_geometry_fields(tmp_path):
    root = _make_mouse_root(
        tmp_path,
        ephys="rec1/ProbeD",
        channels_rel="rec1/ProbeD",
    )
    probe_dir = root / "rec1" / "ProbeD"
    (probe_dir / "channels.rawInd.npy").touch()
    (probe_dir / "channels.contactId.npy").touch()
    (probe_dir / "channels.shankInd.npy").touch()
    (probe_dir / "xyz_picks_shank4.json").touch()
    (probe_dir / "xyz_picks_shank4_image_space.json").touch()

    dp_path = root / "datapackage.json"
    data = json.loads(dp_path.read_text())
    data["probes"] = {
        "rec1": {
            "ProbeD": {
                "probe_id": "track-probe0-shank3",
                "logical_probe": "probe0",
                "ephys_collection": "ProbeD",
                "num_shanks": 1,
                "ephys": _ref("rec1/ProbeD"),
                "channel_table": {
                    "local_coordinates": _ref(
                        "rec1/ProbeD/channels.localCoordinates.npy"
                    ),
                    "raw_ind": _ref("rec1/ProbeD/channels.rawInd.npy"),
                    "contact_id": _ref("rec1/ProbeD/channels.contactId.npy"),
                    "shank_ind": _ref("rec1/ProbeD/channels.shankInd.npy"),
                },
                "xyz_picks": [
                    {
                        "ccf": _ref("rec1/ProbeD/xyz_picks_shank4.json"),
                        "image_space": _ref(
                            "rec1/ProbeD/xyz_picks_shank4_image_space.json"
                        ),
                        "histology_track_id": "track-probe0-shank3",
                        "histology_shank": 3,
                        "ephys_shank": 0,
                        "shank": 1,
                    }
                ],
            }
        }
    }
    dp_path.write_text(json.dumps(data))

    mr = _load(root)
    probe = mr.get_probe("rec1", "ProbeD")
    assert probe.logical_probe == "probe0"
    assert probe.ephys_collection == "ProbeD"
    assert probe.channel_table is not None
    assert probe.channel_table.contact_id.name == "channels.contactId.npy"
    assert probe.channel_table.shank_ind.name == "channels.shankInd.npy"
    picks = probe.picks_for_shank(0)
    assert picks.histology_shank == 3
    assert picks.ephys_shank == 0
    assert picks.shank == 1


def test_loads_channel_table_without_optional_contact_id(tmp_path):
    root = _make_mouse_root(
        tmp_path,
        ephys="rec1/ProbeD",
        channels_rel="rec1/ProbeD",
    )
    probe_dir = root / "rec1" / "ProbeD"
    (probe_dir / "channels.rawInd.npy").touch()
    (probe_dir / "channels.shankInd.npy").touch()
    (probe_dir / "xyz_picks.json").touch()
    (probe_dir / "xyz_picks_image_space.json").touch()

    dp_path = root / "datapackage.json"
    data = json.loads(dp_path.read_text())
    data["probes"] = {
        "rec1": {
            "ProbeD": {
                "probe_id": "track-probe0",
                "num_shanks": 1,
                "ephys": _ref("rec1/ProbeD"),
                "channel_table": {
                    "local_coordinates": _ref(
                        "rec1/ProbeD/channels.localCoordinates.npy"
                    ),
                    "raw_ind": _ref("rec1/ProbeD/channels.rawInd.npy"),
                    "shank_ind": _ref("rec1/ProbeD/channels.shankInd.npy"),
                },
                "xyz_picks": [
                    {
                        "ccf": _ref("rec1/ProbeD/xyz_picks.json"),
                        "image_space": _ref("rec1/ProbeD/xyz_picks_image_space.json"),
                    }
                ],
            }
        }
    }
    dp_path.write_text(json.dumps(data))

    mr = _load(root)
    probe = mr.get_probe("rec1", "ProbeD")
    assert probe.channel_table is not None
    assert probe.channel_table.contact_id is None


def test_ephys_dir_does_not_heal_bad_spikes_subdir_in_v3(tmp_path, caplog):
    """Schema 3 trusts the explicit ephys reference and does no legacy healing."""
    root = _make_mouse_root(
        tmp_path, ephys="rec1/probeA/spikes", channels_rel="rec1/probeA"
    )
    with caplog.at_level("WARNING"):
        mr = _load(root)
    probe = mr.get_probe("rec1", "probeA")
    assert probe.ephys_dir == (root / "rec1" / "probeA" / "spikes")
    assert not any("parent" in r.getMessage() for r in caplog.records)


def test_ephys_dir_unchanged_when_channels_present(tmp_path, caplog):
    """A correct datapackage (channels in the declared ephys dir) is untouched
    and logs no warning."""
    root = _make_mouse_root(tmp_path, ephys="rec1/probeA", channels_rel="rec1/probeA")
    with caplog.at_level("WARNING"):
        mr = _load(root)
    probe = mr.get_probe("rec1", "probeA")
    assert probe.ephys_dir == (root / "rec1" / "probeA")
    assert not any("parent" in r.getMessage() for r in caplog.records)


def test_ephys_dir_left_alone_when_unfixable(tmp_path):
    """The declared ephys directory is returned so downstream validation owns it."""
    root = _make_mouse_root(tmp_path, ephys="rec1/probeA/spikes")
    mr = _load(root)
    probe = mr.get_probe("rec1", "probeA")
    assert probe.ephys_dir == (root / "rec1" / "probeA" / "spikes")


def test_rejects_older_schema(tmp_path):
    """Pre-3.0.0 datapackages used string paths and must be regenerated."""
    root = _make_mouse_root(tmp_path, schema_version="2.1.0")
    with pytest.raises(DataPackageError, match="Unsupported datapackage schema"):
        _load(root)


def test_rejects_unvendored_schema_version(tmp_path):
    root = _make_mouse_root(tmp_path, schema_version="3.0.0")
    with pytest.raises(
        DataPackageError,
        match="GUI supports bundled schemas: 3.1.0, 3.2.0, 4.0.0",
    ):
        _load(root)


def test_rejects_incompatible_major_schema(tmp_path):
    root = _make_mouse_root(tmp_path, schema_version="5.0.0")
    with pytest.raises(DataPackageError, match="Unsupported datapackage schema"):
        _load(root)


def test_rejects_missing_schema_version(tmp_path):
    root = _make_mouse_root(tmp_path)
    dp_path = root / "datapackage.json"
    data = json.loads(dp_path.read_text())
    data.pop("schema_version")
    dp_path.write_text(json.dumps(data))
    with pytest.raises(DataPackageError, match="no schema_version"):
        _load(root)


def test_malformed_json_raises_datapackage_error(tmp_path):
    root = tmp_path / "results" / "mouse42"
    root.mkdir(parents=True)
    (root / "datapackage.json").write_text("{not valid json")
    with pytest.raises(DataPackageError, match="Malformed"):
        _load(root)


def test_schema_rejects_legacy_string_path(tmp_path):
    root = _make_mouse_root(tmp_path)
    dp_path = root / "datapackage.json"
    data = json.loads(dp_path.read_text())
    data["transforms"]["image_to_template_affine"] = (
        "image_atlas_alignment/Ex_561_Em_600/ls_to_template_SyN_0GenericAffine.mat"
    )
    dp_path.write_text(json.dumps(data))

    with pytest.raises(
        DataPackageError,
        match="does not match vendored schema 3.1.0",
    ):
        _load(root)


def test_schema_rejects_path_reference_without_asset_key(tmp_path):
    root = _make_mouse_root(tmp_path)
    dp_path = root / "datapackage.json"
    data = json.loads(dp_path.read_text())
    data["histology"]["image_space"]["registration"] = {
        "path": "image_space_histology/histology_registration.nrrd"
    }
    dp_path.write_text(json.dumps(data))

    with pytest.raises(DataPackageError, match="'asset' is a required property"):
        _load(root)


def test_schema_rejects_missing_required_image_space_pick(tmp_path):
    root = _make_mouse_root(tmp_path)
    dp_path = root / "datapackage.json"
    data = json.loads(dp_path.read_text())
    data["probes"]["rec1"]["probeA"]["xyz_picks"][0].pop("image_space")
    dp_path.write_text(json.dumps(data))

    with pytest.raises(DataPackageError, match="'image_space' is a required property"):
        _load(root)


def test_get_probe_unknown_recording_raises(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = _load(root)
    with pytest.raises(DataPackageError, match="No recording 'recWRONG'"):
        mr.get_probe("recWRONG", "probeA")


def test_get_probe_unknown_probe_in_known_recording_raises(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = _load(root)
    with pytest.raises(
        DataPackageError, match="No probe 'probeNOPE' in recording 'rec1'"
    ):
        mr.get_probe("rec1", "probeNOPE")


def test_multi_shank_probe_picks_by_index(tmp_path):
    extra = {
        "rec1": {
            "probeB": {
                "probe_id": "p-2",
                "num_shanks": 2,
                "ephys": _ref("rec1/probeB/spikes"),
                "xyz_picks": [
                    {
                        "ccf": _ref("rec1/probeB/xyz_picks_shank1.json"),
                        "image_space": _ref(
                            "rec1/probeB/xyz_picks_shank1_image_space.json"
                        ),
                        "shank": 1,
                    },
                    {
                        "ccf": _ref("rec1/probeB/xyz_picks_shank2.json"),
                        "image_space": _ref(
                            "rec1/probeB/xyz_picks_shank2_image_space.json"
                        ),
                        "shank": 2,
                    },
                ],
            }
        }
    }
    root = _make_mouse_root(tmp_path, extra_probes=extra)
    (root / "rec1" / "probeB").mkdir(parents=True)
    for name in (
        "xyz_picks_shank1.json",
        "xyz_picks_shank1_image_space.json",
        "xyz_picks_shank2.json",
        "xyz_picks_shank2_image_space.json",
    ):
        (root / "rec1" / "probeB" / name).touch()
    (root / "rec1" / "probeB" / "spikes").mkdir()

    mr = _load(root)
    probe = mr.get_probe("rec1", "probeB")
    assert probe.num_shanks == 2
    assert probe.picks_for_shank(0).shank == 1
    assert probe.picks_for_shank(1).shank == 2
    with pytest.raises(DataPackageError, match="no shank 3"):
        probe.picks_for_shank(2)


def test_sessions_are_distinct_recordings(tmp_path):
    extra = {
        "rec2": {
            "probeB": {
                "probe_id": "p-2",
                "num_shanks": 1,
                "ephys": _ref("rec2/probeB/spikes"),
                "xyz_picks": [
                    {
                        "ccf": _ref("rec2/probeB/xyz_picks.json"),
                        "image_space": _ref("rec2/probeB/xyz_picks_image_space.json"),
                    }
                ],
            }
        }
    }
    root = _make_mouse_root(tmp_path, extra_probes=extra)
    (root / "rec2" / "probeB").mkdir(parents=True)
    for name in ("xyz_picks.json", "xyz_picks_image_space.json"):
        (root / "rec2" / "probeB" / name).touch()
    (root / "rec2" / "probeB" / "spikes").mkdir()

    mr = _load(root)
    assert mr.sessions == ["rec1", "rec2"]
    assert mr.probes_for_session("rec1") == ["probeA"]
    assert mr.probes_for_session("rec2") == ["probeB"]


def test_same_probe_name_in_two_recordings_kept_distinct(tmp_path):
    """Same probe_name re-used across recordings stays distinct (the 2.0.0 fix)."""
    extra = {
        "rec2": {
            # Same name as rec1's probeA.
            "probeA": {
                "probe_id": "p-1-rec2",
                "num_shanks": 1,
                "ephys": _ref("rec2/probeA/spikes"),
                "xyz_picks": [
                    {
                        "ccf": _ref("rec2/probeA/xyz_picks.json"),
                        "image_space": _ref("rec2/probeA/xyz_picks_image_space.json"),
                    }
                ],
            }
        }
    }
    root = _make_mouse_root(tmp_path, extra_probes=extra)
    (root / "rec2" / "probeA").mkdir(parents=True)
    for name in ("xyz_picks.json", "xyz_picks_image_space.json"):
        (root / "rec2" / "probeA" / name).touch()
    (root / "rec2" / "probeA" / "spikes").mkdir()

    mr = _load(root)
    assert mr.sessions == ["rec1", "rec2"]
    assert mr.probes_for_session("rec1") == ["probeA"]
    assert mr.probes_for_session("rec2") == ["probeA"]
    p1 = mr.get_probe("rec1", "probeA")
    p2 = mr.get_probe("rec2", "probeA")
    assert p1.probe_id == "p-1"
    assert p2.probe_id == "p-1-rec2"
    assert p1.recording_id == "rec1"
    assert p2.recording_id == "rec2"
