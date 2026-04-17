"""Tests for the mouse-root datapackage loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ephys_alignment_gui.datapackage_loader import (
    DataPackageError,
    load_mouse_root,
)


def _make_mouse_root(
    tmp_path: Path,
    *,
    schema_version: str = "1.1.0",
    extra_probes: dict | None = None,
) -> Path:
    """Create a minimal mouse-root directory with a datapackage.json.

    Also touches the target files on disk so `.resolve()` checks hold even if
    relative paths cross ``..`` boundaries.
    """
    mouse_root = tmp_path / "results" / "mouse42"
    mouse_root.mkdir(parents=True)

    # Histology files (inside mouse root).
    img_dir = mouse_root / "image_space_histology"
    img_dir.mkdir()
    for name in (
        "histology_registration.nrrd",
        "histology_registration_pipeline.nrrd",
        "ccf_in_mouse.nrrd",
        "labels_in_mouse.nrrd",
        "Ex_561_Em_600.nrrd",
    ):
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

    probes = {
        "probeA": {
            "probe_id": "p-1",
            "recording_id": "rec1",
            "num_shanks": 1,
            "ephys": "rec1/probeA/spikes",
            "xyz_picks": [
                {
                    "ccf": "rec1/probeA/xyz_picks.json",
                    "image_space": "rec1/probeA/xyz_picks_image_space.json",
                }
            ],
        }
    }
    if extra_probes:
        probes.update(extra_probes)

    spim_rel = "../../SmartSPIM_mouse42_123/image_atlas_alignment/Ex_561_Em_600"
    tmpl_rel = "../../spim_template_to_ccf"
    img_rel = "image_space_histology"
    i2t_aff = f"{spim_rel}/ls_to_template_SyN_0GenericAffine.mat"
    i2t_warp = f"{spim_rel}/ls_to_template_SyN_1InverseWarp.nii.gz"
    t2c_aff = f"{tmpl_rel}/syn_0GenericAffine.mat"
    t2c_warp = f"{tmpl_rel}/syn_1InverseWarp.nii.gz"
    dp = {
        "schema_version": schema_version,
        "mouse_id": "mouse42",
        "transforms": {
            "image_to_template_affine": i2t_aff,
            "image_to_template_warp": i2t_warp,
            "template_to_ccf_affine": t2c_aff,
            "template_to_ccf_warp": t2c_warp,
        },
        "histology": {
            "image_space": {
                "registration": f"{img_rel}/histology_registration.nrrd",
                "registration_pipeline": (
                    f"{img_rel}/histology_registration_pipeline.nrrd"
                ),
                "ccf_template": f"{img_rel}/ccf_in_mouse.nrrd",
                "labels": f"{img_rel}/labels_in_mouse.nrrd",
                "additional_channels": [f"{img_rel}/Ex_561_Em_600.nrrd"],
            },
            "ccf_space": {
                "registration": "ccf_space_histology/histology_registration.nrrd",
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
    mr = load_mouse_root(root)
    assert mr.mouse_id == "mouse42"
    assert mr.schema_version == "1.1.0"
    assert mr.sessions == ["rec1"]
    assert mr.probes_for_session("rec1") == ["probeA"]


def test_transforms_resolve_via_parent_traversal(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = load_mouse_root(root)
    # The transform lives in a sibling asset outside mouse_root — must resolve
    # to an absolute path that exists.
    assert mr.transforms.image_to_template_affine.is_absolute()
    assert mr.transforms.image_to_template_affine.is_file()
    assert mr.transforms.template_to_ccf_warp.is_file()


def test_histology_paths_are_absolute(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = load_mouse_root(root)
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


def test_probe_info_resolves_paths(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = load_mouse_root(root)
    probe = mr.get_probe("rec1", "probeA")
    assert probe.probe_id == "p-1"
    assert probe.num_shanks == 1
    assert probe.ephys_dir is not None and probe.ephys_dir.is_dir()
    assert len(probe.xyz_picks) == 1
    pk = probe.picks_for_shank(0)
    assert pk.image_space.is_file()
    assert pk.ccf.is_file()
    assert pk.shank is None


def test_rejects_older_schema(tmp_path):
    root = _make_mouse_root(tmp_path, schema_version="1.0.0")
    with pytest.raises(DataPackageError, match="Unsupported datapackage schema"):
        load_mouse_root(root)


def test_rejects_incompatible_major_schema(tmp_path):
    root = _make_mouse_root(tmp_path, schema_version="2.0.0")
    with pytest.raises(DataPackageError, match="Unsupported datapackage schema"):
        load_mouse_root(root)


def test_rejects_missing_schema_version(tmp_path):
    root = _make_mouse_root(tmp_path)
    dp_path = root / "datapackage.json"
    data = json.loads(dp_path.read_text())
    data.pop("schema_version")
    dp_path.write_text(json.dumps(data))
    with pytest.raises(DataPackageError, match="no schema_version"):
        load_mouse_root(root)


def test_malformed_json_raises_datapackage_error(tmp_path):
    root = tmp_path / "results" / "mouse42"
    root.mkdir(parents=True)
    (root / "datapackage.json").write_text("{not valid json")
    with pytest.raises(DataPackageError, match="Malformed"):
        load_mouse_root(root)


def test_get_probe_wrong_session_raises(tmp_path):
    root = _make_mouse_root(tmp_path)
    mr = load_mouse_root(root)
    with pytest.raises(DataPackageError, match="belongs to recording"):
        mr.get_probe("recWRONG", "probeA")


def test_multi_shank_probe_picks_by_index(tmp_path):
    extra = {
        "probeB": {
            "probe_id": "p-2",
            "recording_id": "rec1",
            "num_shanks": 2,
            "ephys": "rec1/probeB/spikes",
            "xyz_picks": [
                {
                    "ccf": "rec1/probeB/xyz_picks_shank1.json",
                    "image_space": "rec1/probeB/xyz_picks_shank1_image_space.json",
                    "shank": 1,
                },
                {
                    "ccf": "rec1/probeB/xyz_picks_shank2.json",
                    "image_space": "rec1/probeB/xyz_picks_shank2_image_space.json",
                    "shank": 2,
                },
            ],
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

    mr = load_mouse_root(root)
    probe = mr.get_probe("rec1", "probeB")
    assert probe.num_shanks == 2
    assert probe.picks_for_shank(0).shank == 1
    assert probe.picks_for_shank(1).shank == 2
    with pytest.raises(DataPackageError, match="no shank 3"):
        probe.picks_for_shank(2)


def test_sessions_are_distinct_recordings(tmp_path):
    extra = {
        "probeB": {
            "probe_id": "p-2",
            "recording_id": "rec2",
            "num_shanks": 1,
            "ephys": "rec2/probeB/spikes",
            "xyz_picks": [
                {
                    "ccf": "rec2/probeB/xyz_picks.json",
                    "image_space": "rec2/probeB/xyz_picks_image_space.json",
                }
            ],
        }
    }
    root = _make_mouse_root(tmp_path, extra_probes=extra)
    (root / "rec2" / "probeB").mkdir(parents=True)
    for name in ("xyz_picks.json", "xyz_picks_image_space.json"):
        (root / "rec2" / "probeB" / name).touch()
    (root / "rec2" / "probeB" / "spikes").mkdir()

    mr = load_mouse_root(root)
    assert mr.sessions == ["rec1", "rec2"]
    assert mr.probes_for_session("rec1") == ["probeA"]
    assert mr.probes_for_session("rec2") == ["probeB"]
