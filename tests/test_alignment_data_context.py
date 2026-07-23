"""Tests for selected datapackage/probe metadata context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.datapackage_loader import MouseRoot, ProbeInfo
from ephys_alignment_gui.ephys_data_service import ChannelTable, EphysStreamData


def _probe(
    *,
    recording_id: str = "rec1",
    probe_name: str = "probeA",
    ephys_collection: str = "streamA",
    num_shanks: int = 2,
) -> ProbeInfo:
    return ProbeInfo(
        probe_id=f"{recording_id}:{ephys_collection}",
        probe_name=probe_name,
        recording_id=recording_id,
        logical_probe=probe_name,
        ephys_collection=ephys_collection,
        num_shanks=num_shanks,
        ephys_dir=Path("/tmp/ephys"),
        channel_table=None,
        xyz_picks=(),
    )


def _mouse_root(root: Path, probe: ProbeInfo) -> MouseRoot:
    return MouseRoot(
        root=root,
        schema_version="3.1.0",
        mouse_id="mouse1",
        transforms=None,
        histology=None,
        probes={probe.recording_id: {probe.probe_name: probe}},
    )


def _channel_table() -> ChannelTable:
    return ChannelTable(
        local_coordinates=np.array(
            [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
        ),
        shank_indices=np.array([0, 0, 1, 1]),
    )


def test_context_selects_probe_and_attaches_channel_table(tmp_path) -> None:
    probe = _probe(num_shanks=4)
    context = AlignmentDataContext(mouse_root=_mouse_root(tmp_path, probe))

    selected = context.select_probe("rec1", "probeA")
    context.attach_channel_table(_channel_table())

    assert selected is probe
    assert context.probe_id == "rec1:streamA"
    assert context.n_shanks == 2
    assert context.shank_labels() == ["1/2", "2/2"]


def test_select_probe_clears_previous_channel_table(tmp_path) -> None:
    probe = _probe()
    context = AlignmentDataContext(mouse_root=_mouse_root(tmp_path, probe))
    context.select_probe("rec1", "probeA")
    context.attach_channel_table(_channel_table())

    context.select_probe("rec1", "probeA")

    assert context.channel_table is None
    assert context.n_shanks == 0


def test_set_mouse_root_clears_selected_probe_and_channel_table(
    monkeypatch, tmp_path
) -> None:
    old_probe = _probe()
    new_probe = _probe(recording_id="rec2", ephys_collection="streamB")
    context = AlignmentDataContext(mouse_root=_mouse_root(tmp_path / "old", old_probe))
    context.select_probe("rec1", "probeA")
    context.attach_channel_table(_channel_table())

    monkeypatch.setattr(
        "ephys_alignment_gui.alignment_data_context.load_mouse_root",
        lambda root: _mouse_root(root, new_probe),
    )

    loaded = context.set_mouse_root(tmp_path / "new")

    assert loaded.root == tmp_path / "new"
    assert context.probe_info is None
    assert context.channel_table is None


def test_validate_cached_stream_matches_selected_probe(tmp_path) -> None:
    probe = _probe()
    table = _channel_table()
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection="streamA",
        ephys_dir=Path("/tmp/ephys"),
        channel_table=table,
        alf_data={},
        session_notes="notes",
    )
    context = AlignmentDataContext(mouse_root=_mouse_root(tmp_path, probe))
    context.select_probe("rec1", "probeA")

    context.validate_cached_stream(stream)


def test_validate_cached_stream_rejects_wrong_collection(tmp_path) -> None:
    probe = _probe()
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection="other",
        ephys_dir=Path("/tmp/ephys"),
        channel_table=_channel_table(),
        alf_data={},
        session_notes="notes",
    )
    context = AlignmentDataContext(mouse_root=_mouse_root(tmp_path, probe))
    context.select_probe("rec1", "probeA")

    with pytest.raises(ValueError, match="collection"):
        context.validate_cached_stream(stream)
