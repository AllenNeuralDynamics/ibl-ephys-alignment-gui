"""Tests for probe-track loading service."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ephys_alignment_gui.datapackage_loader import ProbeInfo, XyzPicks
from ephys_alignment_gui.services.probe_track import ProbeTrackService


class FakeBrainAtlas:
    def rotate_to_canonical(self, points):
        return np.asarray(points) + 1.0


def _probe_with_pick(path) -> ProbeInfo:
    return ProbeInfo(
        probe_id="rec1:streamA",
        probe_name="probeA",
        recording_id="rec1",
        logical_probe="probeA",
        ephys_collection="streamA",
        num_shanks=1,
        ephys_dir=None,
        channel_table=None,
        xyz_picks=(XyzPicks(image_space=path, ccf=path),),
    )


def test_load_track_annotations_reads_microns_and_rotates_to_canonical(tmp_path):
    path = tmp_path / "xyz_picks.json"
    path.write_text(json.dumps({"xyz_picks": [[1000.0, 2000.0, 3000.0]]}))

    track = ProbeTrackService().load_track_annotations(
        probe=_probe_with_pick(path),
        shank_idx=0,
        brain_atlas=FakeBrainAtlas(),
    )

    np.testing.assert_allclose(track, [[1.001, 1.002, 1.003]])


def test_load_track_annotations_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ProbeTrackService().load_track_annotations(
            probe=_probe_with_pick(tmp_path / "missing.json"),
            shank_idx=0,
            brain_atlas=FakeBrainAtlas(),
        )
