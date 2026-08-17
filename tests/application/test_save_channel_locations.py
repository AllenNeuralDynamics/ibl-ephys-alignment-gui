"""Tests for lightweight save channel-location construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from ephys_alignment_gui.application.save_channel_locations import (
    AlignmentSaveChannelLocationBuilder,
    SaveChannelLocationError,
)
from ephys_alignment_gui.application.save_geometry_catalog import SaveGeometry
from ephys_alignment_gui.core.alignment_output import (
    AlignmentOutputMetadata,
    ChannelOutputIdentity,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.io.input_dataset_snapshot import InputProbeSnapshot


def test_save_channel_location_builder_uses_track_picks_and_geometry_depths() -> None:
    _FakeAlignment.instances = []
    probe_track_service = _FakeProbeTrackService()
    alignment_cls = _FakeAlignment
    builder = AlignmentSaveChannelLocationBuilder(
        probe_track_service=probe_track_service,
        alignment_cls=alignment_cls,
    )
    geometry = _geometry()
    feature = np.array([1.0, 2.0])
    track = np.array([3.0, 4.0])

    locations = builder.compute(
        geometry=geometry,
        feature=feature,
        track=track,
        brain_atlas="atlas",
    )

    np.testing.assert_array_equal(
        locations,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    assert probe_track_service.calls == [
        {
            "probe": geometry.probe,
            "shank_idx": 0,
            "brain_atlas": "atlas",
        }
    ]
    assert alignment_cls.instances[0].track_annotations_ras is _TRACK_ANNOTATIONS
    np.testing.assert_array_equal(
        alignment_cls.instances[0].chn_depths,
        geometry.channel_depths_um,
    )
    assert alignment_cls.instances[0].brain_atlas == "atlas"
    np.testing.assert_array_equal(alignment_cls.instances[0].feature, feature)
    np.testing.assert_array_equal(alignment_cls.instances[0].track, track)


def test_save_channel_location_builder_rejects_wrong_location_shape() -> None:
    builder = AlignmentSaveChannelLocationBuilder(
        probe_track_service=_FakeProbeTrackService(),
        alignment_cls=_WrongShapeAlignment,
    )

    with pytest.raises(SaveChannelLocationError, match="expected"):
        builder.compute(
            geometry=_geometry(),
            feature=np.array([1.0]),
            track=np.array([2.0]),
            brain_atlas="atlas",
        )


_TRACK_ANNOTATIONS = np.array([[0.0, 0.0, 0.0]])


@dataclass
class _FakeProbeTrackService:
    calls: list[dict[str, Any]]

    def __init__(self) -> None:
        self.calls = []

    def load_track_annotations(self, **kwargs):
        self.calls.append(kwargs)
        return _TRACK_ANNOTATIONS


class _FakeAlignment:
    instances: list[Any] = []

    def __init__(self, **kwargs) -> None:
        self.track_annotations_ras = kwargs["track_annotations_ras"]
        self.chn_depths = kwargs["chn_depths"]
        self.brain_atlas = kwargs["brain_atlas"]
        self.feature = None
        self.track = None
        self.instances.append(self)

    def get_channel_locations(self, feature, track):
        self.feature = feature
        self.track = track
        return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class _WrongShapeAlignment(_FakeAlignment):
    def get_channel_locations(self, feature, track):
        self.feature = feature
        self.track = track
        return np.array([[1.0, 2.0, 3.0]])


def _geometry() -> SaveGeometry:
    key = AlignmentKey("rec", "stream", 0)
    return SaveGeometry(
        key=key,
        probe=InputProbeSnapshot(
            probe_id="probe-id",
            probe_name="stream",
            recording_id="rec",
            logical_probe="stream",
            ephys_collection="stream",
            num_shanks=1,
            ephys_dir=None,
            channel_table=None,
            xyz_picks=(),
        ),
        channel_coordinates=np.array([[0.0, 0.0], [0.0, 20.0]]),
        channel_depths_um=np.array([0.0, 20.0]),
        channel_identity=ChannelOutputIdentity(raw_ind=np.array([0, 1])),
        output_metadata=AlignmentOutputMetadata(
            recording_id="rec",
            ephys_collection="stream",
            logical_probe="stream",
            shank_idx=0,
            n_shanks=1,
            probe_id="probe-id",
        ),
        multi_shank=False,
    )
