"""Tests for Qt-free alignment controller commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_repository import (
    LoadedAlignmentHistory,
    SavedAlignmentOutputs,
)
from ephys_alignment_gui.controller import (
    AlignmentChoicesUpdated,
    AlignmentController,
    AlignmentEditApplied,
    AlignmentOutputBuilt,
    AlignmentOutputsSaved,
    MouseRootLoaded,
    NoPreviousAlignments,
    OutputRootSet,
    PendingReferenceLinesUpdated,
    PreviousAlignmentSelected,
    PreviousAlignmentsLoaded,
    ProbeSelected,
    RecordingSelected,
    ShankRuntimeInitialized,
    ShankSelected,
)
from ephys_alignment_gui.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.workflow import Blocked, Failed, Ok


@dataclass(frozen=True)
class FakeMouseRoot:
    root: Path
    mouse_id: str
    sessions: list[str]
    probes: dict[str, dict[str, FakeProbeInfo]]


@dataclass(frozen=True)
class FakeProbeInfo:
    recording_id: str
    probe_name: str
    probe_id: str
    num_shanks: int
    ephys_collection: str = "streamA"


@dataclass(frozen=True)
class FakeChannelTable:
    n_shanks: int


@dataclass(frozen=True)
class FakeCachedStream:
    recording_id: str
    ephys_collection: str
    channel_table: FakeChannelTable


class FakeDataContext:
    def __init__(self, mouse_root: FakeMouseRoot | None = None) -> None:
        self.mouse_root = mouse_root
        self.probe_info: FakeProbeInfo | None = None
        self.channel_table: FakeChannelTable | None = None

    def set_mouse_root(self, mouse_root: Path) -> FakeMouseRoot:
        probes = {
            "rec1": {
                "probeA": FakeProbeInfo("rec1", "probeA", "rec1:probeA", 2),
                "probeB": FakeProbeInfo("rec1", "probeB", "rec1:probeB", 1),
            }
        }
        self.mouse_root = FakeMouseRoot(
            root=mouse_root,
            mouse_id="mouse1",
            sessions=["rec1"],
            probes=probes,
        )
        self.probe_info = None
        self.channel_table = None
        return self.mouse_root

    def list_probes(self, recording_id: str) -> list[str]:
        assert self.mouse_root is not None
        return sorted(self.mouse_root.probes[recording_id].keys())

    def select_probe(self, recording_id: str, probe_name: str) -> FakeProbeInfo:
        assert self.mouse_root is not None
        probe = self.mouse_root.probes[recording_id][probe_name]
        self.probe_info = probe
        self.channel_table = None
        return probe

    def attach_channel_table(self, channel_table: FakeChannelTable) -> None:
        if self.probe_info is None:
            raise RuntimeError("no probe selected")
        self.channel_table = channel_table

    def validate_cached_stream(self, ephys_stream: FakeCachedStream) -> None:
        if self.probe_info is None:
            raise RuntimeError("no probe selected")
        if ephys_stream.recording_id != self.probe_info.recording_id:
            raise ValueError("recording mismatch")
        if ephys_stream.ephys_collection != self.probe_info.ephys_collection:
            raise ValueError("collection mismatch")

    @property
    def n_shanks(self) -> int:
        if self.channel_table is None:
            return 0
        return self.channel_table.n_shanks

    def shank_labels(self) -> list[str]:
        if self.n_shanks == 1:
            return ["1/1"]
        return [f"{idx + 1}/{self.n_shanks}" for idx in range(self.n_shanks)]


class FakeEphysDataService:
    def __init__(self) -> None:
        self.loaded_probe = None

    def load_channel_table(self, probe: FakeProbeInfo) -> FakeChannelTable:
        self.loaded_probe = probe
        return FakeChannelTable(probe.num_shanks)


class FailingEphysDataService(FakeEphysDataService):
    def load_channel_table(self, probe: FakeProbeInfo) -> FakeChannelTable:
        raise RuntimeError(f"cannot load {probe.probe_name}")


class FakeOutputBuilder:
    def __init__(self, context: FakeDataContext) -> None:
        self.context = context

    def get_alignment_results(self, channel_locations_ras, channel_coordinates):
        return (
            {"channels": list(channel_locations_ras)},
            {"ccf_channels": list(channel_coordinates)},
            self.context.n_shanks > 1,
        )


class FakeBatchOutputBuilder(FakeOutputBuilder):
    def __init__(self, context: FakeDataContext) -> None:
        super().__init__(context)
        self.batched_alignments = None

    def get_alignment_results_batch(self, alignments):
        self.batched_alignments = alignments
        return {
            key: self.get_alignment_results(
                channel_locations_ras,
                channel_coordinates,
            )
            for key, (
                channel_locations_ras,
                channel_coordinates,
            ) in alignments.items()
        }


class FakeAlignmentRuntimeService:
    def __init__(self) -> None:
        self.calls = []

    def initialize_shank_runtime(self, shank_runtime, **kwargs):
        self.calls.append((shank_runtime, kwargs))
        shank_runtime.ephysalign = "alignment-engine"
        shank_runtime.region_fp = "region"
        shank_runtime.region_label_fp = "label"
        shank_runtime.region_colour_fp = "colour"
        shank_runtime.track_annos_and_ends_ras = np.array([[1.0, 2.0, 3.0]])
        return SimpleNamespace(
            feature_init=np.array([1.0, 2.0]),
            track_init=np.array([3.0, 4.0]),
            track_annos_and_ends_ras=np.array([[1.0, 2.0, 3.0]]),
        )


class FakeEphysAlignment:
    feature_init = np.array([1.0, 3.0])
    track_init = np.array([2.0, 4.0])

    @staticmethod
    def feature2track(depths_track, feature_ref, track_ref):
        return np.asarray(depths_track, dtype=float) + 10.0

    @staticmethod
    def adjust_extremes_uniform(feature, track):
        return np.asarray(track, dtype=float) + 1.0

    @staticmethod
    def adjust_extremes_linear(feature, track, extend_feature=1):
        return (
            np.asarray(feature, dtype=float) + extend_feature,
            np.asarray(track, dtype=float) + extend_feature,
        )


def make_controller(
    doc: AlignmentDocument | None = None,
    context: FakeDataContext | None = None,
    ephys_data_service: FakeEphysDataService | None = None,
    repo: FakeAlignmentRepository | None = None,
    alignment_runtime_service: FakeAlignmentRuntimeService | None = None,
    output_builder: FakeOutputBuilder | None = None,
) -> tuple[AlignmentController, FakeDataContext, FakeEphysDataService]:
    doc = doc or AlignmentDocument()
    context = context or FakeDataContext()
    ephys_data_service = ephys_data_service or FakeEphysDataService()
    controller = AlignmentController(
        doc,
        context,
        ephys_data_service,
        alignment_repository=repo,
        alignment_runtime_service=alignment_runtime_service,
        output_builder=output_builder,
    )
    return controller, context, ephys_data_service


class FakeAlignmentRepository:
    def __init__(self) -> None:
        self.loaded_alignments = None
        self.saved_kwargs = None

    def load_previous_alignments(self, **kwargs):
        self.loaded_kwargs = kwargs
        if self.loaded_alignments is None:
            return None
        return LoadedAlignmentHistory(self.loaded_alignments)

    def save_alignment_outputs(self, **kwargs):
        self.saved_kwargs = kwargs
        return SavedAlignmentOutputs(
            channel_results_path=kwargs["output_directory"] / "channel_locations.json",
            previous_alignments_path=kwargs["output_directory"]
            / "prev_alignments.json",
            ccf_channel_results_path=kwargs["output_directory"]
            / "ccf_channel_locations.json",
            docdb_probe_name="probeA_0" if kwargs["use_docdb"] else None,
        )


def test_set_mouse_root_updates_document(tmp_path):
    doc = AlignmentDocument()
    controller, _, _ = make_controller(doc)

    result = controller.set_mouse_root(tmp_path)

    assert isinstance(result, MouseRootLoaded)
    assert doc.mouse_root == tmp_path
    assert doc.mouse_id == "mouse1"
    assert not result.root_changed


def test_set_mouse_root_reports_root_changed(tmp_path):
    old_root = FakeMouseRoot(
        root=tmp_path / "old",
        mouse_id="old",
        sessions=[],
        probes={},
    )
    doc = AlignmentDocument()
    context = FakeDataContext(mouse_root=old_root)
    controller, _, _ = make_controller(doc, context=context)

    new_root = tmp_path / "new"
    new_root.mkdir()
    result = controller.set_mouse_root(new_root)

    assert isinstance(result, MouseRootLoaded)
    assert result.root_changed


def test_set_mouse_root_rejects_missing_directory(tmp_path):
    doc = AlignmentDocument()
    controller, _, _ = make_controller(doc)

    result = controller.set_mouse_root(tmp_path / "missing")

    assert isinstance(result, Failed)
    assert doc.mouse_root is None


def test_select_recording_clears_probe_and_returns_probes(tmp_path):
    doc = AlignmentDocument(selected_recording="old", selected_probe="probeZ")
    controller, _, _ = make_controller(doc)
    controller.set_mouse_root(tmp_path)

    result = controller.select_recording("rec1")

    assert isinstance(result, RecordingSelected)
    assert result.probes == ["probeA", "probeB"]
    assert not doc.probe_selected


def test_select_probe_loads_channel_info_and_derives_output(tmp_path):
    doc = AlignmentDocument()
    ephys_data_service = FakeEphysDataService()
    controller, _, _ = make_controller(doc, ephys_data_service=ephys_data_service)
    mouse_root = tmp_path / "mouse"
    mouse_root.mkdir()
    controller.set_mouse_root(mouse_root)

    output_root = tmp_path / "results"
    output_root.mkdir()
    controller.set_output_root(output_root)
    result = controller.select_probe("rec1", "probeA")

    assert isinstance(result, ProbeSelected)
    assert ephys_data_service.loaded_probe is not None
    assert ephys_data_service.loaded_probe.probe_name == "probeA"
    assert doc.selected_recording == "rec1"
    assert doc.selected_probe == "probeA"
    assert doc.channel_info_loaded
    assert result.shanks == ["1/2", "2/2"]
    assert result.n_shanks == 2
    assert result.output_directory == output_root / "rec1" / "probeA"
    assert result.output_directory.is_dir()
    assert doc.selected_alignment_key == AlignmentKey("rec1", "streamA", 0)
    assert doc.active_alignment_state is not None


def test_set_output_root_creates_missing_root_and_derives_output(tmp_path):
    doc = AlignmentDocument()
    controller, _, _ = make_controller(doc, ephys_data_service=FakeEphysDataService())
    mouse_root = tmp_path / "mouse"
    mouse_root.mkdir()
    controller.set_mouse_root(mouse_root)

    output_root = tmp_path / "new-results"
    root_result = controller.set_output_root(output_root)
    probe_result = controller.select_probe("rec1", "probeA")

    assert isinstance(root_result, OutputRootSet)
    assert output_root.is_dir()
    assert isinstance(probe_result, ProbeSelected)
    assert doc.output_directory == output_root / "rec1" / "probeA"
    assert isinstance(controller.can_load_data(), Ok)


def test_select_probe_can_restore_cached_stream_without_loading_channel_info(tmp_path):
    doc = AlignmentDocument()
    ephys_data_service = FakeEphysDataService()
    controller, context, _ = make_controller(
        doc,
        ephys_data_service=ephys_data_service,
    )
    mouse_root = tmp_path / "mouse"
    mouse_root.mkdir()
    controller.set_mouse_root(mouse_root)
    cached_stream = FakeCachedStream("rec1", "streamA", FakeChannelTable(2))

    result = controller.select_probe("rec1", "probeA", ephys_stream=cached_stream)

    assert isinstance(result, ProbeSelected)
    assert ephys_data_service.loaded_probe is None
    assert context.channel_table is cached_stream.channel_table
    assert doc.channel_info_loaded
    assert doc.selected_alignment_key == AlignmentKey("rec1", "streamA", 0)


def test_select_probe_failure_does_not_create_alignment_state(tmp_path):
    doc = AlignmentDocument()
    controller, _, _ = make_controller(
        doc,
        ephys_data_service=FailingEphysDataService(),
    )
    mouse_root = tmp_path / "mouse"
    mouse_root.mkdir()
    controller.set_mouse_root(mouse_root)

    result = controller.select_probe("rec1", "probeA")

    assert isinstance(result, Failed)
    assert not doc.channel_info_loaded
    assert doc.selected_alignment_key is None
    assert doc.alignment_states == {}


def test_output_root_does_not_derive_from_stale_loader_probe(tmp_path):
    doc = AlignmentDocument(selected_recording="rec1", selected_probe="probeA")
    context = FakeDataContext()
    context.probe_info = FakeProbeInfo("rec1", "probeB", "rec1:probeB", 1)
    controller, _, _ = make_controller(doc, context=context)

    output_root = tmp_path / "results"
    output_root.mkdir()
    result = controller.set_output_root(output_root)

    assert isinstance(result, OutputRootSet)
    assert result.output_directory is None
    assert doc.output_directory is None


def test_load_data_preparation_and_finish_updates_document():
    doc = AlignmentDocument(data_loaded=True, selected_shank=1)
    context = FakeDataContext()
    context.probe_info = FakeProbeInfo("rec1", "probeA", "rec1:probeA", 3)
    controller, _, _ = make_controller(doc, context=context)

    prepared = controller.prepare_load_data()
    controller.finish_load_data(shank_idx=2)

    assert prepared.preserve_plot_selection
    assert doc.data_loaded
    assert doc.selected_shank == 2
    assert doc.selected_alignment_key == AlignmentKey("rec1", "streamA", 2)


def test_set_selected_shank_updates_document_alignment_key(tmp_path):
    doc = AlignmentDocument()
    controller, _, _ = make_controller(doc)
    controller.set_mouse_root(tmp_path)
    controller.select_probe("rec1", "probeA")

    controller.set_selected_shank(1)

    assert doc.selected_alignment_key == AlignmentKey("rec1", "streamA", 1)
    assert doc.selected_shank == 1


def test_select_shank_returns_transition_metadata(tmp_path):
    doc = AlignmentDocument()
    controller, _, _ = make_controller(doc)
    controller.set_mouse_root(tmp_path)
    controller.select_probe("rec1", "probeA")
    doc.mark_data_loaded(True)
    previous_key = doc.selected_alignment_key

    result = controller.select_shank(1)

    assert isinstance(result, ShankSelected)
    assert result.previous_key == previous_key
    assert result.selected_key == AlignmentKey("rec1", "streamA", 1)
    assert result.previous_shank_idx == 0
    assert result.shank_idx == 1
    assert result.data_loaded
    assert doc.selected_shank == 1


def test_select_shank_rejects_out_of_range_when_channel_info_is_loaded(tmp_path):
    doc = AlignmentDocument()
    context = FakeDataContext()
    controller, _, _ = make_controller(doc, context=context)
    controller.set_mouse_root(tmp_path)
    controller.select_probe("rec1", "probeA")
    previous_key = doc.selected_alignment_key

    result = controller.select_shank(2)

    assert isinstance(result, Failed)
    assert "outside valid range" in result.message
    assert doc.selected_alignment_key == previous_key
    assert doc.selected_shank == 0


def test_can_load_data_delegates_to_policy(tmp_path):
    doc = AlignmentDocument(channel_info_loaded=True)
    doc.select_probe("rec1", "probeA")
    doc.set_channel_info_loaded(True)
    doc.set_output_directory(tmp_path / "rec1" / "probeA")
    controller, _, _ = make_controller(doc)

    assert isinstance(controller.can_load_data(), Ok)


def test_load_previous_alignments_uses_active_probe_and_repository(tmp_path):
    doc = AlignmentDocument()
    repo = FakeAlignmentRepository()
    repo.loaded_alignments = {"saved": [[1.0], [2.0]]}
    controller, _, _ = make_controller(doc, repo=repo)
    controller.set_mouse_root(tmp_path)
    controller.select_probe("rec1", "probeA")

    result = controller.load_previous_alignments(
        folder=tmp_path,
        shank_idx=1,
        use_docdb=True,
    )

    assert isinstance(result, PreviousAlignmentsLoaded)
    assert result.alignments == {"saved": [[1.0], [2.0]]}
    assert repo.loaded_kwargs["recording_id"] == "rec1"
    assert repo.loaded_kwargs["probe_name"] == "probeA"
    assert repo.loaded_kwargs["shank_idx"] == 1
    assert repo.loaded_kwargs["n_shanks"] == 2


def test_can_load_previous_alignments_requires_channel_info():
    controller, _, _ = make_controller(AlignmentDocument())

    assert isinstance(controller.can_load_previous_alignments(), Failed)


def test_load_previous_alignments_reports_empty_result(tmp_path):
    doc = AlignmentDocument()
    repo = FakeAlignmentRepository()
    controller, _, _ = make_controller(doc, repo=repo)
    controller.set_mouse_root(tmp_path)
    controller.select_probe("rec1", "probeA")

    result = controller.load_previous_alignments(
        folder=None,
        shank_idx=0,
        use_docdb=True,
    )

    assert isinstance(result, NoPreviousAlignments)


def test_set_previous_alignments_filters_legacy_auto_and_returns_choices() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    controller, _, _ = make_controller(doc)

    result = controller.set_previous_alignments(
        {
            "auto": [[100.0], [200.0]],
            "saved": [[1.0], [2.0]],
        },
        shank_idx=0,
    )

    assert isinstance(result, AlignmentChoicesUpdated)
    assert result.choices == ["saved", "original"]
    state = doc.active_alignment_state
    assert state is not None
    assert state.alignments == {"saved": [[1.0], [2.0]]}


def test_set_pending_reference_lines_updates_document_state() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    controller, _, _ = make_controller(doc)

    result = controller.set_pending_reference_lines(
        feature_positions_um=np.array([0.0, 1.0]),
        track_positions_um=np.array([2.0, 3.0]),
        shank_idx=0,
    )

    assert isinstance(result, PendingReferenceLinesUpdated)
    assert result.lines is not None
    np.testing.assert_array_equal(result.lines.feature_positions_um, [0.0, 1.0])
    np.testing.assert_array_equal(result.lines.track_positions_um, [2.0, 3.0])
    state = doc.active_alignment_state
    assert state is not None
    assert state.pending_reference_lines is result.lines
    assert state.prev_align == ["original"]


def test_select_previous_alignment_rebases_working_state_and_clears_lines() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    doc.set_active_alignments({"saved": [[1.0, 2.0], [3.0, 4.0]]})
    controller, _, _ = make_controller(doc)
    controller.set_pending_reference_lines(
        feature_positions_um=np.array([9.0]),
        track_positions_um=np.array([10.0]),
        shank_idx=0,
    )

    result = controller.select_previous_alignment(0, shank_idx=0)

    assert isinstance(result, PreviousAlignmentSelected)
    assert result.choice == "saved"
    np.testing.assert_array_equal(result.feature_prev, [1.0, 2.0])
    np.testing.assert_array_equal(result.track_prev, [3.0, 4.0])
    state = doc.active_alignment_state
    assert state is not None
    np.testing.assert_array_equal(state.feature_prev, [1.0, 2.0])
    np.testing.assert_array_equal(state.track_prev, [3.0, 4.0])
    assert state.active_alignment is not None
    np.testing.assert_array_equal(state.active_alignment.feature, [1.0, 2.0])
    np.testing.assert_array_equal(state.active_alignment.track, [3.0, 4.0])
    assert state.pending_reference_lines is None


def test_initialize_shank_runtime_seeds_empty_document_state() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    runtime_service = FakeAlignmentRuntimeService()
    controller, _, _ = make_controller(
        doc,
        alignment_runtime_service=runtime_service,
    )
    shank_runtime = SimpleNamespace(shank_idx=0, chn_depths=np.array([10.0, 20.0]))

    result = controller.initialize_shank_runtime(
        shank_runtime,
        track_annotations_ras=np.array([[0.0, 0.0, 0.0]]),
        brain_atlas="atlas",
    )

    assert isinstance(result, ShankRuntimeInitialized)
    assert result.seeded_document_alignment
    state = doc.active_alignment_state
    assert state is not None
    assert state.active_alignment is not None
    np.testing.assert_array_equal(state.active_alignment.feature, [1.0, 2.0])
    np.testing.assert_array_equal(state.active_alignment.track, [3.0, 4.0])
    assert shank_runtime.ephysalign == "alignment-engine"
    assert runtime_service.calls[0][1]["brain_atlas"] == "atlas"


def test_initialize_shank_runtime_preserves_existing_alignment() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    state = doc.active_alignment_state
    assert state is not None
    state.feature_prev = np.array([5.0, 6.0])
    state.track_prev = np.array([7.0, 8.0])
    state.active_alignment = ActiveAlignment(
        np.array([9.0, 10.0]),
        np.array([11.0, 12.0]),
    )
    runtime_service = FakeAlignmentRuntimeService()
    controller, _, _ = make_controller(
        doc,
        alignment_runtime_service=runtime_service,
    )

    result = controller.initialize_shank_runtime(
        SimpleNamespace(shank_idx=0, chn_depths=np.array([10.0, 20.0])),
        track_annotations_ras=np.array([[0.0, 0.0, 0.0]]),
        brain_atlas="atlas",
    )

    assert isinstance(result, ShankRuntimeInitialized)
    assert not result.seeded_document_alignment
    assert state.active_alignment is not None
    np.testing.assert_array_equal(state.active_alignment.feature, [9.0, 10.0])
    np.testing.assert_array_equal(state.active_alignment.track, [11.0, 12.0])
    np.testing.assert_array_equal(
        runtime_service.calls[0][1]["feature_prev"],
        [5.0, 6.0],
    )
    np.testing.assert_array_equal(
        runtime_service.calls[0][1]["track_prev"],
        [7.0, 8.0],
    )


def test_initialize_shank_runtime_rejects_shank_mismatch() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    controller, _, _ = make_controller(
        doc,
        alignment_runtime_service=FakeAlignmentRuntimeService(),
    )

    result = controller.initialize_shank_runtime(
        SimpleNamespace(shank_idx=1, chn_depths=np.array([10.0, 20.0])),
        track_annotations_ras=np.array([[0.0, 0.0, 0.0]]),
        brain_atlas="atlas",
    )

    assert isinstance(result, Failed)
    assert "does not match" in result.message


def test_fit_alignment_to_reference_lines_updates_active_document_state() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    state = doc.active_alignment_state
    assert state is not None
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
    )
    controller, _, _ = make_controller(doc)
    shank_runtime = SimpleNamespace(shank_idx=0, ephysalign=FakeEphysAlignment())

    result = controller.fit_alignment_to_reference_lines(
        shank_runtime,
        line_features_um=np.array([2_000_000.0]),
        line_tracks_um=np.array([12_000_000.0]),
        lin_fit=False,
        extend_feature=2,
    )

    assert isinstance(result, AlignmentEditApplied)
    np.testing.assert_array_equal(result.alignment.feature, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(result.alignment.track, [21.0, 23.0, 25.0])
    assert result.lin_fit is False
    assert state.active_alignment is not None
    np.testing.assert_array_equal(state.active_alignment.feature, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(state.active_alignment.track, [21.0, 23.0, 25.0])


def test_offset_alignment_from_tip_updates_active_document_state() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    state = doc.active_alignment_state
    assert state is not None
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
        lin_fit=True,
    )
    controller, _, _ = make_controller(doc)

    result = controller.offset_alignment_from_tip(
        tip_position_um=100.0,
        probe_tip_um=0.0,
        lin_fit=False,
    )

    assert isinstance(result, AlignmentEditApplied)
    np.testing.assert_array_equal(result.alignment.feature, [0.0, 4.0])
    np.testing.assert_allclose(result.alignment.track, [10.0001, 14.0001])
    assert result.lin_fit is False
    assert state.active_alignment is not None
    np.testing.assert_allclose(state.active_alignment.track, [10.0001, 14.0001])


def test_offset_alignment_from_tip_rejects_shank_mismatch() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    state = doc.active_alignment_state
    assert state is not None
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
    )
    controller, _, _ = make_controller(doc)

    result = controller.offset_alignment_from_tip(
        tip_position_um=100.0,
        probe_tip_um=0.0,
        lin_fit=False,
        shank_idx=1,
    )

    assert isinstance(result, Failed)
    assert "does not match" in result.message


def test_go_previous_and_next_alignment_update_active_document_state() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    state = doc.active_alignment_state
    assert state is not None
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
        lin_fit=True,
    )
    controller, _, _ = make_controller(doc)
    controller.offset_alignment_from_tip(
        tip_position_um=100.0,
        probe_tip_um=0.0,
        lin_fit=False,
    )

    previous_result = controller.go_previous_alignment()

    assert isinstance(previous_result, AlignmentEditApplied)
    np.testing.assert_array_equal(previous_result.alignment.track, [10.0, 14.0])
    assert previous_result.lin_fit is True
    assert state.active_alignment is not None
    np.testing.assert_array_equal(state.active_alignment.track, [10.0, 14.0])

    next_result = controller.go_next_alignment()

    assert isinstance(next_result, AlignmentEditApplied)
    np.testing.assert_allclose(next_result.alignment.track, [10.0001, 14.0001])
    assert next_result.lin_fit is False
    assert state.active_alignment is not None
    np.testing.assert_allclose(state.active_alignment.track, [10.0001, 14.0001])


def test_reset_alignment_to_initial_updates_active_document_state() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    state = doc.active_alignment_state
    assert state is not None
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
    )
    controller, _, _ = make_controller(doc)
    shank_runtime = SimpleNamespace(shank_idx=0, ephysalign=FakeEphysAlignment())

    result = controller.reset_alignment_to_initial(shank_runtime, lin_fit=False)

    assert isinstance(result, AlignmentEditApplied)
    np.testing.assert_array_equal(result.alignment.feature, [1.0, 3.0])
    np.testing.assert_array_equal(result.alignment.track, [2.0, 4.0])
    assert result.lin_fit is False
    assert state.active_alignment is not None
    np.testing.assert_array_equal(state.active_alignment.feature, [1.0, 3.0])
    np.testing.assert_array_equal(state.active_alignment.track, [2.0, 4.0])


def test_fit_alignment_to_reference_lines_requires_runtime_alignment() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    controller, _, _ = make_controller(doc)

    result = controller.fit_alignment_to_reference_lines(
        SimpleNamespace(shank_idx=0, ephysalign=None),
        line_features_um=np.array([2_000_000.0]),
        line_tracks_um=np.array([12_000_000.0]),
        lin_fit=False,
        extend_feature=2,
    )

    assert isinstance(result, Failed)
    assert "not initialized" in result.message


def test_fit_alignment_to_reference_lines_rejects_shank_mismatch() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    controller, _, _ = make_controller(doc)

    result = controller.fit_alignment_to_reference_lines(
        SimpleNamespace(shank_idx=1, ephysalign=FakeEphysAlignment()),
        line_features_um=np.array([2_000_000.0]),
        line_tracks_um=np.array([12_000_000.0]),
        lin_fit=False,
        extend_feature=2,
    )

    assert isinstance(result, Failed)
    assert "does not match" in result.message


def test_can_save_alignment_output_requires_output_directory():
    controller, _, _ = make_controller(AlignmentDocument())

    assert isinstance(controller.can_save_alignment_output(), Blocked)


def test_build_and_save_alignment_output_filters_auto(tmp_path):
    doc = AlignmentDocument(output_directory=tmp_path)
    context = FakeDataContext()
    context.probe_info = FakeProbeInfo("rec1", "probeA", "rec1:probeA", 2)
    context.channel_table = FakeChannelTable(2)
    repo = FakeAlignmentRepository()
    output_builder = FakeOutputBuilder(context)
    controller, _, _ = make_controller(
        doc,
        context=context,
        repo=repo,
        output_builder=output_builder,
    )

    output = controller.build_alignment_output([1, 2], [3, 4])
    assert isinstance(output, AlignmentOutputBuilt)
    saved = controller.save_alignment_output(
        output,
        alignments={"auto": [[0], [0]], "saved": [[1], [2]]},
        shank_idx=1,
        use_docdb=False,
    )

    assert isinstance(saved, AlignmentOutputsSaved)
    assert saved.previous_alignments == {"saved": [[1], [2]]}
    assert repo.saved_kwargs["previous_alignments"] == {"saved": [[1], [2]]}
    assert repo.saved_kwargs["multi_shank"]
    assert repo.saved_kwargs["shank_idx"] == 1


def test_build_alignment_outputs_uses_batch_builder(tmp_path):
    doc = AlignmentDocument(output_directory=tmp_path)
    context = FakeDataContext()
    context.probe_info = FakeProbeInfo("rec1", "probeA", "rec1:probeA", 2)
    context.channel_table = FakeChannelTable(2)
    output_builder = FakeBatchOutputBuilder(context)
    controller, _, _ = make_controller(
        doc,
        context=context,
        output_builder=output_builder,
    )
    shank0 = AlignmentKey("rec1", "streamA", 0)
    shank1 = AlignmentKey("rec1", "streamA", 1)

    outputs = controller.build_alignment_outputs(
        {
            shank0: ([1, 2], [3, 4]),
            shank1: ([5, 6], [7, 8]),
        }
    )

    assert not isinstance(outputs, Failed)
    assert output_builder.batched_alignments is not None
    assert set(outputs) == {shank0, shank1}
    assert outputs[shank0].channel_results == {"channels": [1, 2]}
    assert outputs[shank1].ccf_channel_results == {"ccf_channels": [7, 8]}
