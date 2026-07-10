"""Tests for Qt-free alignment controller commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ephys_alignment_gui.alignment_repository import (
    LoadedAlignmentHistory,
    SavedAlignmentOutputs,
)
from ephys_alignment_gui.controller import (
    AlignmentController,
    AlignmentOutputBuilt,
    AlignmentOutputsSaved,
    MouseRootLoaded,
    NoPreviousAlignments,
    OutputRootSet,
    PreviousAlignmentsLoaded,
    ProbeSelected,
    RecordingSelected,
)
from ephys_alignment_gui.document import AlignmentDocument
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


class FakeOutputBuilder:
    def __init__(self, context: FakeDataContext) -> None:
        self.context = context

    def get_alignment_results(self, channel_locations_ras, channel_coordinates):
        return (
            {"channels": list(channel_locations_ras)},
            {"ccf_channels": list(channel_coordinates)},
            self.context.n_shanks > 1,
        )


def make_controller(
    doc: AlignmentDocument | None = None,
    context: FakeDataContext | None = None,
    ephys_data_service: FakeEphysDataService | None = None,
    repo: FakeAlignmentRepository | None = None,
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
    controller, _, _ = make_controller(doc)

    prepared = controller.prepare_load_data()
    controller.finish_load_data(shank_idx=2)

    assert prepared.preserve_plot_selection
    assert doc.data_loaded
    assert doc.selected_shank == 2


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
