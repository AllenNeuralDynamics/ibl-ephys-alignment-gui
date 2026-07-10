"""Tests for Qt-free alignment controller commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ephys_alignment_gui.controller import (
    AlignmentController,
    MouseRootLoaded,
    OutputRootSet,
    ProbeSelected,
    RecordingSelected,
)
from ephys_alignment_gui.document import AlignmentDocument
from ephys_alignment_gui.workflow import Failed, Ok


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


class FakeLoader:
    def __init__(self, mouse_root: FakeMouseRoot | None = None) -> None:
        self.mouse_root = mouse_root
        self.probe_info: FakeProbeInfo | None = None
        self.n_shanks = 0
        self.load_channel_info_called = False

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
        self.n_shanks = 0
        return self.mouse_root

    def list_probes(self, recording_id: str) -> list[str]:
        assert self.mouse_root is not None
        return sorted(self.mouse_root.probes[recording_id].keys())

    def select_probe(self, recording_id: str, probe_name: str) -> FakeProbeInfo:
        assert self.mouse_root is not None
        probe = self.mouse_root.probes[recording_id][probe_name]
        self.probe_info = probe
        self.n_shanks = probe.num_shanks
        return probe

    def load_channel_info(self) -> None:
        if self.probe_info is None:
            raise RuntimeError("no probe selected")
        self.load_channel_info_called = True

    def get_shank_list(self) -> list[str] | None:
        if self.n_shanks == 1:
            return ["1/1"]
        return [f"{idx + 1}/{self.n_shanks}" for idx in range(self.n_shanks)]


def test_set_mouse_root_updates_document(tmp_path):
    doc = AlignmentDocument()
    loader = FakeLoader()
    controller = AlignmentController(doc, loader)

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
    loader = FakeLoader(mouse_root=old_root)
    controller = AlignmentController(doc, loader)

    new_root = tmp_path / "new"
    new_root.mkdir()
    result = controller.set_mouse_root(new_root)

    assert isinstance(result, MouseRootLoaded)
    assert result.root_changed


def test_set_mouse_root_rejects_missing_directory(tmp_path):
    doc = AlignmentDocument()
    controller = AlignmentController(doc, FakeLoader())

    result = controller.set_mouse_root(tmp_path / "missing")

    assert isinstance(result, Failed)
    assert doc.mouse_root is None


def test_select_recording_clears_probe_and_returns_probes(tmp_path):
    doc = AlignmentDocument(selected_recording="old", selected_probe="probeZ")
    loader = FakeLoader()
    controller = AlignmentController(doc, loader)
    controller.set_mouse_root(tmp_path)

    result = controller.select_recording("rec1")

    assert isinstance(result, RecordingSelected)
    assert result.probes == ["probeA", "probeB"]
    assert not doc.probe_selected


def test_select_probe_loads_channel_info_and_derives_output(tmp_path):
    doc = AlignmentDocument()
    loader = FakeLoader()
    controller = AlignmentController(doc, loader)
    mouse_root = tmp_path / "mouse"
    mouse_root.mkdir()
    controller.set_mouse_root(mouse_root)

    output_root = tmp_path / "results"
    output_root.mkdir()
    controller.set_output_root(output_root)
    result = controller.select_probe("rec1", "probeA")

    assert isinstance(result, ProbeSelected)
    assert loader.load_channel_info_called
    assert doc.selected_recording == "rec1"
    assert doc.selected_probe == "probeA"
    assert doc.channel_info_loaded
    assert result.shanks == ["1/2", "2/2"]
    assert result.n_shanks == 2
    assert result.output_directory == output_root / "rec1" / "probeA"
    assert result.output_directory.is_dir()


def test_output_root_does_not_derive_from_stale_loader_probe(tmp_path):
    doc = AlignmentDocument(selected_recording="rec1", selected_probe="probeA")
    loader = FakeLoader()
    loader.probe_info = FakeProbeInfo("rec1", "probeB", "rec1:probeB", 1)
    controller = AlignmentController(doc, loader)

    output_root = tmp_path / "results"
    output_root.mkdir()
    result = controller.set_output_root(output_root)

    assert isinstance(result, OutputRootSet)
    assert result.output_directory is None
    assert doc.output_directory is None


def test_load_data_preparation_and_finish_updates_document():
    doc = AlignmentDocument(data_loaded=True, selected_shank=1)
    controller = AlignmentController(doc, FakeLoader())

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
    controller = AlignmentController(doc, FakeLoader())

    assert isinstance(controller.can_load_data(), Ok)
