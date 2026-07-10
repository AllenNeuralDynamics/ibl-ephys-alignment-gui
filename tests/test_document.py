"""Tests for the Qt-free alignment document model."""

from __future__ import annotations

from pathlib import Path

from ephys_alignment_gui.document import AlignmentDocument


def test_document_records_mouse_root_and_clears_probe_state(tmp_path):
    doc = AlignmentDocument()
    doc.set_output_root(tmp_path / "results")
    doc.select_probe("rec1", "probeA")
    doc.set_channel_info_loaded(True)
    doc.mark_data_loaded(True)
    doc.set_output_directory(tmp_path / "rec1" / "probeA")

    doc.set_mouse_root(tmp_path / "mouse42", mouse_id="mouse42")

    assert doc.mouse_root == tmp_path / "mouse42"
    assert doc.mouse_id == "mouse42"
    assert not doc.probe_selected
    assert not doc.channel_info_loaded
    assert not doc.data_loaded
    assert doc.output_root == tmp_path / "results"
    assert doc.output_directory is None


def test_select_probe_resets_probe_derived_state(tmp_path):
    doc = AlignmentDocument(
        output_root=tmp_path / "results",
        output_directory=tmp_path / "old",
        channel_info_loaded=True,
        data_loaded=True,
        dirty=True,
    )

    doc.select_probe("rec1", "probeA")

    assert doc.selected_recording == "rec1"
    assert doc.selected_probe == "probeA"
    assert doc.selected_shank == 0
    assert doc.probe_selected
    assert not doc.channel_info_loaded
    assert not doc.data_loaded
    assert not doc.dirty
    assert doc.output_root == tmp_path / "results"
    assert doc.output_directory is None


def test_channel_info_unloaded_clears_data_loaded():
    doc = AlignmentDocument(channel_info_loaded=True, data_loaded=True)

    doc.set_channel_info_loaded(False)

    assert not doc.channel_info_loaded
    assert not doc.data_loaded


def test_output_paths_are_stored_as_paths(tmp_path):
    doc = AlignmentDocument()

    doc.set_output_root(tmp_path)
    doc.set_output_directory(tmp_path / "rec1" / "probeA")

    assert doc.output_root == Path(tmp_path)
    assert doc.output_directory == tmp_path / "rec1" / "probeA"
