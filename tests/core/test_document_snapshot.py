"""Tests for alignment document snapshots used by autosave."""

from __future__ import annotations

import json

import numpy as np

from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_state import PendingReferenceLines
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.core.document_snapshot import AlignmentDocumentSnapshot


def test_document_snapshot_roundtrips_document_owned_state(tmp_path) -> None:
    doc = AlignmentDocument(
        mouse_root=tmp_path / "input",
        mouse_id="776259",
        selected_recording="rec-a",
        selected_probe="probe-a",
        selected_shank=1,
        output_root=tmp_path / "results",
        output_package_directory=tmp_path / "results" / "ibl_annotations_776259",
        output_directory=tmp_path / "results" / "ibl_annotations_776259" / "rec-a",
        channel_info_loaded=True,
        data_loaded=True,
        dirty=True,
    )
    active_key = AlignmentKey("rec-a", "stream-a", 1)
    doc.select_alignment_key(active_key)
    state = doc.alignment_state_for(active_key)
    state.set_alignments(
        {
            "2026-08-16T12:00:00": [[0.0, 1.0], [0.5, 1.5]],
            "2026-08-16T12:05:00": [[2.0, 3.0], [2.5, 3.5]],
        }
    )
    state.feature_prev = np.array([0.0, 1.0])
    state.track_prev = np.array([0.5, 1.5])
    state.active_alignment = ActiveAlignment(
        np.array([10.0, 20.0]),
        np.array([11.0, 21.0]),
        lin_fit=True,
    )
    state.set_pending_reference_lines(
        PendingReferenceLines(
            np.array([100.0, 200.0]),
            np.array([110.0, 210.0]),
        )
    )
    state.mark_alignment_changed()

    clean_key = AlignmentKey("rec-b", "stream-b", 0)
    clean_state = doc.alignment_state_for(clean_key)
    clean_state.active_alignment = ActiveAlignment(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )
    clean_state.mark_alignment_changed()
    clean_state.mark_saved()

    snapshot = AlignmentDocumentSnapshot.from_document(doc)
    restored = snapshot.restore_document()

    assert restored.mouse_root == tmp_path / "input"
    assert restored.mouse_id == "776259"
    assert restored.selected_alignment_key == active_key
    assert restored.selected_recording == "rec-a"
    assert restored.selected_probe == "probe-a"
    assert restored.selected_shank == 1
    assert restored.output_root == tmp_path / "results"
    assert restored.output_package_directory == (
        tmp_path / "results" / "ibl_annotations_776259"
    )
    assert restored.output_directory == (
        tmp_path / "results" / "ibl_annotations_776259" / "rec-a"
    )
    assert restored.channel_info_loaded
    assert restored.data_loaded
    assert restored.dirty
    assert list(restored.alignment_states) == [active_key, clean_key]

    restored_state = restored.alignment_state_for(active_key)
    assert restored_state.prev_align == [
        "2026-08-16T12:05:00",
        "2026-08-16T12:00:00",
        "original",
    ]
    np.testing.assert_array_equal(restored_state.feature_prev, [0.0, 1.0])
    np.testing.assert_array_equal(restored_state.track_prev, [0.5, 1.5])
    assert restored_state.active_alignment is not None
    np.testing.assert_array_equal(
        restored_state.active_alignment.feature,
        [10.0, 20.0],
    )
    np.testing.assert_array_equal(
        restored_state.active_alignment.track,
        [11.0, 21.0],
    )
    assert restored_state.active_alignment.lin_fit
    assert restored_state.pending_reference_lines is not None
    np.testing.assert_array_equal(
        restored_state.pending_reference_lines.feature_positions_um,
        [100.0, 200.0],
    )
    np.testing.assert_array_equal(
        restored_state.pending_reference_lines.warped_positions_um,
        [110.0, 210.0],
    )
    assert restored_state.save_state.revision == 1
    assert restored_state.has_unsaved_alignment

    restored_clean = restored.alignment_state_for(clean_key)
    assert restored_clean.active_alignment is not None
    assert not restored_clean.has_unsaved_alignment


def test_document_snapshot_writes_and_reads_json_atomically(tmp_path) -> None:
    doc = AlignmentDocument(
        mouse_root=tmp_path / "input",
        mouse_id="mouse",
        output_root=tmp_path / "results",
        dirty=True,
    )
    key = AlignmentKey("rec", "stream", 0)
    state = doc.select_alignment_key(key)
    state.active_alignment = ActiveAlignment(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )
    state.mark_alignment_changed()

    path = tmp_path / "autosave" / "alignment_document.json"
    AlignmentDocumentSnapshot.from_document(doc).write_json(path)

    assert path.exists()
    assert not (path.parent / ".alignment_document.json.tmp").exists()
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    assert raw["schema_version"] == 1
    assert raw["alignment_states"][0]["key"] == {
        "recording_id": "rec",
        "ephys_collection": "stream",
        "shank_idx": 0,
    }

    restored = AlignmentDocumentSnapshot.read_json(path).restore_document()

    assert restored.mouse_root == tmp_path / "input"
    assert restored.output_root == tmp_path / "results"
    assert restored.selected_alignment_key == key
    assert restored.has_unsaved_alignments
