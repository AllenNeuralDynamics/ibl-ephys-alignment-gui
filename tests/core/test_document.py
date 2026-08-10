"""Tests for the Qt-free alignment document model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey


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


def test_alignment_key_rejects_negative_shank() -> None:
    with pytest.raises(ValueError, match="shank_idx"):
        AlignmentKey("rec1", "streamA", -1)


def test_select_alignment_key_creates_and_selects_state() -> None:
    doc = AlignmentDocument(selected_probe="probeA")
    key = AlignmentKey("rec1", "streamA", 1)

    state = doc.select_alignment_key(key)

    assert doc.selected_alignment_key == key
    assert doc.selected_recording == "rec1"
    assert doc.selected_shank == 1
    assert doc.active_alignment_state is state
    assert doc.alignment_state_for(key) is state


def test_alignment_states_are_isolated_by_key() -> None:
    doc = AlignmentDocument()
    shank0 = AlignmentKey("rec1", "streamA", 0)
    shank1 = AlignmentKey("rec1", "streamA", 1)

    doc.select_alignment_key(shank0)
    doc.active_add_alignment(np.array([0.0]), np.array([1.0]))
    doc.select_alignment_key(shank1)

    assert doc.active_prev_align == ["original"]
    assert len(doc.alignment_state_for(shank0).alignments) == 1
    assert doc.alignment_state_for(shank1).alignments == {}


def test_alignment_states_for_current_probe_filters_by_stream() -> None:
    doc = AlignmentDocument()
    active = AlignmentKey("rec1", "streamA", 0)
    same_stream = AlignmentKey("rec1", "streamA", 1)
    other_stream = AlignmentKey("rec1", "streamB", 0)
    other_recording = AlignmentKey("rec2", "streamA", 0)

    active_state = doc.select_alignment_key(active)
    same_stream_state = doc.alignment_state_for(same_stream)
    doc.alignment_state_for(other_stream)
    doc.alignment_state_for(other_recording)

    assert doc.alignment_states_for_current_probe() == {
        active: active_state,
        same_stream: same_stream_state,
    }


def test_set_selected_shank_updates_active_alignment_key() -> None:
    doc = AlignmentDocument(selected_probe="probeA")
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))

    doc.set_selected_shank(2)

    assert doc.selected_alignment_key == AlignmentKey("rec1", "streamA", 2)
    assert doc.selected_shank == 2
    assert doc.active_prev_align == ["original"]


def test_active_alignment_history_helpers_roundtrip() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    feature = np.array([0.0, 1.0])
    track = np.array([2.0, 3.0])

    key = doc.active_add_alignment(feature, track)

    assert doc.active_prev_align == [key, "original"]
    saved_feature, saved_track = doc.active_get_alignment_idx(0)
    np.testing.assert_array_equal(saved_feature, feature)
    np.testing.assert_array_equal(saved_track, track)


def test_active_pending_reference_lines_are_separate_from_saved_history() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    doc.active_add_alignment(np.array([0.0]), np.array([1.0]))

    lines = doc.active_set_pending_reference_lines(
        np.array([2.0, 3.0]),
        np.array([4.0, 5.0]),
    )

    assert doc.active_prev_align is not None
    assert len(doc.active_prev_align) == 2
    assert doc.active_prev_align[-1] == "original"
    assert doc.active_alignments is not None
    assert "auto" not in doc.active_alignments
    assert lines is not None
    np.testing.assert_array_equal(lines.feature_positions_um, [2.0, 3.0])
    np.testing.assert_array_equal(lines.track_positions_um, [4.0, 5.0])
    state = doc.active_alignment_state
    assert state is not None
    assert state.pending_reference_lines is lines

    doc.active_clear_pending_reference_lines()

    assert state.pending_reference_lines is None


def test_alignment_registry_survives_probe_clear_but_can_clear_on_new_root(
    tmp_path,
) -> None:
    doc = AlignmentDocument(selected_probe="probeA")
    key = AlignmentKey("rec1", "streamA", 0)
    doc.select_alignment_key(key)
    doc.active_add_alignment(np.array([0.0]), np.array([1.0]))

    doc.clear_probe()

    assert doc.selected_alignment_key is None
    assert key in doc.alignment_states

    doc.set_mouse_root(tmp_path / "mouse", clear_alignment_states=True)

    assert doc.alignment_states == {}
