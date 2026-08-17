"""Tests for application-level autosave checkpoint commands."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.application.commands.autosave import (
    AUTOSAVE_DIRECTORY_NAME,
    AUTOSAVE_DOCUMENT_FILENAME,
    AutosaveCheckpointCommandHandler,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointCleared,
    AutosaveCheckpointRead,
    AutosaveCheckpointRestored,
    AutosaveCheckpointWritten,
)
from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.core.workflow import Failed


def test_autosave_checkpoint_writes_reads_restores_and_clears_default_path(
    tmp_path,
) -> None:
    package_dir = tmp_path / "ibl_annotations_mouse"
    document = AlignmentDocument(
        mouse_id="mouse",
        output_package_directory=package_dir,
        dirty=True,
    )
    key = AlignmentKey("rec", "stream", 0)
    state = document.select_alignment_key(key)
    state.active_alignment = ActiveAlignment(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )
    state.mark_alignment_changed()
    handler = AutosaveCheckpointCommandHandler(AlignmentController(document))

    written = handler.write_checkpoint()

    assert isinstance(written, AutosaveCheckpointWritten)
    expected_path = package_dir / AUTOSAVE_DIRECTORY_NAME / AUTOSAVE_DOCUMENT_FILENAME
    assert written.path == expected_path
    assert written.alignment_state_count == 1
    assert expected_path.exists()

    read = handler.read_checkpoint()

    assert isinstance(read, AutosaveCheckpointRead)
    assert read.path == expected_path
    assert len(read.snapshot.alignment_states) == 1

    document.mouse_id = "changed"
    document.selected_alignment_key = None
    document.alignment_states.clear()
    document.dirty = False
    live_state_dict = document.alignment_states

    restored = handler.restore_checkpoint()

    assert isinstance(restored, AutosaveCheckpointRestored)
    assert restored.path == expected_path
    assert restored.selected_alignment_key == key
    assert restored.alignment_state_count == 1
    assert document.alignment_states is live_state_dict
    assert document.mouse_id == "mouse"
    assert document.selected_alignment_key == key
    assert document.has_unsaved_alignments
    restored_state = document.alignment_state_for(key)
    assert restored_state.active_alignment is not None
    np.testing.assert_array_equal(restored_state.active_alignment.feature, [1.0, 2.0])
    np.testing.assert_array_equal(restored_state.active_alignment.track, [3.0, 4.0])

    cleared = handler.clear_checkpoint()

    assert isinstance(cleared, AutosaveCheckpointCleared)
    assert cleared.path == expected_path
    assert cleared.existed
    assert not expected_path.exists()

    cleared_again = handler.clear_checkpoint()

    assert isinstance(cleared_again, AutosaveCheckpointCleared)
    assert not cleared_again.existed


def test_autosave_checkpoint_default_path_requires_output_package(tmp_path) -> None:
    document = AlignmentDocument()
    handler = AutosaveCheckpointCommandHandler(AlignmentController(document))

    result = handler.write_checkpoint()

    assert isinstance(result, Failed)
    assert "No alignment output package" in result.message

    explicit_path = tmp_path / "manual" / "alignment_document.json"
    explicit_result = handler.write_checkpoint(explicit_path)

    assert isinstance(explicit_result, AutosaveCheckpointWritten)
    assert explicit_result.path == explicit_path
    assert explicit_path.exists()


def test_autosave_checkpoint_read_reports_missing_checkpoint(tmp_path) -> None:
    document = AlignmentDocument(
        output_package_directory=tmp_path / "ibl_annotations_mouse",
    )
    handler = AutosaveCheckpointCommandHandler(AlignmentController(document))

    result = handler.read_checkpoint()

    assert isinstance(result, Failed)
    assert "No autosave checkpoint found" in result.message
