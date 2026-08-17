"""Tests for autosave checkpoint trigger points in app commands."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.application.alignment_save_job import (
    AlignmentSaveJobCompleted,
    PreparedAlignmentSave,
)
from ephys_alignment_gui.application.results import (
    AlignmentEditApplied,
    AutosaveCheckpointWritten,
    EditedAlignmentOutputsSaved,
    ShankSelected,
)
from ephys_alignment_gui.application.workspace import AlignmentWorkspace
from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.core.document_snapshot import AlignmentDocumentSnapshot


def test_alignment_edit_writes_autosave_checkpoint_when_package_exists(
    tmp_path,
) -> None:
    workspace = AlignmentWorkspace()
    package_dir = tmp_path / "ibl_annotations_mouse"
    workspace.document.set_output_package_directory(package_dir)
    key = AlignmentKey("rec", "stream", 0)
    state = workspace.document.select_alignment_key(key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )

    result = workspace.app.commands.edit.offset_alignment_from_tip(
        tip_position_um=100.0,
        probe_tip_um=0.0,
        lin_fit=False,
    )

    assert isinstance(result, AlignmentEditApplied)
    checkpoint_path = package_dir / "autosave" / "alignment_document.json"
    assert checkpoint_path.exists()
    restored = AlignmentDocumentSnapshot.read_json(checkpoint_path).restore_document()
    restored_state = restored.alignment_state_for(key)
    assert restored_state.active_alignment is not None
    assert restored_state.has_unsaved_alignment
    np.testing.assert_allclose(
        restored_state.active_alignment.track,
        [0.0001, 1.0001],
    )


def test_outgoing_reference_line_capture_checkpoints_before_shank_switch(
    tmp_path,
) -> None:
    workspace = AlignmentWorkspace()
    package_dir = tmp_path / "ibl_annotations_mouse"
    workspace.document.set_output_package_directory(package_dir)
    key = AlignmentKey("rec", "stream", 0)
    workspace.document.select_alignment_key(key)
    workspace.document.mark_data_loaded(True)

    result = workspace.app.commands.shanks.select_shank(
        1,
        outgoing_reference_lines=([100.0], [200.0]),
    )

    assert isinstance(result, ShankSelected)
    assert workspace.document.selected_alignment_key == AlignmentKey("rec", "stream", 1)
    checkpoint_path = package_dir / "autosave" / "alignment_document.json"
    snapshot = AlignmentDocumentSnapshot.read_json(checkpoint_path)
    restored = snapshot.restore_document()
    restored_state = restored.alignment_state_for(key)
    assert restored.selected_alignment_key == key
    assert restored_state.pending_reference_lines is not None
    np.testing.assert_array_equal(
        restored_state.pending_reference_lines.feature_positions_um,
        [100.0],
    )
    np.testing.assert_array_equal(
        restored_state.pending_reference_lines.warped_positions_um,
        [200.0],
    )


def test_successful_save_publication_clears_autosave_checkpoint(tmp_path) -> None:
    workspace = AlignmentWorkspace()
    package_dir = tmp_path / "ibl_annotations_mouse"
    workspace.document.set_output_package_directory(package_dir)
    written = workspace.app.commands.autosave.write_checkpoint()
    assert isinstance(written, AutosaveCheckpointWritten)
    checkpoint_path = package_dir / "autosave" / "alignment_document.json"
    assert checkpoint_path.exists()

    result = workspace.app.commands.persistence.publish_prepared_alignment_save_result(
        PreparedAlignmentSave((), use_docdb=False),
        AlignmentSaveJobCompleted(saved_outputs={}),
    )

    assert isinstance(result, EditedAlignmentOutputsSaved)
    assert not checkpoint_path.exists()
