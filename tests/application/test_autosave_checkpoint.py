"""Tests for application-level autosave checkpoint commands."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ephys_alignment_gui.application.commands.autosave import (
    AUTOSAVE_DIRECTORY_NAME,
    AUTOSAVE_DOCUMENT_FILENAME,
    AutosaveCheckpointCommandHandler,
)
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointCleared,
    AutosaveCheckpointInspected,
    AutosaveCheckpointRead,
    AutosaveCheckpointRecovered,
    AutosaveCheckpointRestored,
    AutosaveCheckpointWritten,
)
from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_events import AutosaveRecovered
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.core.event_bus import EventBus
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


def test_inspect_checkpoint_reports_recoverable_and_skipped_keys(tmp_path) -> None:
    checkpoint_path = _write_checkpoint_with_valid_and_invalid_keys(tmp_path)
    live_document = AlignmentDocument()
    handler = AutosaveCheckpointCommandHandler(
        AlignmentController(live_document),
        input_dataset_provider=lambda: _input_dataset(mouse_id="mouse"),
    )

    inspected = handler.inspect_checkpoint(checkpoint_path)

    assert isinstance(inspected, AutosaveCheckpointInspected)
    assert inspected.path == checkpoint_path
    assert inspected.mouse_id == "mouse"
    assert inspected.alignment_state_count == 2
    assert inspected.recoverable_alignment_count == 1
    assert len(inspected.skipped_keys) == 1
    assert inspected.skipped_keys[0].key == AlignmentKey("rec", "missing", 0)


def test_recover_checkpoint_validates_restores_and_normalizes_for_gui(
    tmp_path,
) -> None:
    checkpoint_path = _write_checkpoint_with_valid_and_invalid_keys(tmp_path)
    live_document = AlignmentDocument(
        mouse_id="mouse",
        output_package_directory=tmp_path / "current_package",
    )
    live_document.alignment_state_for(AlignmentKey("old", "old-stream", 0))
    events = EventBus()
    emitted: list[AutosaveRecovered] = []
    events.subscribe(AutosaveRecovered, emitted.append)
    valid_key = AlignmentKey("rec", "stream", 0)
    handler = AutosaveCheckpointCommandHandler(
        AlignmentController(live_document),
        input_dataset_provider=lambda: _input_dataset(mouse_id="mouse"),
        events=events,
    )

    recovered = handler.recover_checkpoint(checkpoint_path)

    assert isinstance(recovered, AutosaveCheckpointRecovered)
    assert recovered.selected_alignment_key == valid_key
    assert recovered.restored_alignment_count == 1
    assert len(recovered.skipped_keys) == 1
    assert recovered.backup_path == (
        checkpoint_path.parent / "alignment_document.pre_restore.json"
    )
    assert recovered.backup_path.exists()
    assert live_document.alignment_states.keys() == {valid_key}
    assert live_document.selected_alignment_key == valid_key
    assert live_document.selected_recording == "rec"
    assert live_document.selected_probe == "probe-name"
    assert live_document.selected_shank == 0
    assert live_document.output_package_directory == tmp_path / "package"
    assert live_document.output_directory == tmp_path / "package" / "rec" / "probe-name"
    assert not live_document.channel_info_loaded
    assert not live_document.data_loaded
    assert live_document.dirty
    assert len(emitted) == 1
    assert emitted[0].selected_key == valid_key
    assert emitted[0].restored_alignment_count == 1
    assert emitted[0].skipped_alignment_count == 1


def test_recover_checkpoint_requires_loaded_input_dataset_by_default(tmp_path) -> None:
    checkpoint_path = _write_checkpoint_with_valid_and_invalid_keys(tmp_path)
    handler = AutosaveCheckpointCommandHandler(AlignmentController(AlignmentDocument()))

    result = handler.recover_checkpoint(checkpoint_path)

    assert isinstance(result, Failed)
    assert "Load the matching mouse root" in result.message


def test_recover_checkpoint_rejects_mouse_mismatch(tmp_path) -> None:
    checkpoint_path = _write_checkpoint_with_valid_and_invalid_keys(tmp_path)
    handler = AutosaveCheckpointCommandHandler(
        AlignmentController(AlignmentDocument()),
        input_dataset_provider=lambda: _input_dataset(mouse_id="other"),
    )

    result = handler.recover_checkpoint(checkpoint_path)

    assert isinstance(result, Failed)
    assert "does not match loaded mouse" in result.message


class _InputDataset:
    def __init__(self, *, mouse_id: str) -> None:
        self.mouse_id = mouse_id
        self._probes = {
            ("rec", "stream"): SimpleNamespace(
                probe_name="probe-name",
                num_shanks=1,
            ),
        }

    def probe_for_stream_key(self, recording_id: str, ephys_collection: str):
        return self._probes[(recording_id, ephys_collection)]


def _input_dataset(*, mouse_id: str) -> _InputDataset:
    return _InputDataset(mouse_id=mouse_id)


def _write_checkpoint_with_valid_and_invalid_keys(tmp_path):
    package_dir = tmp_path / "package"
    source_document = AlignmentDocument(
        mouse_id="mouse",
        output_package_directory=package_dir,
        output_directory=package_dir / "rec" / "stale-probe",
        channel_info_loaded=True,
        data_loaded=True,
    )
    valid_key = AlignmentKey("rec", "stream", 0)
    invalid_key = AlignmentKey("rec", "missing", 0)
    source_document.selected_alignment_key = invalid_key
    source_document.selected_recording = "rec"
    source_document.selected_probe = "stale-probe"
    source_document.selected_shank = 0
    for key in (valid_key, invalid_key):
        state = source_document.alignment_state_for(key)
        state.active_alignment = ActiveAlignment(
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        )
        state.mark_alignment_changed()

    checkpoint_path = package_dir / "autosave" / "alignment_document.json"
    writer = AutosaveCheckpointCommandHandler(AlignmentController(source_document))
    written = writer.write_checkpoint(checkpoint_path)
    assert isinstance(written, AutosaveCheckpointWritten)
    return checkpoint_path
