"""Tests for Qt-free alignment controller commands."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ephys_alignment_gui.application.results import (
    AlignmentChoicesUpdated,
    AlignmentEditApplied,
    PendingReferenceLinesUpdated,
    PreviousAlignmentSelected,
    ShankRuntimeInitialized,
    ShankSelected,
)
from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_key_context import AlignmentKeyContext
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.core.workflow import Blocked, Failed, Ok


@dataclass(frozen=True)
class FakeProbeInfo:
    recording_id: str
    probe_name: str
    probe_id: str
    num_shanks: int
    ephys_collection: str = "streamA"


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
    key_context: AlignmentKeyContext | None = None,
    alignment_runtime_service: FakeAlignmentRuntimeService | None = None,
) -> tuple[AlignmentController, AlignmentKeyContext]:
    doc = doc or AlignmentDocument()
    key_context = key_context or AlignmentKeyContext()
    controller = AlignmentController(
        doc,
        key_context,
        alignment_runtime_service=alignment_runtime_service,
    )
    return controller, key_context


def _set_loaded_probe_state(
    doc: AlignmentDocument,
    controller: AlignmentController,
    *,
    shank_idx: int = 0,
    n_shanks: int = 2,
) -> None:
    probe = FakeProbeInfo(
        recording_id="rec1",
        probe_name="probeA",
        probe_id="rec1:probeA",
        num_shanks=n_shanks,
    )
    doc.select_probe(probe.recording_id, probe.probe_name)
    controller.record_probe_channel_info(
        probe,
        n_shanks=n_shanks,
        shank_idx=shank_idx,
    )


def test_load_data_preparation_and_finish_updates_document():
    doc = AlignmentDocument(data_loaded=True, selected_shank=1)
    controller, _ = make_controller(doc)
    _set_loaded_probe_state(doc, controller, shank_idx=1, n_shanks=3)
    doc.mark_data_loaded(True)

    prepared = controller.prepare_load_data()
    controller.finish_load_data(shank_idx=2)

    assert prepared.preserve_plot_selection
    assert doc.data_loaded
    assert doc.selected_shank == 2
    assert doc.selected_alignment_key == AlignmentKey("rec1", "streamA", 2)


def test_set_selected_shank_updates_document_alignment_key():
    doc = AlignmentDocument()
    controller, _ = make_controller(doc)
    _set_loaded_probe_state(doc, controller)

    controller.set_selected_shank(1)

    assert doc.selected_alignment_key == AlignmentKey("rec1", "streamA", 1)
    assert doc.selected_shank == 1


def test_select_shank_returns_transition_metadata():
    doc = AlignmentDocument()
    controller, _ = make_controller(doc)
    _set_loaded_probe_state(doc, controller)
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


def test_select_shank_rejects_out_of_range_when_channel_info_is_loaded():
    doc = AlignmentDocument()
    controller, _ = make_controller(doc)
    _set_loaded_probe_state(doc, controller)
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
    controller, _ = make_controller(doc)

    assert isinstance(controller.can_load_data(), Ok)


def test_can_load_previous_alignments_requires_channel_info():
    controller, _ = make_controller(AlignmentDocument())

    assert isinstance(controller.can_load_previous_alignments(), Failed)


def test_set_previous_alignments_filters_legacy_auto_and_returns_choices() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    controller, _ = make_controller(doc)

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
    controller, _ = make_controller(doc)

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
    controller, _ = make_controller(doc)
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
    assert not state.has_unsaved_alignment


def test_select_previous_alignment_can_mark_user_selection_dirty() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    state = doc.active_alignment_state
    assert state is not None
    state.active_alignment = ActiveAlignment(
        np.array([9.0, 10.0]),
        np.array([11.0, 12.0]),
    )
    state.mark_saved()
    doc.set_active_alignments({"saved": [[1.0, 2.0], [3.0, 4.0]]})
    controller, _ = make_controller(doc)

    result = controller.select_previous_alignment(
        0,
        shank_idx=0,
        mark_changed=True,
    )

    assert isinstance(result, PreviousAlignmentSelected)
    assert state.has_unsaved_alignment
    assert doc.dirty


def test_initialize_shank_runtime_seeds_empty_document_state() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    runtime_service = FakeAlignmentRuntimeService()
    controller, _ = make_controller(
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
    controller, _ = make_controller(
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
    controller, _ = make_controller(
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
    controller, _ = make_controller(doc)
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
    controller, _ = make_controller(doc)

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
    controller, _ = make_controller(doc)

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
    controller, _ = make_controller(doc)
    state.mark_saved()
    controller.offset_alignment_from_tip(
        tip_position_um=100.0,
        probe_tip_um=0.0,
        lin_fit=False,
    )
    assert state.has_unsaved_alignment
    assert doc.dirty

    previous_result = controller.go_previous_alignment()

    assert isinstance(previous_result, AlignmentEditApplied)
    np.testing.assert_array_equal(previous_result.alignment.track, [10.0, 14.0])
    assert previous_result.lin_fit is True
    assert state.active_alignment is not None
    np.testing.assert_array_equal(state.active_alignment.track, [10.0, 14.0])
    assert not state.has_unsaved_alignment
    assert not doc.dirty

    next_result = controller.go_next_alignment()

    assert isinstance(next_result, AlignmentEditApplied)
    np.testing.assert_allclose(next_result.alignment.track, [10.0001, 14.0001])
    assert next_result.lin_fit is False
    assert state.active_alignment is not None
    np.testing.assert_allclose(state.active_alignment.track, [10.0001, 14.0001])
    assert state.has_unsaved_alignment
    assert doc.dirty


def test_reset_alignment_to_initial_updates_active_document_state() -> None:
    doc = AlignmentDocument()
    doc.select_alignment_key(AlignmentKey("rec1", "streamA", 0))
    state = doc.active_alignment_state
    assert state is not None
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
    )
    controller, _ = make_controller(doc)
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
    controller, _ = make_controller(doc)

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
    controller, _ = make_controller(doc)

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
    controller, _ = make_controller(AlignmentDocument())

    assert isinstance(controller.can_save_alignment_output(), Blocked)
