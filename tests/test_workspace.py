"""Tests for the Qt-free alignment workspace."""

from __future__ import annotations

from ephys_alignment_gui.workspace import AlignmentWorkspace


def test_workspace_wires_shared_services() -> None:
    workspace = AlignmentWorkspace()

    assert workspace.controller.document is workspace.document
    assert (
        workspace.controller.alignment_key_context
        is workspace.alignment_key_context
    )
    assert workspace.controller.workflow_policy is workspace.workflow_policy
    assert workspace.app.commands._data_context is workspace.data_context
    assert workspace.app.commands._ephys_data_service is workspace.ephys_data_service
    assert (
        workspace.app.commands._alignment_repository
        is workspace.alignment_repository
    )
    assert (
        workspace.app.commands._alignment_output_service
        is workspace.alignment_output_service
    )
    assert workspace.ephys_stream_loader.data_context is workspace.data_context
    assert (
        workspace.ephys_stream_loader.ephys_data_service
        is workspace.ephys_data_service
    )
    assert workspace.histology_data_service is not None
    assert workspace.alignment_output_service.data_context is workspace.data_context
    assert (
        workspace.alignment_output_service.histology_context
        is workspace.histology_context
    )
    assert workspace.alignment_edit_service is not None
    assert workspace.alignment_derived_data_service is not None
    assert workspace.probe_track_service is not None
    assert workspace.region_lookup_service is not None
    assert workspace.slice_service is not None
    assert workspace.slice_display_policy is not None
    assert workspace.runtime is not None
    assert workspace.events is not None
