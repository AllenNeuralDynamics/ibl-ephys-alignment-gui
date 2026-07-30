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
    assert workspace.path_commands.data_context is workspace.data_context
    assert workspace.metadata_commands.data_context is workspace.data_context
    assert (
        workspace.metadata_commands.ephys_data_service
        is workspace.ephys_data_service
    )
    assert workspace.metadata_commands.path_commands is workspace.path_commands
    assert workspace.load_data_commands.metadata_commands is workspace.metadata_commands
    assert workspace.loaded_shank_commands.data_context is workspace.data_context
    assert (
        workspace.persistence_commands.alignment_repository
        is workspace.alignment_repository
    )
    assert (
        workspace.persistence_commands.output_builder
        is workspace.alignment_output_service
    )
    assert workspace.edit_commands.runtime is workspace.runtime
    assert workspace.app.commands.metadata_commands is workspace.metadata_commands
    assert workspace.app.commands.load_data_commands is workspace.load_data_commands
    assert (
        workspace.app.commands.loaded_shank_commands
        is workspace.loaded_shank_commands
    )
    assert workspace.app.commands.persistence_commands is workspace.persistence_commands
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
