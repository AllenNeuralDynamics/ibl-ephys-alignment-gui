"""Tests for the Qt-free alignment workspace."""

from __future__ import annotations

from ephys_alignment_gui.workspace import AlignmentWorkspace


def test_workspace_wires_shared_services() -> None:
    workspace = AlignmentWorkspace()

    assert workspace.controller.document is workspace.document
    assert workspace.controller.data_context is workspace.data_context
    assert workspace.controller.ephys_data_service is workspace.ephys_data_service
    assert workspace.controller.workflow_policy is workspace.workflow_policy
    assert workspace.controller.alignment_repository is workspace.alignment_repository
    assert workspace.controller.output_builder is workspace.loader
    assert workspace.probe_data_workflow.data_context is workspace.data_context
    assert (
        workspace.probe_data_workflow.ephys_data_service
        is workspace.ephys_data_service
    )
    assert workspace.histology_data_service is not None
    assert workspace.alignment_edit_service is not None
    assert workspace.alignment_derived_data_service is not None
    assert workspace.loader.data_context is workspace.data_context
    assert workspace.loader.slice_service is workspace.slice_service
    assert workspace.slice_display_policy is not None
    assert workspace.runtime is not None
