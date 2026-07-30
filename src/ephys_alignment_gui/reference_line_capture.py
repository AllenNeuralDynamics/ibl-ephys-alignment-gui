"""Shared command helpers for document-owned reference-line state."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.controller import (
    AlignmentController,
    Failed,
    PendingReferenceLinesUpdated,
)
from ephys_alignment_gui.workflow import Ok


class ReferenceLinesNotProvided:
    """Sentinel for commands whose caller did not provide reference lines."""


REFERENCE_LINES_NOT_PROVIDED = ReferenceLinesNotProvided()
ReferenceLineCapture = tuple[Any, Any] | None | ReferenceLinesNotProvided


def capture_outgoing_reference_lines(
    controller: AlignmentController,
    outgoing_reference_lines: ReferenceLineCapture,
) -> PendingReferenceLinesUpdated | None | Failed:
    """Store or clear pending reference lines for the active outgoing shank."""
    outgoing_shank_idx = controller.document.selected_shank
    if outgoing_reference_lines is None:
        return controller.clear_pending_reference_lines(outgoing_shank_idx)

    if outgoing_reference_lines is REFERENCE_LINES_NOT_PROVIDED:
        return None

    feature_positions_um, track_positions_um = outgoing_reference_lines
    return controller.set_pending_reference_lines(
        feature_positions_um=feature_positions_um,
        track_positions_um=track_positions_um,
        shank_idx=outgoing_shank_idx,
    )


def capture_active_reference_lines(
    controller: AlignmentController,
    reference_lines: tuple[Any, Any] | None,
) -> PendingReferenceLinesUpdated | Ok | Failed:
    """Capture active reference-line coordinates as document state."""
    if not controller.document.data_loaded:
        return Ok()
    result = capture_outgoing_reference_lines(controller, reference_lines)
    if result is None:
        return Ok()
    return result


def capture_active_reference_lines_if_provided(
    controller: AlignmentController,
    reference_lines: ReferenceLineCapture,
) -> PendingReferenceLinesUpdated | Ok | Failed:
    """Capture active reference lines only when a caller supplied them."""
    if reference_lines is REFERENCE_LINES_NOT_PROVIDED:
        return Ok()
    return capture_active_reference_lines(controller, reference_lines)
