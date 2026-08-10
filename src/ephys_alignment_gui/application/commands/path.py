"""App-level input/output path command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ephys_alignment_gui.application.results.path import (
    OutputDirectoryDerived,
    OutputRootSet,
)
from ephys_alignment_gui.application.workflow import Failed
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext


@dataclass
class PathCommandHandler:
    """Coordinate path validation, directory creation, and document path state."""

    controller: AlignmentController
    data_context: AlignmentDataContext

    def set_output_root(self, output_root: Path) -> OutputRootSet | Failed:
        """Set the output root and derive the active probe output directory."""
        if not output_root or str(output_root).strip() == "":
            return Failed("Empty save-root path provided")
        output_root = Path(output_root)
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return Failed(f"Failed to create save-root directory {output_root}: {exc}")
        if not output_root.is_dir():
            return Failed(f"Save-root is not a directory: {output_root}")

        self.controller.record_output_root(output_root)
        output_result = self.derive_output_directory()
        if isinstance(output_result, Failed):
            return output_result
        return OutputRootSet(output_root, output_result.output_directory)

    def derive_output_directory(self) -> OutputDirectoryDerived | Failed:
        """Derive the per-probe output directory from document + probe metadata."""
        probe = self.data_context.probe_info
        document = self.controller.document
        output_root = document.output_root
        if (
            output_root is None
            or probe is None
            or probe.recording_id != document.selected_recording
            or probe.probe_name != document.selected_probe
        ):
            self.controller.record_output_directory(None)
            return OutputDirectoryDerived(None)

        output_directory = output_root / probe.recording_id / probe.probe_name
        try:
            output_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return Failed(
                f"Failed to create probe output directory {output_directory}: {exc}"
            )
        self.controller.record_output_directory(output_directory)
        return OutputDirectoryDerived(output_directory)
