"""App-level mouse, recording, and probe metadata selection commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.controller import AlignmentController
from ephys_alignment_gui.ephys_data_service import EphysDataService
from ephys_alignment_gui.histology_data_service import HistologyDataContext
from ephys_alignment_gui.metadata_results import (
    MouseRootLoaded,
    ProbeSelected,
    RecordingSelected,
)
from ephys_alignment_gui.path_commands import PathCommandHandler
from ephys_alignment_gui.workflow import Failed, Ok


@dataclass
class MetadataSelectionCommandHandler:
    """Coordinate metadata context IO and document selection state."""

    controller: AlignmentController
    data_context: AlignmentDataContext
    ephys_data_service: EphysDataService
    path_commands: PathCommandHandler
    histology_context: HistologyDataContext | None = None

    def clear_histology_context(self) -> Ok:
        """Clear loaded histology runtime data after a mouse-root change."""
        if self.histology_context is not None:
            self.histology_context.clear()
        return Ok()

    def set_mouse_root(self, mouse_root: Path) -> MouseRootLoaded | Failed:
        """Load a mouse root and update document metadata."""
        if not mouse_root or str(mouse_root).strip() == "":
            return Failed("Empty mouse-root path provided")
        mouse_root = Path(mouse_root)
        if not mouse_root.is_dir():
            return Failed(f"Mouse-root is not a directory: {mouse_root}")

        old_root = (
            self.data_context.mouse_root.root
            if self.data_context.mouse_root is not None
            else None
        )
        try:
            loaded_root = self.data_context.set_mouse_root(mouse_root)
        except Exception as exc:
            return Failed(f"Failed to load mouse root {mouse_root}: {exc}")

        root_changed = old_root is not None and old_root != loaded_root.root
        self.controller.record_mouse_root_loaded(
            loaded_root,
            root_changed=root_changed,
        )
        return MouseRootLoaded(loaded_root, root_changed=root_changed)

    def select_recording_metadata(
        self,
        recording_id: str,
    ) -> RecordingSelected | Failed:
        """Select a recording and return its available probes."""
        if self.data_context.mouse_root is None:
            return Failed("No mouse root loaded. Please select a mouse root first.")
        if not recording_id:
            return Failed("No recording selected.")

        self.controller.clear_probe_selection()
        try:
            probes = self.data_context.list_probes(recording_id)
        except Exception as exc:
            return Failed(f"Failed to list probes for {recording_id}: {exc}")
        return RecordingSelected(recording_id, probes=list(probes))

    def select_probe_metadata(
        self,
        recording_id: str,
        probe_name: str,
        *,
        ephys_stream: Any | None = None,
    ) -> ProbeSelected | Failed:
        """Select a probe and load lightweight channel metadata."""
        if self.data_context.mouse_root is None:
            return Failed("No mouse root loaded. Please select a mouse root first.")
        if not recording_id:
            return Failed("No recording selected.")
        if not probe_name:
            return Failed("No probe selected.")

        self.controller.record_probe_selected(recording_id, probe_name)
        try:
            self.data_context.select_probe(recording_id, probe_name)
            probe = self.data_context.probe_info
            assert probe is not None
            if ephys_stream is None:
                channel_table = self.ephys_data_service.load_channel_table(probe)
            else:
                self.data_context.validate_cached_stream(ephys_stream)
                channel_table = ephys_stream.channel_table
            self.data_context.attach_channel_table(channel_table)
            self.controller.record_probe_channel_info(
                probe,
                n_shanks=self.data_context.n_shanks,
                shank_idx=0,
            )
            shanks = self.data_context.shank_labels()
            output_result = self.path_commands.derive_output_directory()
        except Exception as exc:
            self.controller.record_channel_info_loaded(False)
            return Failed(f"Failed to select probe {probe_name}: {exc}")

        if isinstance(output_result, Failed):
            return output_result

        return ProbeSelected(
            recording_id=recording_id,
            probe_name=probe_name,
            shanks=list(shanks),
            n_shanks=self.data_context.n_shanks,
            output_directory=output_result.output_directory,
        )
