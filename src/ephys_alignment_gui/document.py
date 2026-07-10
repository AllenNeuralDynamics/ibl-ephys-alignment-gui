"""Qt-free document model for the active alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AlignmentDocument:
    """In-memory state for the alignment workspace.

    This object owns workflow-relevant state, not heavy arrays or Qt items.
    It is intentionally small at first; fields should move here only when they
    replace scattered state checks in the GUI or loader.
    """

    mouse_root: Path | None = None
    mouse_id: str | None = None
    selected_recording: str | None = None
    selected_probe: str | None = None
    selected_shank: int = 0
    output_root: Path | None = None
    output_directory: Path | None = None
    channel_info_loaded: bool = False
    data_loaded: bool = False
    dirty: bool = False

    @property
    def probe_selected(self) -> bool:
        """Whether a recording/probe pair is selected."""
        return self.selected_recording is not None and self.selected_probe is not None

    def set_mouse_root(self, mouse_root: Path, mouse_id: str | None = None) -> None:
        """Record the active mouse root and clear probe/data state."""
        self.mouse_root = Path(mouse_root)
        self.mouse_id = mouse_id
        self.clear_probe()

    def clear_probe(self) -> None:
        """Clear selected probe and dependent state."""
        self.selected_recording = None
        self.selected_probe = None
        self.selected_shank = 0
        self.channel_info_loaded = False
        self.data_loaded = False
        self.dirty = False
        self.output_directory = None

    def select_probe(self, recording_id: str, probe_name: str) -> None:
        """Record the active probe and reset probe-derived state."""
        self.selected_recording = recording_id
        self.selected_probe = probe_name
        self.selected_shank = 0
        self.channel_info_loaded = False
        self.data_loaded = False
        self.dirty = False
        self.output_directory = None

    def set_channel_info_loaded(self, loaded: bool = True) -> None:
        """Record whether channel metadata is ready for the selected probe."""
        self.channel_info_loaded = loaded
        if not loaded:
            self.data_loaded = False

    def set_selected_shank(self, shank_idx: int) -> None:
        """Record the active shank index."""
        self.selected_shank = shank_idx

    def set_output_root(self, output_root: Path) -> None:
        """Record the root under which per-probe outputs are written."""
        self.output_root = Path(output_root)

    def set_output_directory(self, output_directory: Path | None) -> None:
        """Record the derived per-probe output directory."""
        self.output_directory = Path(output_directory) if output_directory else None

    def mark_data_loaded(self, loaded: bool = True) -> None:
        """Record whether heavy data has been loaded for the selected probe."""
        self.data_loaded = loaded
