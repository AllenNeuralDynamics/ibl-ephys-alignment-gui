"""Qt-free document model for the active alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from numpy.typing import NDArray

from ephys_alignment_gui.alignment_state import AlignmentState, PendingReferenceLines


@dataclass(frozen=True)
class AlignmentKey:
    """Stable document key for one editable alignment state."""

    recording_id: str
    ephys_collection: str
    shank_idx: int

    def __post_init__(self) -> None:
        if self.shank_idx < 0:
            raise ValueError("shank_idx must be non-negative")


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
    selected_alignment_key: AlignmentKey | None = None
    alignment_states: dict[AlignmentKey, AlignmentState] = field(default_factory=dict)
    output_root: Path | None = None
    output_directory: Path | None = None
    channel_info_loaded: bool = False
    data_loaded: bool = False
    dirty: bool = False

    @property
    def probe_selected(self) -> bool:
        """Whether a recording/probe pair is selected."""
        return self.selected_recording is not None and self.selected_probe is not None

    def set_mouse_root(
        self,
        mouse_root: Path,
        mouse_id: str | None = None,
        *,
        clear_alignment_states: bool = False,
    ) -> None:
        """Record the active mouse root and clear probe/data state."""
        self.mouse_root = Path(mouse_root)
        self.mouse_id = mouse_id
        self.clear_probe()
        if clear_alignment_states:
            self.alignment_states.clear()

    def clear_probe(self) -> None:
        """Clear selected probe and dependent state."""
        self.selected_recording = None
        self.selected_probe = None
        self.selected_shank = 0
        self.selected_alignment_key = None
        self.channel_info_loaded = False
        self.data_loaded = False
        self.dirty = False
        self.output_directory = None

    def select_probe(self, recording_id: str, probe_name: str) -> None:
        """Record the active probe and reset probe-derived state."""
        self.selected_recording = recording_id
        self.selected_probe = probe_name
        self.selected_shank = 0
        self.selected_alignment_key = None
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
        if self.selected_alignment_key is not None:
            self.select_alignment_key(
                AlignmentKey(
                    recording_id=self.selected_alignment_key.recording_id,
                    ephys_collection=self.selected_alignment_key.ephys_collection,
                    shank_idx=shank_idx,
                )
            )

    def select_alignment_key(self, key: AlignmentKey) -> AlignmentState:
        """Select and return the editable state for an alignment key."""
        self.selected_alignment_key = key
        self.selected_recording = key.recording_id
        self.selected_shank = key.shank_idx
        return self.alignment_state_for(key)

    def alignment_state_for(self, key: AlignmentKey) -> AlignmentState:
        """Return the editable state for a key, creating it if needed."""
        if key not in self.alignment_states:
            self.alignment_states[key] = AlignmentState()
        return self.alignment_states[key]

    def alignment_states_for_current_probe(
        self,
    ) -> dict[AlignmentKey, AlignmentState]:
        """Return known alignment states for the active recording/stream."""
        if self.selected_alignment_key is None:
            return {}
        active = self.selected_alignment_key
        return {
            key: state
            for key, state in self.alignment_states.items()
            if key.recording_id == active.recording_id
            and key.ephys_collection == active.ephys_collection
        }

    @property
    def active_alignment_state(self) -> AlignmentState | None:
        """Editable state for the selected alignment key, if one is active."""
        if self.selected_alignment_key is None:
            return None
        return self.alignment_state_for(self.selected_alignment_key)

    @property
    def active_alignments(self) -> dict[str, list[list[float]]] | None:
        """Saved alignment history for the active alignment state."""
        state = self.active_alignment_state
        return None if state is None else state.alignments

    @property
    def active_prev_align(self) -> list[str] | None:
        """Dropdown-ordered alignment keys for the active alignment state."""
        state = self.active_alignment_state
        return None if state is None else state.prev_align

    def set_active_alignments(self, alignments: dict[str, list[list[float]]]) -> None:
        """Replace persisted history on the active alignment state."""
        state = self._require_active_alignment_state()
        state.set_alignments(alignments)

    def active_add_alignment(self, feature: NDArray, track: NDArray) -> str:
        """Record a saved alignment on the active alignment state."""
        state = self._require_active_alignment_state()
        return state.add_alignment(feature, track)

    def active_get_alignment_idx(
        self, idx: int
    ) -> tuple[NDArray | None, NDArray | None]:
        """Return an alignment from the active state's dropdown index."""
        state = self._require_active_alignment_state()
        return state.get_alignment_idx(idx)

    def active_select_alignment_idx(
        self, idx: int
    ) -> tuple[NDArray | None, NDArray | None]:
        """Select an alignment choice on the active state and return it."""
        state = self._require_active_alignment_state()
        return state.select_alignment_idx(idx)

    def active_set_pending_reference_lines(
        self,
        feature_positions_um,
        track_positions_um,
    ) -> PendingReferenceLines | None:
        """Store active-state reference-line coordinates."""
        state = self._require_active_alignment_state()
        lines = PendingReferenceLines.from_values(
            feature_positions_um,
            track_positions_um,
        )
        state.set_pending_reference_lines(lines)
        return lines

    def active_clear_pending_reference_lines(self) -> None:
        """Clear active-state reference-line coordinates."""
        state = self._require_active_alignment_state()
        state.clear_pending_reference_lines()

    @property
    def active_pending_reference_lines(self) -> PendingReferenceLines | None:
        """Pending reference-line coordinates for the active alignment state."""
        state = self.active_alignment_state
        return None if state is None else state.pending_reference_lines

    def _require_active_alignment_state(self) -> AlignmentState:
        state = self.active_alignment_state
        if state is None:
            raise RuntimeError("No active alignment key selected")
        return state

    def set_output_root(self, output_root: Path) -> None:
        """Record the root under which per-probe outputs are written."""
        self.output_root = Path(output_root)

    def set_output_directory(self, output_directory: Path | None) -> None:
        """Record the derived per-probe output directory."""
        self.output_directory = Path(output_directory) if output_directory else None

    def mark_data_loaded(self, loaded: bool = True) -> None:
        """Record whether heavy data has been loaded for the selected probe."""
        self.data_loaded = loaded
