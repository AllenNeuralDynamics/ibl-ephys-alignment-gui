"""Command-side application facade for the alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.alignment_edit_commands import AlignmentEditCommandHandler
from ephys_alignment_gui.alignment_persistence_commands import (
    AlignmentPersistenceCommandHandler,
)
from ephys_alignment_gui.alignment_persistence_results import NoPreviousAlignments
from ephys_alignment_gui.app_results import (
    ActiveStreamDetached,
    CachedEphysDataActivated,
    LoadDataBeginResult,
    LoadDataFreshCompleted,
    LoadDataFreshPrepared,
    LoadedShankPrepared,
    ProbeSelectionCacheResult,
    StreamCacheEvicted,
    VisitedAlignmentOutputsSaved,
)
from ephys_alignment_gui.controller import (
    AlignmentChoicesUpdated,
    AlignmentEditApplied,
    AlignmentEditNoop,
    Failed,
    LoadDataPrepared,
    PendingReferenceLinesUpdated,
    PreviousAlignmentSelected,
    ShankSelected,
)
from ephys_alignment_gui.display_commands import DisplayCommandHandler
from ephys_alignment_gui.ephys_stream_runtime import StreamKey
from ephys_alignment_gui.load_data_commands import LoadDataCommandHandler
from ephys_alignment_gui.loaded_shank_commands import LoadedShankCommandHandler
from ephys_alignment_gui.metadata_results import (
    MouseRootLoaded,
    ProbeSelected,
    RecordingSelected,
)
from ephys_alignment_gui.metadata_selection_commands import (
    MetadataSelectionCommandHandler,
)
from ephys_alignment_gui.path_commands import PathCommandHandler
from ephys_alignment_gui.path_results import OutputDirectoryDerived, OutputRootSet
from ephys_alignment_gui.reference_line_capture import (
    REFERENCE_LINES_NOT_PROVIDED,
    ReferenceLineCapture,
)
from ephys_alignment_gui.shank_selection_commands import ShankSelectionCommandHandler
from ephys_alignment_gui.workflow import Blocked, Ok, PolicyResult


@dataclass
class AlignmentCommands:
    """Command-side app port.

    This object is the stable UI-facing command surface. Non-trivial commands
    delegate to focused command handlers, which coordinate runtime services,
    controller document mutations, and semantic events.
    """

    shank_selection_commands: ShankSelectionCommandHandler
    load_data_commands: LoadDataCommandHandler
    loaded_shank_commands: LoadedShankCommandHandler
    path_commands: PathCommandHandler
    metadata_commands: MetadataSelectionCommandHandler
    persistence_commands: AlignmentPersistenceCommandHandler
    edit_commands: AlignmentEditCommandHandler
    display_commands: DisplayCommandHandler

    def select_shank(
        self,
        shank_idx: int,
        *,
        outgoing_reference_lines: ReferenceLineCapture = REFERENCE_LINES_NOT_PROVIDED,
        source: str = "command",
        preserve_plot_selection: bool | None = None,
    ) -> ShankSelected | Failed:
        """Select a shank as a complete app-level transaction."""
        return self.shank_selection_commands.select_shank(
            shank_idx,
            outgoing_reference_lines=outgoing_reference_lines,
            source=source,
            preserve_plot_selection=preserve_plot_selection,
        )

    def capture_active_reference_lines(
        self,
        reference_lines: tuple[Any, Any] | None,
    ) -> PendingReferenceLinesUpdated | Ok | Failed:
        """Capture active reference-line coordinates as document state."""
        return self.shank_selection_commands.capture_active_reference_lines(
            reference_lines
        )

    def set_mouse_root(self, mouse_root: Path) -> MouseRootLoaded | Failed:
        """Load a mouse root and update document metadata."""
        return self.metadata_commands.set_mouse_root(mouse_root)

    def clear_histology_context(self) -> Ok:
        """Clear loaded histology runtime data after a mouse-root change."""
        return self.metadata_commands.clear_histology_context()

    def set_output_root(self, output_root: Path) -> OutputRootSet | Failed:
        """Set the output root and derive the active probe output directory."""
        return self.path_commands.set_output_root(output_root)

    def derive_output_directory(self) -> OutputDirectoryDerived | Failed:
        """Derive the active per-probe output directory from document state."""
        return self.path_commands.derive_output_directory()

    def load_previous_alignments(
        self,
        *,
        folder: Path | None,
        use_docdb: bool,
        shank_idx: int | None = None,
    ) -> AlignmentChoicesUpdated | NoPreviousAlignments | Failed:
        """Load and store previous alignments for a document-selected shank."""
        return self.persistence_commands.load_previous_alignments(
            folder=folder,
            use_docdb=use_docdb,
            shank_idx=shank_idx,
        )

    def select_recording_metadata(
        self,
        recording_id: str,
    ) -> RecordingSelected | Failed:
        """Select a recording and return its probe choices."""
        return self.metadata_commands.select_recording_metadata(recording_id)

    def select_probe_metadata(
        self,
        recording_id: str,
        probe_name: str,
    ) -> ProbeSelected | Failed:
        """Select a probe and load lightweight channel metadata."""
        return self.metadata_commands.select_probe_metadata(
            recording_id,
            probe_name,
        )

    def select_previous_alignment(
        self,
        idx: int,
        *,
        shank_idx: int | None = None,
    ) -> PreviousAlignmentSelected | Failed:
        """Select a previous/original alignment on a document-selected shank."""
        return self.persistence_commands.select_previous_alignment(
            idx,
            shank_idx=shank_idx,
        )

    def can_load_previous_alignments(self) -> Ok | Failed:
        """Return whether previous alignments can be loaded."""
        return self.persistence_commands.can_load_previous_alignments()

    def begin_load_data(
        self,
        *,
        recording_id: str,
        probe_name: str,
        target_shank: int,
        outgoing_reference_lines: ReferenceLineCapture = REFERENCE_LINES_NOT_PROVIDED,
    ) -> LoadDataBeginResult | Failed:
        """Prepare or activate the selected stream/shank load transaction."""
        return self.load_data_commands.begin_load_data(
            recording_id=recording_id,
            probe_name=probe_name,
            target_shank=target_shank,
            outgoing_reference_lines=outgoing_reference_lines,
        )

    def activate_cached_probe_selection(
        self,
        *,
        recording_id: str,
        probe_name: str,
        target_shank: int,
    ) -> ProbeSelectionCacheResult | Failed:
        """Activate a cached probe selection or report that fresh loading is needed."""
        return self.load_data_commands.activate_cached_probe_selection(
            recording_id=recording_id,
            probe_name=probe_name,
            target_shank=target_shank,
        )

    def complete_fresh_load_data(
        self,
        prepared: LoadDataFreshPrepared,
    ) -> LoadDataFreshCompleted | Failed:
        """Run fresh ephys and histology load steps for a prepared transaction."""
        return self.load_data_commands.complete_fresh_load_data(prepared)

    def prepare_fresh_ephys_load(
        self,
        stream_key: StreamKey | None,
    ) -> LoadDataPrepared:
        """Mark data unloaded and discard stale active/cache state."""
        return self.load_data_commands.prepare_fresh_ephys_load(stream_key)

    def detach_active_stream(self) -> ActiveStreamDetached:
        """Detach the active stream while preserving cached runtimes."""
        return self.load_data_commands.detach_active_stream()

    def evict_stream_cache(self) -> StreamCacheEvicted:
        """Evict cached stream runtimes for a recording/session transition."""
        return self.load_data_commands.evict_stream_cache()

    def activate_cached_ephys_data(
        self,
        *,
        recording_id: str,
        probe_name: str,
        stream_key: StreamKey,
        shank_idx: int,
    ) -> CachedEphysDataActivated | Failed:
        """Activate cached ephys runtime data for one explicit shank."""
        return self.load_data_commands.activate_cached_ephys_data(
            recording_id=recording_id,
            probe_name=probe_name,
            stream_key=stream_key,
            shank_idx=shank_idx,
        )

    def prepare_loaded_shank(
        self,
        shank_idx: int,
        *,
        select_default_alignment_if_empty: bool = True,
    ) -> LoadedShankPrepared | Failed:
        """Prepare Qt-free runtime state for a loaded active shank."""
        return self.loaded_shank_commands.prepare_loaded_shank(
            shank_idx,
            select_default_alignment_if_empty=select_default_alignment_if_empty,
        )

    def can_load_data(self) -> PolicyResult:
        """Return whether the selected stream can be loaded."""
        return self.load_data_commands.can_load_data()

    def can_save_alignment_output(self) -> Ok | Blocked:
        """Return whether visited alignment outputs can be saved."""
        return self.persistence_commands.can_save_alignment_output()

    def save_visited_alignment_outputs(
        self,
        *,
        use_docdb: bool,
    ) -> VisitedAlignmentOutputsSaved | Blocked | Failed:
        """Persist outputs for every visited alignment in the active stream."""
        return self.persistence_commands.save_visited_alignment_outputs(
            use_docdb=use_docdb
        )

    def set_unit_filter(self, unit_filter: str) -> Ok:
        """Select the unit subset used when preparing ephys plot data."""
        return self.edit_commands.set_unit_filter(unit_filter)

    def toggle_reference_lines_visible(self) -> bool:
        """Toggle whether reference lines should be rendered."""
        return self.display_commands.toggle_reference_lines_visible()

    def toggle_histology_boundaries_visible(self) -> bool:
        """Toggle whether histology boundary overlays should be rendered."""
        return self.display_commands.toggle_histology_boundaries_visible()

    def toggle_region_annotation_source(self) -> str:
        """Toggle the displayed region annotation label source."""
        return self.display_commands.toggle_region_annotation_source()

    def set_linear_fit_enabled(self, enabled: bool) -> bool:
        """Set whether fit commands should use linear fitting."""
        return self.display_commands.set_linear_fit_enabled(enabled)

    def offset_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        probe_tip_um: float,
        lin_fit: bool,
        track_shift_m: float = 0.0,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply an offset edit on a document-selected shank."""
        return self.edit_commands.offset_alignment_from_tip(
            tip_position_um=tip_position_um,
            probe_tip_um=probe_tip_um,
            lin_fit=lin_fit,
            track_shift_m=track_shift_m,
            shank_idx=shank_idx,
        )

    def offset_active_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        track_shift_m: float = 0.0,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a tip-offset edit using app-owned display settings."""
        return self.edit_commands.offset_active_alignment_from_tip(
            tip_position_um=tip_position_um,
            track_shift_m=track_shift_m,
        )

    def nudge_active_alignment_from_tip(
        self,
        *,
        tip_position_um: float,
        track_shift_m: float,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a bounded tip-offset nudge for the active alignment."""
        return self.edit_commands.nudge_active_alignment_from_tip(
            tip_position_um=tip_position_um,
            track_shift_m=track_shift_m,
        )

    def fit_alignment_to_reference_lines(
        self,
        shank_runtime: Any,
        *,
        line_features_um: Any,
        line_tracks_um: Any,
        lin_fit: bool,
        extend_feature: int,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a reference-line fit for a document-selected shank runtime."""
        return self.edit_commands.fit_alignment_to_reference_lines(
            shank_runtime,
            line_features_um=line_features_um,
            line_tracks_um=line_tracks_um,
            lin_fit=lin_fit,
            extend_feature=extend_feature,
        )

    def fit_active_alignment_from_pending_reference_lines(
        self,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Apply a fit edit from document-owned pending reference lines."""
        return self.edit_commands.fit_active_alignment_from_pending_reference_lines()

    def go_next_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor forward."""
        return self.edit_commands.go_next_alignment(shank_idx)

    def go_previous_alignment(
        self,
        shank_idx: int | None = None,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Move the active alignment edit cursor backward."""
        return self.edit_commands.go_previous_alignment(shank_idx)

    def reset_alignment_to_initial(
        self,
        shank_runtime: Any,
        *,
        lin_fit: bool,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Reset active alignment state to the loaded runtime's initial geometry."""
        return self.edit_commands.reset_alignment_to_initial(
            shank_runtime,
            lin_fit=lin_fit,
        )

    def reset_active_alignment_to_initial(
        self,
    ) -> AlignmentEditApplied | AlignmentEditNoop | Failed:
        """Reset active alignment using the active runtime and display settings."""
        return self.edit_commands.reset_active_alignment_to_initial()
