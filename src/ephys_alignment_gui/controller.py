"""Qt-free controller commands for alignment workflow state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ephys_alignment_gui.alignment_repository import (
    AlignmentHistory,
    AlignmentRepository,
    SavedAlignmentOutputs,
)
from ephys_alignment_gui.document import AlignmentDocument
from ephys_alignment_gui.workflow import Failed, Ok, PolicyResult, WorkflowPolicy


@dataclass(frozen=True)
class MouseRootLoaded:
    """A mouse root was loaded and the document was updated."""

    mouse_root: Any
    root_changed: bool


@dataclass(frozen=True)
class RecordingSelected:
    """A recording was selected and its probe choices are available."""

    recording_id: str
    probes: list[str]


@dataclass(frozen=True)
class ProbeSelected:
    """A probe was selected and channel metadata is ready."""

    recording_id: str
    probe_name: str
    shanks: list[str]
    n_shanks: int
    output_directory: Path | None


@dataclass(frozen=True)
class OutputRootSet:
    """The output root was stored and the per-probe output was refreshed."""

    output_root: Path
    output_directory: Path | None


@dataclass(frozen=True)
class OutputDirectoryDerived:
    """The per-probe output directory was refreshed."""

    output_directory: Path | None


@dataclass(frozen=True)
class LoadDataPrepared:
    """State needed by the GUI before heavy data loading starts."""

    preserve_plot_selection: bool


@dataclass(frozen=True)
class PreviousAlignmentsLoaded:
    """Previous alignments were loaded for the active probe/shank."""

    alignments: AlignmentHistory


@dataclass(frozen=True)
class NoPreviousAlignments:
    """No previous alignments were available."""


@dataclass(frozen=True)
class AlignmentOutputBuilt:
    """Output dictionaries computed from the current alignment."""

    channel_results: dict
    ccf_channel_results: dict
    multi_shank: bool


@dataclass(frozen=True)
class AlignmentOutputsSaved:
    """Alignment output files were persisted."""

    saved: SavedAlignmentOutputs
    previous_alignments: AlignmentHistory


class AlignmentController:
    """Coordinates workflow commands across the document and loader.

    The controller owns command ordering and document mutations. It deliberately
    stays Qt-free; callers render returned results in the UI layer.
    """

    def __init__(
        self,
        document: AlignmentDocument,
        loader: Any,
        workflow_policy: WorkflowPolicy | None = None,
        alignment_repository: AlignmentRepository | None = None,
    ) -> None:
        self.document = document
        self.loader = loader
        self.workflow_policy = workflow_policy or WorkflowPolicy()
        self.alignment_repository = alignment_repository or AlignmentRepository()

    def can_load_data(self) -> PolicyResult:
        """Return whether the Load Data command can proceed."""
        return self.workflow_policy.can_load_data(self.document)

    def set_mouse_root(self, mouse_root: Path) -> MouseRootLoaded | Failed:
        """Load a mouse root through the loader and update document state."""
        if not mouse_root or str(mouse_root).strip() == "":
            return Failed("Empty mouse-root path provided")
        mouse_root = Path(mouse_root)
        if not mouse_root.is_dir():
            return Failed(f"Mouse-root is not a directory: {mouse_root}")

        old_root = (
            self.loader.mouse_root.root if self.loader.mouse_root is not None else None
        )
        try:
            loaded_root = self.loader.set_mouse_root(mouse_root)
        except Exception as exc:
            return Failed(f"Failed to load mouse root {mouse_root}: {exc}")

        self.document.set_mouse_root(mouse_root, mouse_id=loaded_root.mouse_id)
        root_changed = old_root is not None and old_root != loaded_root.root
        return MouseRootLoaded(loaded_root, root_changed=root_changed)

    def select_recording(self, recording_id: str) -> RecordingSelected | Failed:
        """Select a recording and return its available probes."""
        if self.loader.mouse_root is None:
            return Failed("No mouse root loaded. Please select a mouse root first.")
        if not recording_id:
            return Failed("No recording selected.")

        self.document.clear_probe()
        try:
            probes = self.loader.list_probes(recording_id)
        except Exception as exc:
            return Failed(f"Failed to list probes for {recording_id}: {exc}")
        return RecordingSelected(recording_id, probes=list(probes))

    def select_probe(
        self, recording_id: str, probe_name: str
    ) -> ProbeSelected | Failed:
        """Select a probe, load channel metadata, and refresh output state."""
        if self.loader.mouse_root is None:
            return Failed("No mouse root loaded. Please select a mouse root first.")
        if not recording_id:
            return Failed("No recording selected.")
        if not probe_name:
            return Failed("No probe selected.")

        self.document.select_probe(recording_id, probe_name)
        try:
            self.loader.select_probe(recording_id, probe_name)
            self.loader.load_channel_info()
            self.document.set_channel_info_loaded(True)
            shanks = self.loader.get_shank_list() or []
            output_result = self.derive_output_directory()
        except Exception as exc:
            self.document.set_channel_info_loaded(False)
            return Failed(f"Failed to select probe {probe_name}: {exc}")

        if isinstance(output_result, Failed):
            return output_result

        return ProbeSelected(
            recording_id=recording_id,
            probe_name=probe_name,
            shanks=list(shanks),
            n_shanks=self.loader.n_shanks,
            output_directory=output_result.output_directory,
        )

    def set_output_root(self, output_root: Path) -> OutputRootSet | Failed:
        """Set the output root and derive the active probe output directory."""
        if not output_root or str(output_root).strip() == "":
            return Failed("Empty save-root path provided")
        output_root = Path(output_root)
        if not output_root.is_dir():
            return Failed(f"Save-root is not a directory: {output_root}")

        self.document.set_output_root(output_root)
        output_result = self.derive_output_directory()
        if isinstance(output_result, Failed):
            return output_result
        return OutputRootSet(output_root, output_result.output_directory)

    def derive_output_directory(self) -> OutputDirectoryDerived | Failed:
        """Derive the per-probe output directory from document + loader state."""
        probe = self.loader.probe_info
        output_root = self.document.output_root
        if (
            output_root is None
            or probe is None
            or probe.recording_id != self.document.selected_recording
            or probe.probe_name != self.document.selected_probe
        ):
            self.document.set_output_directory(None)
            return OutputDirectoryDerived(None)

        output_directory = output_root / probe.recording_id / probe.probe_name
        try:
            output_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return Failed(
                f"Failed to create probe output directory {output_directory}: {exc}"
            )
        self.document.set_output_directory(output_directory)
        return OutputDirectoryDerived(output_directory)

    def prepare_load_data(self) -> LoadDataPrepared:
        """Mark data unloaded and return render state for the upcoming load."""
        preserve_plot_selection = self.document.data_loaded
        self.document.mark_data_loaded(False)
        return LoadDataPrepared(preserve_plot_selection=preserve_plot_selection)

    def finish_load_data(self, shank_idx: int) -> None:
        """Record successful heavy data load for the active shank."""
        self.document.mark_data_loaded(True)
        self.document.set_selected_shank(shank_idx)

    def set_selected_shank(self, shank_idx: int) -> None:
        """Record the active shank selected by the user."""
        self.document.set_selected_shank(shank_idx)

    def can_load_previous_alignments(self) -> Ok | Failed:
        """Return whether previous alignments can be loaded."""
        if self.loader.n_shanks == 0:
            return Failed("Channel info not loaded. Please select a probe first.")
        if self.loader.probe_info is None:
            return Failed("No probe selected. Please select a probe first.")
        return Ok()

    def can_save_alignment_output(self) -> PolicyResult:
        """Return whether the current alignment output can be saved."""
        return self.workflow_policy.can_save_alignment_output(self.document)

    def load_previous_alignments(
        self,
        folder: Path | None,
        shank_idx: int,
        use_docdb: bool,
    ) -> PreviousAlignmentsLoaded | NoPreviousAlignments | Failed:
        """Load previous alignments for the selected probe/shank."""
        ready = self.can_load_previous_alignments()
        if isinstance(ready, Failed):
            return ready
        probe = self.loader.probe_info
        assert probe is not None

        try:
            loaded = self.alignment_repository.load_previous_alignments(
                folder=folder,
                recording_id=probe.recording_id,
                probe_name=probe.probe_name,
                shank_idx=shank_idx,
                n_shanks=self.loader.n_shanks,
                use_docdb=use_docdb,
            )
        except Exception as exc:
            return Failed(f"Failed to load previous alignments: {exc}")

        if loaded is None:
            return NoPreviousAlignments()
        return PreviousAlignmentsLoaded(loaded.alignments)

    def build_alignment_output(
        self,
        channel_locations_ras: Any,
        channel_coordinates: Any,
    ) -> AlignmentOutputBuilt | Failed:
        """Compute output dictionaries for the current alignment."""
        try:
            channel_results, ccf_channel_results, multi_shank = (
                self.loader.get_alignment_results(
                    channel_locations_ras,
                    channel_coordinates,
                )
            )
        except Exception as exc:
            return Failed(f"Failed to build alignment output: {exc}")
        return AlignmentOutputBuilt(
            channel_results=channel_results,
            ccf_channel_results=ccf_channel_results,
            multi_shank=multi_shank,
        )

    def save_alignment_output(
        self,
        output: AlignmentOutputBuilt,
        alignments: AlignmentHistory,
        shank_idx: int,
        use_docdb: bool,
    ) -> AlignmentOutputsSaved | Failed:
        """Persist output dictionaries and alignment history."""
        output_directory = self.document.output_directory
        if output_directory is None:
            return Failed("Choose an output folder before saving.")

        persistable_alignments = {
            key: value for key, value in alignments.items() if key != "auto"
        }
        try:
            saved = self.alignment_repository.save_alignment_outputs(
                output_directory=output_directory,
                shank_idx=shank_idx,
                multi_shank=output.multi_shank,
                channel_results=output.channel_results,
                previous_alignments=persistable_alignments,
                ccf_channel_results=output.ccf_channel_results,
                use_docdb=use_docdb,
            )
        except Exception as exc:
            return Failed(f"Failed to save alignment output: {exc}")

        return AlignmentOutputsSaved(
            saved=saved,
            previous_alignments=persistable_alignments,
        )
