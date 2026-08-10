"""App-level loaded-shank preparation commands."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.app_results import LoadedShankPrepared
from ephys_alignment_gui.controller import AlignmentController
from ephys_alignment_gui.histology_data_service import HistologyDataContext
from ephys_alignment_gui.probe_track_service import ProbeTrackService
from ephys_alignment_gui.session_runtime import SessionRuntime
from ephys_alignment_gui.workflow import Failed


@dataclass
class LoadedShankCommandHandler:
    """Prepare runtime/document state for rendering a loaded shank."""

    controller: AlignmentController
    data_context: AlignmentDataContext
    runtime: SessionRuntime
    histology_context: HistologyDataContext
    probe_track_service: ProbeTrackService

    def prepare_loaded_shank(
        self,
        shank_idx: int,
        *,
        select_default_alignment_if_empty: bool = True,
    ) -> LoadedShankPrepared | Failed:
        """Prepare Qt-free runtime state for a loaded active shank."""
        stream_runtime = self.runtime.active_stream_runtime
        if stream_runtime is None:
            return Failed("No active stream runtime for shank preparation")

        try:
            shank_runtime = stream_runtime.shank_runtime_for(shank_idx)
        except Exception as exc:
            return Failed(f"Failed to prepare shank runtime: {exc}")

        n_channels = len(shank_runtime.collection.depths)
        brain_atlas = self.histology_context.brain_atlas
        if brain_atlas is None:
            return LoadedShankPrepared(
                shank_idx=shank_idx,
                n_channels=n_channels,
                histology_available=False,
            )

        probe = self.data_context.probe_info
        if probe is None:
            return Failed("No probe selected. Please select a probe first.")

        try:
            track_annotations_ras = shank_runtime.track_annotations_ras
            if track_annotations_ras is None:
                track_annotations_ras = (
                    self.probe_track_service.load_track_annotations(
                        probe=probe,
                        shank_idx=shank_idx,
                        brain_atlas=brain_atlas,
                    )
                )
        except Exception as exc:
            return Failed(f"Failed to load shank track annotations: {exc}")

        choices = self.controller.active_alignment_choices(shank_idx)
        if isinstance(choices, Failed):
            return choices

        active_state = self.controller.document.active_alignment_state
        if (
            select_default_alignment_if_empty
            and active_state is not None
            and active_state.active_alignment is None
        ):
            selected = self.controller.select_previous_alignment(
                0,
                shank_idx=shank_idx,
            )
            if isinstance(selected, Failed):
                return selected

        initialized = self.controller.initialize_shank_runtime(
            shank_runtime,
            track_annotations_ras=track_annotations_ras,
            brain_atlas=brain_atlas,
        )
        if isinstance(initialized, Failed):
            return initialized

        return LoadedShankPrepared(
            shank_idx=shank_idx,
            n_channels=n_channels,
            histology_available=True,
            alignment_choices=choices.choices,
        )
