"""Reload missing runtime data required by edited-alignment saves."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.application.save_runtime_dependencies import (
    SaveRuntimeDependency,
    SaveRuntimeDependencyPlan,
)
from ephys_alignment_gui.application.workflow import Failed, Ok
from ephys_alignment_gui.core.controller import AlignmentController
from ephys_alignment_gui.io.load_data_job import (
    LoadDataJob,
    LoadDataJobCancelled,
    LoadDataJobCompleted,
    LoadDataJobRequest,
)
from ephys_alignment_gui.io.load_data_target import LoadDataJobTarget
from ephys_alignment_gui.plotting.payload_cache_factory import (
    EphysPlotPayloadCacheFactory,
)
from ephys_alignment_gui.runtime.ephys_stream import EphysStreamRuntime
from ephys_alignment_gui.runtime.histology_loader import (
    HistologyDataAlreadyLoaded,
    HistologyDataLoaded,
    HistologyDataUnavailable,
    HistologyRuntimeLoader,
)
from ephys_alignment_gui.runtime.session import SessionRuntime
from ephys_alignment_gui.services.ephys_data import EphysDataService
from ephys_alignment_gui.services.histology_data import HistologyDataContext
from ephys_alignment_gui.services.probe_track import ProbeTrackService


@dataclass
class SaveRuntimeRehydrator:
    """Reload and initialize inactive stream runtimes needed for saving."""

    controller: AlignmentController
    runtime: SessionRuntime
    ephys_data_service: EphysDataService
    load_data_job: LoadDataJob
    histology_runtime_loader: HistologyRuntimeLoader
    plot_payload_cache_factory: EphysPlotPayloadCacheFactory
    histology_context: HistologyDataContext
    probe_track_service: ProbeTrackService

    def rehydrate_missing(
        self,
        plan: SaveRuntimeDependencyPlan,
    ) -> Ok | Failed:
        """Reload every missing runtime dependency that can be resolved."""
        for dependency in plan.dependencies:
            if not dependency.needs_reload:
                continue
            result = self._rehydrate_dependency(dependency)
            if isinstance(result, Failed):
                return result
        return Ok()

    def _rehydrate_dependency(self, dependency: SaveRuntimeDependency) -> Ok | Failed:
        target = self._load_target_for_dependency(dependency)
        if isinstance(target, Failed):
            return target

        job_result = self.load_data_job.run(LoadDataJobRequest(target))
        if isinstance(job_result, Failed):
            return Failed(
                "Failed to reload runtime needed to save edited alignment for "
                f"{self._describe(dependency)}: {job_result.message}"
            )
        if isinstance(job_result, LoadDataJobCancelled):
            return Failed(
                "Reload cancelled while saving edited alignment for "
                f"{self._describe(dependency)}: {job_result.reason}"
            )
        if not isinstance(job_result, LoadDataJobCompleted):
            return Failed(
                "Failed to reload runtime needed to save edited alignment for "
                f"{self._describe(dependency)}."
            )
        if not target.same_identity(job_result.target):
            return Failed(
                "Reloaded runtime target does not match save dependency for "
                f"{self._describe(dependency)}."
            )
        if job_result.ephys.stream.stream_key != target.stream_key:
            return Failed(
                "Reloaded stream does not match save dependency for "
                f"{self._describe(dependency)}: "
                f"{job_result.ephys.stream.stream_key!r} != {target.stream_key!r}"
            )

        self.histology_runtime_loader.activate_result(
            job_result.histology,
            mouse_root=target.mouse_root,
        )
        brain_atlas = self._brain_atlas_from_job(job_result)
        if brain_atlas is None:
            message = (
                job_result.histology.message
                if isinstance(job_result.histology, HistologyDataUnavailable)
                else "Brain atlas is not loaded."
            )
            return Failed(
                "Cannot initialize runtime needed to save edited alignment for "
                f"{self._describe(dependency)}: {message}"
            )

        stream_runtime = EphysStreamRuntime(
            stream=job_result.ephys.stream,
            plot_payload_cache_factory=self.plot_payload_cache_factory,
        )
        shank_runtime = stream_runtime.shank_runtime_for(target.shank_idx)
        try:
            track_annotations_ras = self.probe_track_service.load_track_annotations(
                probe=target.probe_info,
                shank_idx=target.shank_idx,
                brain_atlas=brain_atlas,
            )
        except Exception as exc:
            return Failed(
                "Failed to load track annotations needed to save edited alignment "
                f"for {self._describe(dependency)}: {exc}"
            )

        initialized = self.controller.initialize_shank_runtime_for_key(
            dependency.key,
            shank_runtime,
            track_annotations_ras=track_annotations_ras,
            brain_atlas=brain_atlas,
        )
        if isinstance(initialized, Failed):
            return initialized
        self.runtime.cache_loaded_stream(stream_runtime, activate=False)
        return Ok()

    def _load_target_for_dependency(
        self,
        dependency: SaveRuntimeDependency,
    ) -> LoadDataJobTarget | Failed:
        if dependency.load_target is not None:
            return dependency.load_target
        if dependency.mouse_root is None or dependency.probe is None:
            return Failed(
                "Cannot reload runtime needed to save edited alignment for "
                f"{self._describe(dependency)}: missing datapackage metadata."
            )
        try:
            channel_table = self.ephys_data_service.load_channel_table(
                dependency.probe
            )
        except Exception as exc:
            return Failed(
                "Failed to load channel metadata needed to save edited alignment "
                f"for {self._describe(dependency)}: {exc}"
            )
        return LoadDataJobTarget(
            recording_id=dependency.key.recording_id,
            probe_name=dependency.probe.probe_name,
            stream_key=dependency.stream_key,
            shank_idx=dependency.key.shank_idx,
            mouse_root=dependency.mouse_root,
            probe_info=dependency.probe,
            channel_table=channel_table,
        )

    def _brain_atlas_from_job(
        self,
        job_result: LoadDataJobCompleted,
    ) -> object | None:
        histology = job_result.histology
        if isinstance(histology, HistologyDataLoaded | HistologyDataAlreadyLoaded):
            runtime_data = histology.runtime_data
            if runtime_data is not None:
                return runtime_data.brain_atlas
        return self.histology_context.brain_atlas

    @staticmethod
    def _describe(dependency: SaveRuntimeDependency) -> str:
        return (
            f"{dependency.key.recording_id}/{dependency.key.ephys_collection} "
            f"shank {dependency.key.shank_idx + 1}"
        )
