"""Qt-free fresh load-data job boundary."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.histology_data_workflow import (
    HistologyDataWorkflow,
    HistologyLoadResult,
)
from ephys_alignment_gui.probe_data_workflow import LoadedProbeData, ProbeDataWorkflow
from ephys_alignment_gui.workflow import Failed


@dataclass(frozen=True)
class LoadDataJobRequest:
    """Inputs for one fresh ephys/histology load job."""

    shank_idx: int


@dataclass(frozen=True)
class LoadDataJobCompleted:
    """Heavy fresh ephys/histology load work completed."""

    ephys: LoadedProbeData
    histology: HistologyLoadResult


@dataclass
class LoadDataJob:
    """Run heavy fresh-load services without depending on Qt."""

    probe_data_workflow: ProbeDataWorkflow
    histology_data_workflow: HistologyDataWorkflow

    def run(
        self,
        request: LoadDataJobRequest,
    ) -> LoadDataJobCompleted | Failed:
        """Load fresh ephys data and best-effort histology runtime data."""
        ephys_result = self._load_ephys(request.shank_idx)
        if isinstance(ephys_result, Failed):
            return ephys_result

        return LoadDataJobCompleted(
            ephys=ephys_result,
            histology=self.histology_data_workflow.load_if_needed(),
        )

    def _load_ephys(self, shank_idx: int) -> LoadedProbeData | Failed:
        try:
            loaded = self.probe_data_workflow.load(shank_idx)
            if not loaded.stream.ephys_dir:
                return Failed("Failed to load ephys data")
        except Exception as exc:
            return Failed(f"Failed to load ephys data: {exc}")
        return loaded
