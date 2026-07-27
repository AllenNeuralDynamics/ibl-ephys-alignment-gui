"""Qt-free workflow for loading subject-level histology runtime data."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.histology_data_service import (
    HistologyDataContext,
    HistologyDataService,
)


@dataclass(frozen=True)
class HistologyDataAlreadyLoaded:
    """Histology runtime data is already available."""


@dataclass(frozen=True)
class HistologyDataLoaded:
    """Histology runtime data was loaded into the context."""


@dataclass(frozen=True)
class HistologyDataUnavailable:
    """Histology runtime data could not be loaded, but ephys can continue."""

    message: str


HistologyLoadResult = (
    HistologyDataAlreadyLoaded | HistologyDataLoaded | HistologyDataUnavailable
)


class HistologyDataWorkflow:
    """Load and cache subject-level histology runtime data."""

    def __init__(
        self,
        data_context: AlignmentDataContext,
        histology_data_service: HistologyDataService,
        histology_context: HistologyDataContext,
    ) -> None:
        self.data_context = data_context
        self.histology_data_service = histology_data_service
        self.histology_context = histology_context

    def load_if_needed(self) -> HistologyLoadResult:
        """Load histology runtime data unless it is already cached."""
        if self.histology_context.brain_atlas is not None:
            return HistologyDataAlreadyLoaded()

        mouse_root = self.data_context.mouse_root
        if mouse_root is None:
            return HistologyDataUnavailable(
                "Failed to load atlas/histology: No mouse root loaded"
            )

        try:
            histology_data = self.histology_data_service.load(mouse_root)
            self.histology_context.set(histology_data)
        except Exception as exc:
            return HistologyDataUnavailable(
                f"Failed to load atlas/histology: {exc}"
            )

        return HistologyDataLoaded()
