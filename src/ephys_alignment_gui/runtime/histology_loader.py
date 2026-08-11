"""Qt-free loader for subject-level histology runtime data."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.io.datapackage_loader import MouseRoot
from ephys_alignment_gui.services.histology_data import (
    HistologyDataContext,
    HistologyDataService,
    HistologyRuntimeData,
)


@dataclass(frozen=True)
class HistologyDataAlreadyLoaded:
    """Histology runtime data is already available."""

    runtime_data: HistologyRuntimeData | None = None


@dataclass(frozen=True)
class HistologyDataLoaded:
    """Histology runtime data was loaded into the context."""

    runtime_data: HistologyRuntimeData | None = None


@dataclass(frozen=True)
class HistologyDataUnavailable:
    """Histology runtime data could not be loaded, but ephys can continue."""

    message: str


HistologyLoadResult = (
    HistologyDataAlreadyLoaded | HistologyDataLoaded | HistologyDataUnavailable
)


class HistologyRuntimeLoader:
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
        mouse_root = self.data_context.mouse_root
        if mouse_root is None:
            return HistologyDataUnavailable(
                "Failed to load atlas/histology: No mouse root loaded"
            )
        return self.load_for_mouse_root(mouse_root, store=True)

    def load_for_mouse_root(
        self,
        mouse_root: MouseRoot,
        *,
        store: bool,
    ) -> HistologyLoadResult:
        """Load histology for one immutable mouse-root target."""
        if self._context_loaded_for(mouse_root):
            return HistologyDataAlreadyLoaded(
                getattr(self.histology_context, "runtime_data", None)
            )

        try:
            histology_data = self.histology_data_service.load(mouse_root)
        except Exception as exc:
            return HistologyDataUnavailable(f"Failed to load atlas/histology: {exc}")

        if store:
            self._store(histology_data, mouse_root)

        return HistologyDataLoaded(histology_data)

    def activate_result(
        self,
        result: HistologyLoadResult,
        *,
        mouse_root: MouseRoot,
    ) -> None:
        """Store a job-loaded histology result for the active mouse root."""
        if isinstance(result, HistologyDataLoaded) and result.runtime_data is not None:
            self._store(result.runtime_data, mouse_root)

    def _context_loaded_for(self, mouse_root: MouseRoot) -> bool:
        is_loaded_for = getattr(self.histology_context, "is_loaded_for", None)
        if callable(is_loaded_for):
            return bool(is_loaded_for(mouse_root))
        return self.histology_context.brain_atlas is not None

    def _store(
        self,
        histology_data: HistologyRuntimeData,
        mouse_root: MouseRoot,
    ) -> None:
        try:
            self.histology_context.set(histology_data, mouse_root=mouse_root)
        except TypeError:
            self.histology_context.set(histology_data)
