"""Qt-free loader for subject-level histology runtime data."""

from __future__ import annotations

import threading
from concurrent.futures import Future
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


@dataclass(frozen=True)
class _HistologyWarmup:
    """In-flight or completed mouse-root histology warmup result."""

    root: object
    mouse_root: MouseRoot
    future: Future[HistologyLoadResult]


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
        self._warmup_lock = threading.RLock()
        self._active_warmup: _HistologyWarmup | None = None
        self._completed_warmup: _HistologyWarmup | None = None

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

        warmup = self._warmup_for_mouse_root(mouse_root)
        if warmup is not None:
            result = warmup.future.result()
            if store:
                self.activate_result(result, mouse_root=mouse_root)
            return result

        result = self._load_uncached_for_mouse_root(mouse_root)
        if store:
            self.activate_result(result, mouse_root=mouse_root)
        return result

    def start_warmup_for_mouse_root(self, mouse_root: MouseRoot) -> bool:
        """Begin a background histology load for a mouse root, if useful."""
        if self._context_loaded_for(mouse_root):
            return False

        root = _root_key(mouse_root)
        with self._warmup_lock:
            if _warmup_matches(self._active_warmup, root) or _warmup_matches(
                self._completed_warmup,
                root,
            ):
                return False

            future: Future[HistologyLoadResult] = Future()
            warmup = _HistologyWarmup(root=root, mouse_root=mouse_root, future=future)
            self._active_warmup = warmup
            self._completed_warmup = None

        thread = threading.Thread(
            target=self._run_warmup,
            args=(warmup,),
            name=f"histology-warmup-{root}",
            daemon=True,
        )
        thread.start()
        return True

    def clear_warmup_results(self) -> None:
        """Forget in-flight/completed warmups after a mouse-root change."""
        with self._warmup_lock:
            self._active_warmup = None
            self._completed_warmup = None

    def _run_warmup(self, warmup: _HistologyWarmup) -> None:
        result = self._load_uncached_for_mouse_root(warmup.mouse_root)
        warmup.future.set_result(result)
        with self._warmup_lock:
            if self._active_warmup is warmup:
                self._active_warmup = None
                self._completed_warmup = warmup

    def _warmup_for_mouse_root(
        self,
        mouse_root: MouseRoot,
    ) -> _HistologyWarmup | None:
        root = _root_key(mouse_root)
        with self._warmup_lock:
            if _warmup_matches(self._active_warmup, root):
                return self._active_warmup
            if _warmup_matches(self._completed_warmup, root):
                return self._completed_warmup
        return None

    def _load_uncached_for_mouse_root(
        self,
        mouse_root: MouseRoot,
    ) -> HistologyLoadResult:
        try:
            histology_data = self.histology_data_service.load(mouse_root)
        except Exception as exc:
            return HistologyDataUnavailable(f"Failed to load atlas/histology: {exc}")

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


def _root_key(mouse_root: MouseRoot) -> object:
    return getattr(mouse_root, "root", mouse_root)


def _warmup_matches(warmup: _HistologyWarmup | None, root: object) -> bool:
    return warmup is not None and warmup.root == root
