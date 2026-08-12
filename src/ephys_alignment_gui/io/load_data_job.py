"""Qt-free fresh load-data job boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.io.ephys_stream_loader import (
    EphysStreamLoader,
    LoadedEphysSelection,
)
from ephys_alignment_gui.io.load_data_target import LoadDataJobTarget
from ephys_alignment_gui.runtime.histology_loader import (
    HistologyDataUnavailable,
    HistologyLoadResult,
    HistologyRuntimeLoader,
)

LoadDataJobPhase = Literal["ephys", "histology", "complete", "cancelled"]
LoadDataJobStatus = Literal["started", "completed", "warning", "cancelled"]
LoadDataProgressCallback = Callable[["LoadDataJobProgress"], None]


@dataclass(frozen=True)
class LoadDataJobRequest:
    """Inputs for one fresh ephys/histology load job."""

    target: LoadDataJobTarget
    load_id: int | None = None


@dataclass
class LoadDataCancelToken:
    """Cooperative cancellation flag for synchronous or future worker jobs."""

    reason: str | None = None

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self.reason is not None

    def cancel(self, reason: str = "cancelled") -> None:
        """Request cancellation at the next job checkpoint."""
        self.reason = reason


@dataclass(frozen=True)
class LoadDataJobProgress:
    """Progress event emitted by a fresh load job."""

    target: LoadDataJobTarget
    phase: LoadDataJobPhase
    status: LoadDataJobStatus
    message: str
    load_id: int | None = None


@dataclass(frozen=True)
class LoadDataJobWarning:
    """Non-fatal issue encountered during fresh load."""

    target: LoadDataJobTarget
    phase: LoadDataJobPhase
    message: str


@dataclass(frozen=True)
class LoadDataJobCompleted:
    """Heavy fresh ephys/histology load work completed."""

    target: LoadDataJobTarget
    ephys: LoadedEphysSelection
    histology: HistologyLoadResult
    warnings: tuple[LoadDataJobWarning, ...] = ()


@dataclass(frozen=True)
class LoadDataJobCancelled:
    """Fresh load job was cancelled at a cooperative checkpoint."""

    target: LoadDataJobTarget
    reason: str


@dataclass
class LoadDataJob:
    """Run heavy fresh-load services without depending on Qt."""

    ephys_stream_loader: EphysStreamLoader
    histology_runtime_loader: HistologyRuntimeLoader

    def run(
        self,
        request: LoadDataJobRequest,
        *,
        progress: LoadDataProgressCallback | None = None,
        cancel_token: LoadDataCancelToken | None = None,
    ) -> LoadDataJobCompleted | LoadDataJobCancelled | Failed:
        """Load fresh ephys data and best-effort histology runtime data."""
        target = request.target
        load_id = request.load_id
        cancel_token = cancel_token or LoadDataCancelToken()

        cancelled = self._cancelled(target, cancel_token, progress, load_id)
        if cancelled is not None:
            return cancelled

        self._emit(
            progress,
            LoadDataJobProgress(
                target=target,
                phase="ephys",
                status="started",
                message="Loading ephys data...",
                load_id=load_id,
            ),
        )
        ephys_result = self._load_ephys(target)
        if isinstance(ephys_result, Failed):
            return ephys_result
        self._emit(
            progress,
            LoadDataJobProgress(
                target=target,
                phase="ephys",
                status="completed",
                message="Ephys data loaded",
                load_id=load_id,
            ),
        )

        cancelled = self._cancelled(target, cancel_token, progress, load_id)
        if cancelled is not None:
            return cancelled

        self._emit(
            progress,
            LoadDataJobProgress(
                target=target,
                phase="histology",
                status="started",
                message="Loading atlas and histology data...",
                load_id=load_id,
            ),
        )
        histology = self.histology_runtime_loader.load_for_mouse_root(
            target.mouse_root,
            store=False,
        )
        warnings = self._warnings_for_histology(target, histology)
        for warning in warnings:
            self._emit(
                progress,
                LoadDataJobProgress(
                    target=target,
                    phase=warning.phase,
                    status="warning",
                    message=warning.message,
                    load_id=load_id,
                ),
            )

        self._emit(
            progress,
            LoadDataJobProgress(
                target=target,
                phase="histology",
                status="completed",
                message="Atlas and histology load step complete",
                load_id=load_id,
            ),
        )

        cancelled = self._cancelled(target, cancel_token, progress, load_id)
        if cancelled is not None:
            return cancelled

        self._emit(
            progress,
            LoadDataJobProgress(
                target=target,
                phase="complete",
                status="completed",
                message="Fresh load job complete",
                load_id=load_id,
            ),
        )
        return LoadDataJobCompleted(
            target=target,
            ephys=ephys_result,
            histology=histology,
            warnings=warnings,
        )

    def _load_ephys(self, target: LoadDataJobTarget) -> LoadedEphysSelection | Failed:
        try:
            loaded = self.ephys_stream_loader.load_target(target)
            if not loaded.stream.ephys_dir:
                return Failed("Failed to load ephys data")
        except Exception as exc:
            return Failed(f"Failed to load ephys data: {exc}")
        return loaded

    def _cancelled(
        self,
        target: LoadDataJobTarget,
        cancel_token: LoadDataCancelToken,
        progress: LoadDataProgressCallback | None,
        load_id: int | None,
    ) -> LoadDataJobCancelled | None:
        if not cancel_token.cancelled:
            return None
        reason = cancel_token.reason or "cancelled"
        self._emit(
            progress,
            LoadDataJobProgress(
                target=target,
                phase="cancelled",
                status="cancelled",
                message=f"Load cancelled: {reason}",
                load_id=load_id,
            ),
        )
        return LoadDataJobCancelled(target=target, reason=reason)

    @staticmethod
    def _warnings_for_histology(
        target: LoadDataJobTarget,
        histology: HistologyLoadResult,
    ) -> tuple[LoadDataJobWarning, ...]:
        if isinstance(histology, HistologyDataUnavailable):
            return (
                LoadDataJobWarning(
                    target=target,
                    phase="histology",
                    message=histology.message,
                ),
            )
        return ()

    @staticmethod
    def _emit(
        progress: LoadDataProgressCallback | None,
        event: LoadDataJobProgress,
    ) -> None:
        if progress is not None:
            progress(event)
