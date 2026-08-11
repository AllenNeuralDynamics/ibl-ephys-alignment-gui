"""Foreground fresh-load execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.application.results import (
    FreshLoadExecution,
    LoadDataFreshPrepared,
)
from ephys_alignment_gui.io.load_data_job import LoadDataCancelToken


@dataclass(frozen=True)
class CancelledFreshLoadExecution:
    """A previously active fresh-load request was cancelled."""

    execution: FreshLoadExecution
    reason: str


@dataclass
class _ActiveFreshLoadExecution:
    execution: FreshLoadExecution
    cancel_token: LoadDataCancelToken


@dataclass
class LoadDataExecutionLifecycle:
    """Track foreground fresh-load requests independently from desktop code."""

    _next_load_id: int = 1
    _active: _ActiveFreshLoadExecution | None = None

    @property
    def active_execution(self) -> FreshLoadExecution | None:
        """Return the currently active foreground load execution."""
        if self._active is None:
            return None
        return self._active.execution

    def start(
        self,
        prepared: LoadDataFreshPrepared,
        *,
        cancel_token: LoadDataCancelToken | None = None,
        cancel_previous_reason: str = "superseded by a newer load request",
    ) -> tuple[FreshLoadExecution, CancelledFreshLoadExecution | None]:
        """Start a foreground load execution and cancel any previous one."""
        cancelled = self.cancel_active(cancel_previous_reason)
        execution = FreshLoadExecution(
            load_id=self._next_load_id,
            prepared=prepared,
        )
        self._next_load_id += 1
        self._active = _ActiveFreshLoadExecution(
            execution=execution,
            cancel_token=cancel_token or LoadDataCancelToken(),
        )
        return execution, cancelled

    def cancel_active(
        self,
        reason: str,
    ) -> CancelledFreshLoadExecution | None:
        """Request cancellation for the active load execution, if any."""
        if self._active is None:
            return None
        active = self._active
        active.cancel_token.cancel(reason)
        self._active = None
        return CancelledFreshLoadExecution(active.execution, reason)

    def cancel_token_for(
        self,
        execution: FreshLoadExecution,
    ) -> LoadDataCancelToken | None:
        """Return the cancellation token for the active execution."""
        if not self.is_active(execution):
            return None
        assert self._active is not None
        return self._active.cancel_token

    def is_active(self, execution: FreshLoadExecution) -> bool:
        """Return whether ``execution`` is still the active foreground load."""
        if self._active is None:
            return False
        active = self._active.execution
        return (
            active.load_id == execution.load_id
            and active.prepared.target.same_identity(execution.prepared.target)
        )

    def finish(self, execution: FreshLoadExecution) -> None:
        """Clear the active execution if it matches ``execution``."""
        if self.is_active(execution):
            self._active = None
