"""Cross-workflow foreground operation ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = Any


class ForegroundOperation(str, Enum):
    """Mutating workflows that must not overlap."""

    MOUSE_ROOT_CHANGE = "mouse root change"
    SELECTION_ACTIVATION = "selection activation"
    AUTOSAVE_RECOVERY = "autosave recovery"
    ALIGNMENT_IMPORT = "alignment import"
    FULL_SAVE = "full save"
    OUTPUT_PACKAGE_CHANGE = "output package change"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class ForegroundOperationConflict:
    """A requested foreground operation conflicts with the active owner."""

    requested: ForegroundOperation
    active: ForegroundOperation

    @property
    def message(self) -> str:
        """Return a user-facing conflict explanation."""
        return (
            f"Cannot start {self.requested.value} while "
            f"{self.active.value} is in progress."
        )


@dataclass
class ForegroundOperationLease:
    """Idempotent ownership token issued by a foreground-operation gate."""

    operation: ForegroundOperation
    generation: int
    _gate: ForegroundOperationGate = field(repr=False, compare=False)
    _released: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def active(self) -> bool:
        """Return whether this lease still owns the gate."""
        return not self._released and self._gate.owns(self)

    def release(self) -> None:
        """Release this lease once; repeated calls are harmless."""
        if self._released:
            return
        self._gate.release(self)
        self._released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> bool:
        self.release()
        return False


@dataclass
class ForegroundOperationGate:
    """Serialize document-replacing and document-publishing workflows."""

    _next_generation: int = field(default=1, init=False, repr=False)
    _active: ForegroundOperationLease | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _shutdown_requested: bool = field(default=False, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def try_acquire(
        self,
        operation: ForegroundOperation,
    ) -> ForegroundOperationLease | ForegroundOperationConflict:
        """Acquire exclusive foreground ownership or report the active owner."""
        with self._lock:
            if self._shutdown_requested:
                return ForegroundOperationConflict(
                    requested=operation,
                    active=ForegroundOperation.SHUTDOWN,
                )
            if self._active is not None:
                return ForegroundOperationConflict(
                    requested=operation,
                    active=self._active.operation,
                )
            lease = ForegroundOperationLease(
                operation=operation,
                generation=self._next_generation,
                _gate=self,
            )
            self._next_generation += 1
            self._active = lease
            return lease

    @property
    def active_operation(self) -> ForegroundOperation | None:
        """Return the operation currently holding exclusive ownership."""
        with self._lock:
            if self._active is not None:
                return self._active.operation
            if self._shutdown_requested:
                return ForegroundOperation.SHUTDOWN
            return None

    def request_shutdown(self) -> None:
        """Enter terminal shutdown mode while allowing the owner to settle."""
        with self._lock:
            self._shutdown_requested = True

    def owns(self, lease: ForegroundOperationLease) -> bool:
        """Return whether ``lease`` is the current active token."""
        with self._lock:
            return self._matches_active(lease)

    def release(self, lease: ForegroundOperationLease) -> None:
        """Release the active operation only when the token matches."""
        with self._lock:
            if self._matches_active(lease):
                self._active = None

    def _matches_active(self, lease: ForegroundOperationLease) -> bool:
        active = self._active
        return active is not None and active.generation == lease.generation
