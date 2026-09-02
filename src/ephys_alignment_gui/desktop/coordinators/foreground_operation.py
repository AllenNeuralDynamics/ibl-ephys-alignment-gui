"""Desktop helpers for application-owned foreground operation leases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ephys_alignment_gui.application.foreground_operations import (
    ForegroundOperation,
    ForegroundOperationConflict,
    ForegroundOperationLease,
)

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = Any


@dataclass
class NoOpForegroundOperationLease:
    """Compatibility lease for isolated coordinator tests without an app gate."""

    active: bool = True

    def release(self) -> None:
        self.active = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> bool:
        self.release()
        return False


def acquire_foreground_operation(
    gate: Any | None,
    operation: ForegroundOperation,
) -> (
    ForegroundOperationLease
    | NoOpForegroundOperationLease
    | ForegroundOperationConflict
):
    """Acquire from the application gate, or return a no-op test lease."""
    if gate is None:
        return NoOpForegroundOperationLease()
    return gate.try_acquire(operation)
