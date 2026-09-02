"""Tests for application foreground-operation ownership."""

from __future__ import annotations

from ephys_alignment_gui.application.foreground_operations import (
    ForegroundOperation,
    ForegroundOperationConflict,
    ForegroundOperationGate,
    ForegroundOperationLease,
)


def test_gate_rejects_conflicting_operation_until_lease_released() -> None:
    gate = ForegroundOperationGate()
    load = gate.try_acquire(ForegroundOperation.SELECTION_ACTIVATION)
    assert isinstance(load, ForegroundOperationLease)

    conflict = gate.try_acquire(ForegroundOperation.FULL_SAVE)

    assert conflict == ForegroundOperationConflict(
        requested=ForegroundOperation.FULL_SAVE,
        active=ForegroundOperation.SELECTION_ACTIVATION,
    )
    assert "selection activation is in progress" in conflict.message

    load.release()
    save = gate.try_acquire(ForegroundOperation.FULL_SAVE)
    assert isinstance(save, ForegroundOperationLease)
    assert save.generation > load.generation


def test_lease_release_is_idempotent_and_stale_release_cannot_clear_new_owner() -> None:
    gate = ForegroundOperationGate()
    first = gate.try_acquire(ForegroundOperation.ALIGNMENT_IMPORT)
    assert isinstance(first, ForegroundOperationLease)
    first.release()
    first.release()

    second = gate.try_acquire(ForegroundOperation.AUTOSAVE_RECOVERY)
    assert isinstance(second, ForegroundOperationLease)
    gate.release(first)

    assert second.active
    assert gate.active_operation is ForegroundOperation.AUTOSAVE_RECOVERY

    second.release()
    assert gate.active_operation is None


def test_lease_context_releases_operation_after_error() -> None:
    gate = ForegroundOperationGate()
    lease = gate.try_acquire(ForegroundOperation.MOUSE_ROOT_CHANGE)
    assert isinstance(lease, ForegroundOperationLease)

    try:
        with lease:
            raise RuntimeError("failed")
    except RuntimeError:
        pass

    assert gate.active_operation is None


def test_shutdown_rejects_new_work_while_active_owner_settles() -> None:
    gate = ForegroundOperationGate()
    save = gate.try_acquire(ForegroundOperation.FULL_SAVE)
    assert isinstance(save, ForegroundOperationLease)

    gate.request_shutdown()
    conflict = gate.try_acquire(ForegroundOperation.SELECTION_ACTIVATION)

    assert isinstance(conflict, ForegroundOperationConflict)
    assert conflict.active is ForegroundOperation.SHUTDOWN
    assert gate.active_operation is ForegroundOperation.FULL_SAVE

    save.release()
    assert gate.active_operation is ForegroundOperation.SHUTDOWN
