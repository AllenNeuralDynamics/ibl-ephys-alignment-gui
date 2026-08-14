"""Opt-in performance timing diagnostics."""

from __future__ import annotations

import contextvars
import logging
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

TIMING_ENV_VAR = "EPHYS_ALIGNMENT_GUI_TIMING"

_FALSE_VALUES = {"", "0", "false", "no", "off"}
_CURRENT_TIMING: contextvars.ContextVar[TimingSession | None] = (
    contextvars.ContextVar("ephys_alignment_gui_timing", default=None)
)


def timing_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether timing diagnostics should emit log entries."""
    environ = os.environ if environ is None else environ
    raw = environ.get(TIMING_ENV_VAR, "")
    return raw.strip().lower() not in _FALSE_VALUES


def start_timing(workflow: str, **fields: Any) -> TimingSession:
    """Create a timing session, disabled unless the timing env var is set."""
    return TimingSession(
        workflow=workflow,
        enabled=timing_enabled(),
        fields=dict(fields),
    )


def current_timing_session() -> TimingSession | None:
    """Return the active timing session for this call stack, if any."""
    return _CURRENT_TIMING.get()


@dataclass
class TimingSession:
    """Wall-clock timer that can span async callbacks when explicitly re-entered."""

    workflow: str
    enabled: bool
    fields: dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("ephys_alignment_gui.timing")
    )
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    _start: float = field(default_factory=time.perf_counter, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            self._log("start", status="started", extra=self.fields)

    @contextmanager
    def activate(self) -> Iterator[TimingSession]:
        """Make this session available to nested code in the current call stack."""
        if not self.enabled:
            yield self
            return
        token = _CURRENT_TIMING.set(self)
        try:
            yield self
        finally:
            _CURRENT_TIMING.reset(token)

    @contextmanager
    def step(self, name: str, **fields: Any) -> Iterator[None]:
        """Measure one named synchronous step."""
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        except Exception:
            self._log_step(name, start, status="failed", extra=fields)
            raise
        self._log_step(name, start, status="completed", extra=fields)

    def mark(self, name: str, **fields: Any) -> None:
        """Emit a point-in-time timing marker."""
        if not self.enabled:
            return
        self._log(name, status="mark", extra=fields)

    def finish(self, status: str = "completed", **fields: Any) -> None:
        """Emit the terminal timing entry for this session."""
        if not self.enabled or self._finished:
            return
        self._finished = True
        self._log("finish", status=status, extra=fields)

    def _log_step(
        self,
        name: str,
        start: float,
        *,
        status: str,
        extra: dict[str, Any],
    ) -> None:
        now = time.perf_counter()
        self._log(
            name,
            status=status,
            elapsed_ms=(now - start) * 1000,
            total_ms=(now - self._start) * 1000,
            extra=extra,
        )

    def _log(
        self,
        event: str,
        *,
        status: str,
        extra: dict[str, Any],
        elapsed_ms: float | None = None,
        total_ms: float | None = None,
    ) -> None:
        now = time.perf_counter()
        total_ms = (now - self._start) * 1000 if total_ms is None else total_ms
        elapsed_text = (
            " elapsed_ms=n/a" if elapsed_ms is None else f" elapsed_ms={elapsed_ms:.1f}"
        )
        self.logger.info(
            "timing trace=%s workflow=%s event=%s status=%s%s total_ms=%.1f%s",
            self.trace_id,
            self.workflow,
            event,
            status,
            elapsed_text,
            total_ms,
            _format_fields(extra),
        )


def _format_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    parts = [f"{key}={value!r}" for key, value in sorted(fields.items())]
    return " " + " ".join(parts)
