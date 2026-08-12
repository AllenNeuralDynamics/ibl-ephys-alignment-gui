"""Compatibility re-export for core workflow policy/result primitives.

Core, IO, runtime, and service modules must import from
``ephys_alignment_gui.core.workflow`` directly so dependencies keep pointing
inward.
"""

from __future__ import annotations

from ephys_alignment_gui.core.workflow import (
    CHANNEL_INFO_REQUIRED,
    CHOOSE_OUTPUT_FOLDER,
    OUTPUT_REQUIRED,
    PROBE_REQUIRED,
    Blocked,
    CommandResult,
    Failed,
    Ok,
    PolicyResult,
    Requirement,
    WorkflowPolicy,
)

__all__ = [
    "CHANNEL_INFO_REQUIRED",
    "CHOOSE_OUTPUT_FOLDER",
    "OUTPUT_REQUIRED",
    "PROBE_REQUIRED",
    "Blocked",
    "CommandResult",
    "Failed",
    "Ok",
    "PolicyResult",
    "Requirement",
    "WorkflowPolicy",
]
