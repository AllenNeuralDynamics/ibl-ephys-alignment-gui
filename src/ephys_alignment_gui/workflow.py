"""Workflow policy and command-result primitives.

This module is intentionally Qt-free. It names application workflow rules so
button handlers can render policy failures instead of owning those rules.
"""

from __future__ import annotations

from dataclasses import dataclass

PROBE_REQUIRED = "probe_required"
CHANNEL_INFO_REQUIRED = "channel_info_required"
OUTPUT_REQUIRED = "output_required"
CHOOSE_OUTPUT_FOLDER = "choose_output_folder"


@dataclass(frozen=True)
class Requirement:
    """A missing precondition for an application command."""

    code: str
    message: str
    action: str | None = None


@dataclass(frozen=True)
class Ok:
    """Command or policy check can proceed."""


@dataclass(frozen=True)
class Blocked:
    """Command or policy check is blocked by one or more requirements."""

    requirements: tuple[Requirement, ...]

    @property
    def first(self) -> Requirement:
        """First unmet requirement, for UIs that render one prompt at a time."""
        return self.requirements[0]


@dataclass(frozen=True)
class Failed:
    """Command failed after its preconditions were satisfied."""

    message: str


CommandResult = Ok | Blocked | Failed
PolicyResult = Ok | Blocked


@dataclass(frozen=True)
class LoadDataState:
    """Minimal state needed to decide whether Load Data may run."""

    probe_selected: bool
    channel_info_loaded: bool
    output_directory_set: bool


class WorkflowPolicy:
    """Application workflow preconditions."""

    def can_load_data(self, state: LoadDataState) -> PolicyResult:
        """Return whether the Load Data command can proceed."""
        requirements: list[Requirement] = []
        if not state.probe_selected:
            requirements.append(Requirement(PROBE_REQUIRED, "Select a probe first."))
        if not state.channel_info_loaded:
            requirements.append(
                Requirement(
                    CHANNEL_INFO_REQUIRED,
                    "Channel info not loaded. Please select a probe first.",
                )
            )
        if not state.output_directory_set:
            requirements.append(
                Requirement(
                    OUTPUT_REQUIRED,
                    "Choose an output folder before loading data.",
                    action=CHOOSE_OUTPUT_FOLDER,
                )
            )
        if requirements:
            return Blocked(tuple(requirements))
        return Ok()
