"""Qt-free workflow policy and command-result primitives."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.core.document import AlignmentDocument

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


class WorkflowPolicy:
    """Application workflow preconditions."""

    def can_load_data(self, document: AlignmentDocument) -> PolicyResult:
        """Return whether the Load Data command can proceed."""
        requirements: list[Requirement] = []
        if not document.probe_selected:
            requirements.append(Requirement(PROBE_REQUIRED, "Select a probe first."))
        if not document.channel_info_loaded:
            requirements.append(
                Requirement(
                    CHANNEL_INFO_REQUIRED,
                    "Channel info not loaded. Please select a probe first.",
                )
            )
        if document.output_directory is None:
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

    def can_save_alignment_output(self, document: AlignmentDocument) -> PolicyResult:
        """Return whether the current alignment can be persisted."""
        if document.output_directory is None:
            return Blocked(
                (
                    Requirement(
                        OUTPUT_REQUIRED,
                        "Choose an output folder before saving.",
                        action=CHOOSE_OUTPUT_FOLDER,
                    ),
                )
            )
        return Ok()
