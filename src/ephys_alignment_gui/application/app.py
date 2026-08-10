"""UI-facing application port for the alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.application.commands import AlignmentCommands
from ephys_alignment_gui.application.queries import AlignmentQueries
from ephys_alignment_gui.core.event_bus import EventBus


@dataclass
class AlignmentApp:
    """Small public app port for desktop and future web frontends."""

    commands: AlignmentCommands
    queries: AlignmentQueries
    events: EventBus
