"""UI-facing application port for the alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.app_commands import AlignmentCommands
from ephys_alignment_gui.app_queries import AlignmentQueries
from ephys_alignment_gui.event_bus import EventBus


@dataclass
class AlignmentApp:
    """Small public app port for desktop and future web frontends."""

    commands: AlignmentCommands
    queries: AlignmentQueries
    events: EventBus
