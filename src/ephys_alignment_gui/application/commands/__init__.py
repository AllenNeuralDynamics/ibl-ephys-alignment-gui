"""Grouped command-side application facade for the alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.application.commands.alignment_edit import (
    AlignmentEditCommandHandler,
)
from ephys_alignment_gui.application.commands.alignment_persistence import (
    AlignmentPersistenceCommandHandler,
)
from ephys_alignment_gui.application.commands.display import DisplayCommandHandler
from ephys_alignment_gui.application.commands.load_data import LoadDataCommandHandler
from ephys_alignment_gui.application.commands.loaded_shank import (
    LoadedShankCommandHandler,
)
from ephys_alignment_gui.application.commands.metadata_selection import (
    MetadataSelectionCommandHandler,
)
from ephys_alignment_gui.application.commands.path import PathCommandHandler
from ephys_alignment_gui.application.commands.shank_selection import (
    ShankSelectionCommandHandler,
)


@dataclass(frozen=True)
class AlignmentCommands:
    """Grouped command app port for UI actions."""

    paths: PathCommandHandler
    metadata: MetadataSelectionCommandHandler
    shanks: ShankSelectionCommandHandler
    load: LoadDataCommandHandler
    loaded_shank: LoadedShankCommandHandler
    persistence: AlignmentPersistenceCommandHandler
    edit: AlignmentEditCommandHandler
    display: DisplayCommandHandler
