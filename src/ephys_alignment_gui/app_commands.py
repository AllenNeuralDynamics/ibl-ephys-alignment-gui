"""Grouped command-side application facade for the alignment workspace."""

from __future__ import annotations

from dataclasses import dataclass

from ephys_alignment_gui.alignment_edit_commands import AlignmentEditCommandHandler
from ephys_alignment_gui.alignment_persistence_commands import (
    AlignmentPersistenceCommandHandler,
)
from ephys_alignment_gui.display_commands import DisplayCommandHandler
from ephys_alignment_gui.load_data_commands import LoadDataCommandHandler
from ephys_alignment_gui.loaded_shank_commands import LoadedShankCommandHandler
from ephys_alignment_gui.metadata_selection_commands import (
    MetadataSelectionCommandHandler,
)
from ephys_alignment_gui.path_commands import PathCommandHandler
from ephys_alignment_gui.shank_selection_commands import ShankSelectionCommandHandler


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
