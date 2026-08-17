"""Build prepared alignment save inputs from document state and save geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ephys_alignment_gui.application.save_channel_locations import (
    AlignmentSaveChannelLocationBuilder,
    SaveChannelLocationError,
)
from ephys_alignment_gui.application.save_geometry_catalog import (
    SaveGeometry,
    SaveGeometryCatalog,
    SaveGeometryError,
)
from ephys_alignment_gui.core.alignment_output import AlignmentOutputInput
from ephys_alignment_gui.core.alignment_state import AlignmentState
from ephys_alignment_gui.core.document import AlignmentKey


class AlignmentSaveInputFactoryError(RuntimeError):
    """Raised when save input construction cannot proceed."""


@dataclass(frozen=True)
class AlignmentSaveInput:
    """Prepared document/geometry data needed to persist one alignment key."""

    state: AlignmentState
    output_input: AlignmentOutputInput
    output_metadata: Any
    output_directory: Path
    multi_shank: bool


@dataclass
class AlignmentSaveInputFactory:
    """Create save inputs without reaching into full stream runtimes."""

    save_geometry_catalog: SaveGeometryCatalog
    channel_location_builder: AlignmentSaveChannelLocationBuilder | None = None
    histology_context: Any | None = None

    def build(
        self,
        *,
        key: AlignmentKey,
        state: AlignmentState,
        output_directory: Path,
        channel_locations_ras: Any | None = None,
    ) -> AlignmentSaveInput:
        """Build one prepared save input from active alignment and geometry."""
        alignment = state.active_alignment
        if alignment is None:
            raise AlignmentSaveInputFactoryError(
                "Cannot build save input without an active alignment for "
                f"{key.recording_id}/{key.ephys_collection} shank {key.shank_idx + 1}."
            )
        try:
            geometry = self.save_geometry_catalog.geometry_for_key(key)
        except SaveGeometryError as exc:
            raise AlignmentSaveInputFactoryError(str(exc)) from exc
        if channel_locations_ras is None:
            channel_locations_ras = self._compute_channel_locations(
                geometry,
                feature=alignment.feature,
                track=alignment.track,
            )

        return AlignmentSaveInput(
            state=state,
            output_input=AlignmentOutputInput(
                channel_locations_ras=np.asarray(channel_locations_ras),
                channel_coordinates=geometry.channel_coordinates,
                channel_identity=geometry.channel_identity,
            ),
            output_metadata=geometry.output_metadata,
            output_directory=Path(output_directory),
            multi_shank=geometry.multi_shank,
        )

    def _compute_channel_locations(
        self,
        geometry: SaveGeometry,
        *,
        feature: Any,
        track: Any,
    ) -> Any:
        if self.channel_location_builder is None:
            raise AlignmentSaveInputFactoryError(
                "No save channel-location builder is configured."
            )
        brain_atlas = getattr(self.histology_context, "brain_atlas", None)
        if brain_atlas is None:
            raise AlignmentSaveInputFactoryError(
                "Brain atlas is not loaded, cannot prepare alignment save input."
            )
        try:
            return self.channel_location_builder.compute(
                geometry=geometry,
                feature=feature,
                track=track,
                brain_atlas=brain_atlas,
            )
        except SaveChannelLocationError as exc:
            raise AlignmentSaveInputFactoryError(str(exc)) from exc
