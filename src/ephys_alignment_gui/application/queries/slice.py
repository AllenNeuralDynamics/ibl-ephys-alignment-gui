"""App query builder for coronal and perpendicular slice read models."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentDerivedDataService,
)
from ephys_alignment_gui.alignment_read_models import (
    ActiveSliceDataState,
    ActiveSliceMenuState,
    ActiveSliceRenderState,
    PerpendicularSliceRenderState,
)
from ephys_alignment_gui.application.queries.alignment_render import (
    AlignmentRenderQueries,
)
from ephys_alignment_gui.application.queries.context import AlignmentQueryContext
from ephys_alignment_gui.slice_data_runtime_service import SliceDataRuntimeService
from ephys_alignment_gui.slice_display_policy import SliceDisplayPolicy, SliceSelection

logger = logging.getLogger(__name__)


@dataclass
class SliceQueries:
    """Build anatomical slice and perpendicular slice read models."""

    context: AlignmentQueryContext
    render_queries: AlignmentRenderQueries
    derived_data_service: AlignmentDerivedDataService
    slice_data_runtime_service: SliceDataRuntimeService
    histology_context: Any | None = None
    slice_service: Any | None = None
    slice_display_policy: SliceDisplayPolicy | None = None

    def prepare_active_slice_screen_data(self) -> ActiveSliceDataState | None:
        """Materialize active slice data when histology runtime is available."""
        if not self._histology_slices_available():
            return None
        return self.ensure_active_slice_data_state()

    def ensure_active_slice_data_state(self) -> ActiveSliceDataState | None:
        """Build/cache and return coronal slice data for the active alignment."""
        context = self.context.active_alignment_context()
        if context is None or not self._histology_slices_available():
            return None
        return self.slice_data_runtime_service.ensure_coronal_slice_state(
            key=context.key,
            shank_runtime=context.shank_runtime,
            histology_context=self.histology_context,
            slice_service=self.slice_service,
        )

    def active_slice_data_state(self) -> ActiveSliceDataState | None:
        """Return currently active coronal slice data without building it."""
        context = self.context.active_alignment_context()
        if context is None or not self._histology_slices_available():
            return None
        return self.slice_data_runtime_service.cached_coronal_slice_state(
            key=context.key,
            shank_runtime=context.shank_runtime,
        )

    def active_slice_data_by_attr(self) -> dict[str, Any]:
        """Return active slice data keyed by menu payload data-attr names."""
        state = self.active_slice_data_state()
        if state is None:
            return {"slice_data": None, "fp_slice_data": None}
        return state.data_by_attr

    def active_slice_menu_state(
        self,
        *,
        offline: bool,
        previous_selection: SliceSelection | None = None,
    ) -> ActiveSliceMenuState | None:
        """Return menu and fallback-selection state for active slice data."""
        state = self.active_slice_data_state()
        if state is None or self.slice_display_policy is None:
            return None
        slice_data = state.slice_data or {}
        if not isinstance(slice_data, Mapping):
            return None
        fp_slice_data = (
            state.fp_slice_data if isinstance(state.fp_slice_data, Mapping) else None
        )
        items = self.slice_display_policy.menu_items(
            slice_data=slice_data,
            fp_slice_data=fp_slice_data,
            offline=offline,
        )
        default_selection = self.slice_display_policy.default_selection(slice_data)
        selection = self.slice_display_policy.choose_selection(
            previous=previous_selection,
            default=default_selection,
            data_by_attr=state.data_by_attr,
        )
        return ActiveSliceMenuState(
            key=state.key,
            items=tuple(items),
            default_selection=default_selection,
            selection=selection,
        )

    def active_slice_render_state(
        self,
        selection: SliceSelection,
    ) -> ActiveSliceRenderState | None:
        """Return a render payload for one active coronal slice selection."""
        slice_state = self.active_slice_data_state()
        context = self.context.active_alignment_context()
        if (
            slice_state is None
            or context is None
            or self.slice_display_policy is None
        ):
            return None
        data = slice_state.data_by_attr.get(selection.data_attr)
        if not isinstance(data, Mapping) or selection.key not in data:
            return None
        image = data[selection.key]
        decision = self.slice_display_policy.render_decision(data, selection.key)
        base_slice_data = slice_state.slice_data
        if not isinstance(base_slice_data, Mapping):
            base_slice_data = {}
        scale = np.asarray(data.get("scale", base_slice_data.get("scale")))
        offset = np.asarray(data.get("offset", base_slice_data.get("offset")))
        if scale.size < 2 or offset.size < 2:
            logger.warning(
                "Cannot render slice %s: missing scale/offset metadata",
                selection,
            )
            return None
        track_annos_and_ends_ras = context.shank_runtime.track_annos_and_ends_ras
        if track_annos_and_ends_ras is None:
            return None
        projection = self.derived_data_service.compute_channel_projection(
            ephysalign=context.shank_runtime.ephysalign,
            feature=context.active_alignment.feature,
            track=context.active_alignment.track,
        )
        return ActiveSliceRenderState(
            key=slice_state.key,
            selection=selection,
            image=image,
            scale=scale,
            offset=offset,
            decision=decision,
            track_annos_and_ends_ras=track_annos_and_ends_ras,
            projection=projection,
        )

    def active_perpendicular_slice_state(
        self,
        channel_name: str,
        *,
        extent_m: float = 500e-6,
        probe_margin_um: float = 100.0,
    ) -> PerpendicularSliceRenderState | None:
        """Build/cache and return a perpendicular slice render payload."""
        context = self.context.active_alignment_context()
        if context is None or not self._histology_slices_available():
            return None
        histology = self.render_queries.compute_active_histology(
            context.active_alignment,
            context.shank_runtime,
        )
        return self.slice_data_runtime_service.perpendicular_slice_state(
            key=context.key,
            active_alignment=context.active_alignment,
            shank_runtime=context.shank_runtime,
            histology=histology,
            histology_context=self.histology_context,
            slice_service=self.slice_service,
            channel_name=channel_name,
            extent_m=extent_m,
            probe_margin_um=probe_margin_um,
        )

    def _histology_slices_available(self) -> bool:
        return (
            self.histology_context is not None
            and getattr(self.histology_context, "brain_atlas", None) is not None
            and self.slice_service is not None
        )
