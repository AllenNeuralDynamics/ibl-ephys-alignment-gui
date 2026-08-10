"""App query builder for ephys plot read models."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ephys_alignment_gui.application.queries.context import AlignmentQueryContext
from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.core.alignment_read_models import (
    ActiveShankPlotDataState,
    ClusterDetailRenderState,
)
from ephys_alignment_gui.plotting.menu_state import PlotMenuState, build_plot_menu_state
from ephys_alignment_gui.plotting.registry import (
    PlotMenu,
    PlotSpec,
    resolve_plot_bounds,
    resolve_plot_payload,
)
from ephys_alignment_gui.services.alignment_derived_data import (
    AlignmentDerivedDataService,
)

logger = logging.getLogger(__name__)


@dataclass
class EphysPlotQueries:
    """Build ephys plot and active stream read models."""

    context: AlignmentQueryContext
    display_state: AlignmentDisplayState
    derived_data_service: AlignmentDerivedDataService
    histology_context: Any | None = None

    def active_unit_filter(self) -> str:
        """Return the selected unit subset for active ephys plot data."""
        return self.display_state.unit_filter

    def prepare_active_shank_plot_data_state(
        self,
        *,
        unit_filter: str | None = None,
    ) -> ActiveShankPlotDataState | None:
        """Materialize active shank plot payload cache and return bounds."""
        stream_runtime = getattr(self.context.runtime, "active_stream_runtime", None)
        if stream_runtime is None:
            return None
        shank_idx = self.context.active_shank_idx()
        unit_filter = self.active_unit_filter() if unit_filter is None else unit_filter
        payload_cache = stream_runtime.filtered_plot_payload_cache_for_shank(
            shank_idx,
            unit_filter=unit_filter,
        )
        in_brain_depths_um = self.active_in_brain_depths_for_alignment()
        payload_cache.in_brain_depths_um = in_brain_depths_um
        return ActiveShankPlotDataState(
            key=self.context.document.selected_alignment_key,
            shank_idx=shank_idx,
            unit_filter=unit_filter,
            channel_min_um=float(getattr(payload_cache, "chn_min", 0.0)),
            channel_max_um=float(getattr(payload_cache, "chn_max", 0.0)),
            in_brain_depths_um=in_brain_depths_um,
        )

    def active_plot_menu_state(
        self,
        previous_selected_keys: Mapping[PlotMenu, str | None] | None = None,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> PlotMenuState:
        """Return available plot menu entries for the active shank."""
        payload_cache = self._active_payload_cache()
        return self._plot_menu_state_for_payload_cache(
            payload_cache,
            previous_selected_keys=previous_selected_keys,
            raw_image_payloads=raw_image_payloads,
        )

    def active_plot_spec(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> PlotSpec | None:
        """Return an available plot spec for the active shank."""
        payload_cache = self._active_payload_cache()
        state = self._plot_menu_state_for_payload_cache(
            payload_cache,
            raw_image_payloads=raw_image_payloads,
        )
        return self._find_plot_spec(state, spec_key)

    def active_plot_payload(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> Any:
        """Resolve a plot payload for the active shank."""
        payload_cache = self._active_payload_cache()
        state = self._plot_menu_state_for_payload_cache(
            payload_cache,
            raw_image_payloads=raw_image_payloads,
        )
        spec = self._find_plot_spec(state, spec_key)
        if spec is None:
            return None
        return resolve_plot_payload(payload_cache, spec)

    def active_plot_bounds(
        self,
        spec_key: str,
        *,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> Any:
        """Resolve optional plot bounds for the active shank."""
        payload_cache = self._active_payload_cache()
        state = self._plot_menu_state_for_payload_cache(
            payload_cache,
            raw_image_payloads=raw_image_payloads,
        )
        spec = self._find_plot_spec(state, spec_key)
        if spec is None:
            return None
        return resolve_plot_bounds(payload_cache, spec)

    def active_in_brain_depths_um(self) -> Any:
        """Return active plot payload cache in-brain depths, if available."""
        payload_cache = self._active_payload_cache()
        if payload_cache is None:
            return None
        return getattr(payload_cache, "in_brain_depths_um", None)

    def active_in_brain_depths_for_alignment(self) -> Any:
        """Return active channel depths whose aligned CCF annotation is not root."""
        context = self.context.active_alignment_context()
        if (
            context is None
            or self.histology_context is None
            or self.histology_context.brain_atlas is None
        ):
            return None
        try:
            channel_locations_ras = self.derived_data_service.compute_channel_locations(
                ephysalign=context.shank_runtime.ephysalign,
                feature=context.active_alignment.feature,
                track=context.active_alignment.track,
            )
            region_ids = self.histology_context.brain_atlas.get_labels(
                channel_locations_ras
            )
        except Exception:
            logger.warning(
                "Could not determine in-brain channels for probe cmap",
                exc_info=True,
            )
            return None
        in_brain = np.asarray(region_ids) != 0
        if not in_brain.any():
            return None
        return np.asarray(context.shank_runtime.chn_depths)[in_brain]

    def active_cluster_detail(
        self,
        cluster_idx: int,
    ) -> ClusterDetailRenderState | None:
        """Return autocorrelogram/template detail for one active cluster."""
        payload_cache = self._active_payload_cache()
        if payload_cache is None:
            return None
        autocorr, cluster_no = payload_cache.get_autocorr(cluster_idx)
        template_waveform = payload_cache.get_template_wf(cluster_idx)
        return ClusterDetailRenderState(
            cluster_no=cluster_no,
            autocorr=np.asarray(autocorr),
            t_autocorr=np.asarray(payload_cache.t_autocorr),
            template_waveform=np.asarray(template_waveform),
            t_template=np.asarray(payload_cache.t_template),
        )

    def active_session_notes(self) -> str:
        """Return notes for the active ephys stream, if any."""
        stream_runtime = getattr(self.context.runtime, "active_stream_runtime", None)
        if stream_runtime is None:
            return ""
        return stream_runtime.stream.session_notes

    def _plot_menu_state_for_payload_cache(
        self,
        payload_cache: Any,
        *,
        previous_selected_keys: Mapping[PlotMenu, str | None] | None = None,
        raw_image_payloads: Mapping[Any, Any] | None = None,
    ) -> PlotMenuState:
        return build_plot_menu_state(
            payload_cache,
            previous_selected_keys=previous_selected_keys,
            raw_image_payloads=raw_image_payloads,
        )

    def _find_plot_spec(
        self,
        state: PlotMenuState,
        spec_key: str,
    ) -> PlotSpec | None:
        for spec in state.specs:
            if spec.key == spec_key:
                return spec
        logger.warning("Ignoring unavailable plot spec %s", spec_key)
        return None

    def _active_payload_cache(self) -> Any:
        stream_runtime = getattr(self.context.runtime, "active_stream_runtime", None)
        if stream_runtime is None:
            return None
        return stream_runtime.plot_payload_cache_for_shank(
            self.context.active_shank_idx()
        )
