"""App query builder for alignment and histology render read models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.alignment_read_models import (
    ActiveAlignmentRenderState,
    FitPlotRenderState,
    HistologyPanelRenderState,
    NearbyBoundaryRenderState,
    ProbeExtentRenderState,
    ScaleFactorRenderState,
)
from ephys_alignment_gui.application.queries.context import AlignmentQueryContext
from ephys_alignment_gui.services.alignment_derived_data import (
    AlignmentDerivedDataService,
    AlignmentHistologyData,
)


@dataclass
class AlignmentRenderQueries:
    """Build alignment, histology, and fit render read models."""

    context: AlignmentQueryContext
    display_state: AlignmentDisplayState
    derived_data_service: AlignmentDerivedDataService

    def active_histology_region_id(self, region_idx: int) -> int | None:
        """Return an active histology region id by plotted region index."""
        shank_runtime = self.context.active_shank_runtime()
        if shank_runtime is None or getattr(shank_runtime, "ephysalign", None) is None:
            return None
        try:
            return int(shank_runtime.ephysalign.region_id[region_idx][0])
        except (IndexError, TypeError, ValueError):
            return None

    def active_alignment_render_state(self) -> ActiveAlignmentRenderState | None:
        """Return derived render data for the active alignment, if available."""
        context = self.context.active_alignment_context()
        if context is None:
            return None
        return ActiveAlignmentRenderState(
            key=context.key,
            active_alignment=context.active_alignment,
            histology=self.compute_active_histology(
                context.active_alignment,
                context.shank_runtime,
            ),
            projection=self.derived_data_service.compute_channel_projection(
                ephysalign=context.shank_runtime.ephysalign,
                feature=context.active_alignment.feature,
                track=context.active_alignment.track,
            ),
        )

    def active_histology_panel_state(
        self,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> HistologyPanelRenderState | None:
        """Return histology-region render data for the active alignment."""
        context = self.context.active_alignment_context()
        if context is None:
            return None
        probe_extent = self._probe_extent_render_state(
            context.active_alignment,
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )
        if probe_extent is None:
            return None
        return HistologyPanelRenderState(
            key=context.key,
            histology=self.compute_active_histology(
                context.active_alignment,
                context.shank_runtime,
            ),
            probe_extent=probe_extent,
        )

    def probe_extent_render_state(
        self,
        active_alignment: ActiveAlignment,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> ProbeExtentRenderState | None:
        """Return probe-extent render data for an alignment."""
        return self._probe_extent_render_state(
            active_alignment,
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )

    def active_scale_factor_state(
        self,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> ScaleFactorRenderState | None:
        """Return scale-factor render data for the active alignment."""
        histology_state = self.active_histology_panel_state(
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )
        if histology_state is None:
            return None
        return ScaleFactorRenderState(
            key=histology_state.key,
            region=histology_state.histology.scale.region,
            scale=histology_state.histology.scale.scale,
            probe_extent=histology_state.probe_extent,
        )

    def active_nearby_boundary_state(
        self,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
        allen: Any,
        brain_atlas: Any,
        steps: int = 6,
    ) -> NearbyBoundaryRenderState | None:
        """Return nearby-boundary curves for the active alignment track."""
        context = self.context.active_alignment_context()
        if context is None:
            return None
        probe_extent = self._probe_extent_render_state(
            context.active_alignment,
            probe_tip_um=probe_tip_um,
            probe_top_um=probe_top_um,
            probe_extra_um=probe_extra_um,
        )
        if probe_extent is None:
            return None
        nearby_boundaries = context.shank_runtime.nearby_boundaries
        if nearby_boundaries is None:
            nearby_boundaries = self.derived_data_service.compute_nearby_boundaries(
                ephysalign=context.shank_runtime.ephysalign,
                allen=allen,
                brain_atlas=brain_atlas,
                steps=steps,
            )
            context.shank_runtime.nearby_boundaries = nearby_boundaries
        return NearbyBoundaryRenderState(
            key=context.key,
            x=nearby_boundaries.x,
            y=nearby_boundaries.y,
            colours=nearby_boundaries.colours,
            parent_x=nearby_boundaries.parent_x,
            parent_y=nearby_boundaries.parent_y,
            parent_colours=nearby_boundaries.parent_colours,
            probe_extent=probe_extent,
        )

    def active_fit_plot_state(
        self,
        *,
        depth_um: Any,
        lin_fit: bool,
    ) -> FitPlotRenderState | None:
        """Return feature/track fit curve render data for the active alignment."""
        context = self.context.active_alignment_context()
        if context is None:
            return None
        feature = np.asarray(context.active_alignment.feature, dtype=float)
        track = np.asarray(context.active_alignment.track, dtype=float)
        feature_um = feature * 1e6
        track_um = track * 1e6
        linear_feature_um = None
        linear_track_um = None
        depth_um = np.asarray(depth_um, dtype=float)
        if lin_fit and feature.size >= 5 and depth_um.size > 0:
            depth_lin = context.shank_runtime.ephysalign.feature2track_lin(
                depth_um / 1e6,
                feature,
                track,
            )
            if np.any(depth_lin):
                linear_feature_um = depth_um
                linear_track_um = np.asarray(depth_lin, dtype=float) * 1e6
        return FitPlotRenderState(
            key=context.key,
            feature_um=feature_um,
            track_um=track_um,
            linear_feature_um=linear_feature_um,
            linear_track_um=linear_track_um,
        )

    def compute_active_histology(
        self,
        active_alignment: ActiveAlignment,
        shank_runtime: Any,
    ) -> AlignmentHistologyData:
        """Compute histology render data for one active alignment."""
        return self.derived_data_service.compute_histology(
            ephysalign=shank_runtime.ephysalign,
            feature=active_alignment.feature,
            track=active_alignment.track,
            region_annotation_source=self.display_state.region_annotation_source,
            region_fp=shank_runtime.region_fp,
            region_label_fp=shank_runtime.region_label_fp,
            region_colour_fp=shank_runtime.region_colour_fp,
        )

    def _probe_extent_render_state(
        self,
        active_alignment: ActiveAlignment,
        *,
        probe_tip_um: float,
        probe_top_um: float,
        probe_extra_um: float,
    ) -> ProbeExtentRenderState | None:
        feature = np.asarray(active_alignment.feature, dtype=float)
        if feature.size == 0:
            return None

        offset_um = 1.0
        feature_min_um = float(feature[0] * 1e6)
        feature_max_um = float(feature[-1] * 1e6)
        feature_top_um = feature_max_um - offset_um
        if probe_top_um > feature_top_um:
            fallback_bounds = (
                feature_min_um + offset_um,
                feature_max_um - offset_um,
            )
            tip_bounds_um = fallback_bounds
            top_bounds_um = fallback_bounds
        else:
            tip_bounds_um = (
                feature_min_um + offset_um,
                feature_max_um - (probe_top_um + offset_um),
            )
            top_bounds_um = (
                feature_min_um + (probe_top_um + offset_um),
                feature_max_um - offset_um,
            )

        return ProbeExtentRenderState(
            probe_tip_um=float(probe_tip_um),
            probe_top_um=float(probe_top_um),
            probe_extra_um=float(probe_extra_um),
            feature_min_um=feature_min_um,
            feature_max_um=feature_max_um,
            tip_bounds_um=tip_bounds_um,
            top_bounds_um=top_bounds_um,
        )
