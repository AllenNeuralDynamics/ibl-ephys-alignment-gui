"""Runtime materialization for anatomical slice read models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_derived_data_service import AlignmentHistologyData
from ephys_alignment_gui.alignment_read_models import (
    ActiveSliceDataState,
    PerpendicularSliceRenderState,
)
from ephys_alignment_gui.document import AlignmentKey
from ephys_alignment_gui.shank_runtime import ShankRuntime
from ephys_alignment_gui.slice_runtime import SliceCacheEntry

logger = logging.getLogger(__name__)


class SliceDataRuntimeService:
    """Materialize slice read models from runtime caches and slice services."""

    def ensure_coronal_slice_state(
        self,
        *,
        key: AlignmentKey,
        shank_runtime: ShankRuntime,
        histology_context: Any,
        slice_service: Any,
    ) -> ActiveSliceDataState | None:
        """Build/cache and return coronal slice data for an alignment."""
        brain_atlas = self._brain_atlas(histology_context)
        if brain_atlas is None or slice_service is None:
            return None

        track = shank_runtime.ephysalign.track_interpolation_ras

        def build_slice() -> SliceCacheEntry:
            return SliceCacheEntry(
                slice_data=slice_service.build_slice_set(
                    brain_atlas=brain_atlas,
                    histology_images=histology_context.histology_images,
                    lazy_channel_paths=histology_context.lazy_channel_paths,
                    track_interpolation_ras=track,
                ),
                fp_slice_data=None,
            )

        entry = shank_runtime.slice_runtime.get_or_build_coronal_slice(
            alignment_key=key,
            track_interpolation_ras=track,
            builder=build_slice,
        )
        return ActiveSliceDataState(
            key=key,
            slice_data=entry.slice_data,
            fp_slice_data=entry.fp_slice_data,
        )

    def cached_coronal_slice_state(
        self,
        *,
        key: AlignmentKey,
        shank_runtime: ShankRuntime,
    ) -> ActiveSliceDataState | None:
        """Return cached coronal slice data for an alignment without building."""
        entry = shank_runtime.slice_runtime.cached_coronal_slice(
            alignment_key=key,
            track_interpolation_ras=shank_runtime.ephysalign.track_interpolation_ras,
        )
        if entry is None:
            return None
        return ActiveSliceDataState(
            key=key,
            slice_data=entry.slice_data,
            fp_slice_data=entry.fp_slice_data,
        )

    def perpendicular_slice_state(
        self,
        *,
        key: AlignmentKey,
        active_alignment: ActiveAlignment,
        shank_runtime: ShankRuntime,
        histology: AlignmentHistologyData,
        histology_context: Any,
        slice_service: Any,
        channel_name: str,
        extent_m: float = 500e-6,
        probe_margin_um: float = 100.0,
    ) -> PerpendicularSliceRenderState | None:
        """Build/cache and return a perpendicular slice render payload."""
        brain_atlas = self._brain_atlas(histology_context)
        if brain_atlas is None or slice_service is None:
            return None

        grid = self.perpendicular_feature_grid_um(
            shank_runtime=shank_runtime,
            histology=histology,
            brain_atlas=brain_atlas,
            extent_m=extent_m,
            probe_margin_um=probe_margin_um,
        )
        if grid is None:
            return None
        feature_grid_um, feature_grid_m, n_perp_samples = grid

        cache_key = shank_runtime.slice_runtime.perpendicular_key(
            alignment_key=key,
            channel_name=channel_name,
            track_interpolation_ras=shank_runtime.ephysalign.track_interpolation_ras,
            ephys_depths_along_track=(
                shank_runtime.ephysalign.ephys_depths_along_track
            ),
            feature_ref=active_alignment.feature,
            track_ref=active_alignment.track,
            feature_grid_m=feature_grid_m,
            extent_m=extent_m,
            n_perp_samples=n_perp_samples,
        )

        def build_perpendicular_image() -> NDArray[Any]:
            return slice_service.build_perpendicular_slice_image(
                brain_atlas=brain_atlas,
                histology_images=histology_context.histology_images,
                lazy_channel_paths=histology_context.lazy_channel_paths,
                ephysalign=shank_runtime.ephysalign,
                feature_ref=active_alignment.feature,
                track_ref=active_alignment.track,
                feature_grid_m=feature_grid_m,
                channel_name=channel_name,
                extent_m=extent_m,
                n_perp_samples=n_perp_samples,
            )

        try:
            image = shank_runtime.slice_runtime.get_or_build_perpendicular_slice(
                key=cache_key,
                builder=build_perpendicular_image,
            )
        except Exception:
            logger.warning(
                "Could not build perpendicular slice for channel '%s'",
                channel_name,
                exc_info=True,
            )
            return None

        return PerpendicularSliceRenderState(
            key=key,
            channel_name=channel_name,
            image=image,
            extent_um=float(extent_m) * 1e6,
            feature_min_um=float(feature_grid_um[0]),
            feature_max_um=float(feature_grid_um[-1]),
            n_perp_samples=n_perp_samples,
            n_depths=len(feature_grid_um),
            channel_depths_um=np.asarray(shank_runtime.chn_depths, dtype=float),
        )

    def perpendicular_feature_grid_um(
        self,
        *,
        shank_runtime: ShankRuntime,
        histology: AlignmentHistologyData,
        brain_atlas: Any,
        extent_m: float,
        probe_margin_um: float,
    ) -> tuple[NDArray[Any], NDArray[Any], int] | None:
        """Return feature-space grid bounds and sampling counts for perp slices."""
        depths = shank_runtime.chn_depths
        if depths is None:
            return None
        channel_depths_um = np.asarray(depths, dtype=float)
        if channel_depths_um.size == 0:
            return None
        finite_depths_um = channel_depths_um[np.isfinite(channel_depths_um)]
        if finite_depths_um.size == 0:
            return None

        dv_voxel_m = abs(float(brain_atlas.bc.dxyz[2]))
        if dv_voxel_m <= 0:
            return None

        feat_min_um = min(0.0, float(finite_depths_um.min())) - probe_margin_um
        feat_max_um = float(finite_depths_um.max()) + probe_margin_um
        regions = histology.histology.region
        try:
            has_regions = regions is not None and len(regions) > 0
        except TypeError:
            has_regions = regions is not None
        if has_regions:
            try:
                reg = np.asarray(regions, dtype=float)
            except (TypeError, ValueError):
                logger.debug("Could not coerce histology regions for slice bounds")
            else:
                reg = reg[np.isfinite(reg)]
                if reg.size:
                    feat_min_um = min(feat_min_um, float(reg.min()))
                    feat_max_um = max(feat_max_um, float(reg.max()))

        n_depths = int(round((feat_max_um - feat_min_um) * 1e-6 / dv_voxel_m)) + 1
        if n_depths <= 1:
            n_depths = 2
        feature_grid_um = np.linspace(feat_min_um, feat_max_um, n_depths)
        feature_grid_m = feature_grid_um * 1e-6
        n_perp_samples = int(round(2 * float(extent_m) / dv_voxel_m)) + 1
        if n_perp_samples <= 1:
            n_perp_samples = 2
        return feature_grid_um, feature_grid_m, n_perp_samples

    def _brain_atlas(self, histology_context: Any) -> Any | None:
        if histology_context is None:
            return None
        return getattr(histology_context, "brain_atlas", None)
