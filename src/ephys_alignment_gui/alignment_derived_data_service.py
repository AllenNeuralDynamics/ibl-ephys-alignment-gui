"""Qt-free derived data computations for an active alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.alignment_display_state import RegionAnnotationSource


@dataclass(frozen=True)
class HistologyPlotData:
    """Histology region data ready for plotting."""

    region: Any
    axis_label: Any
    colour: Any


@dataclass(frozen=True)
class ScaleFactorData:
    """Scale-factor data ready for plotting."""

    region: Any
    scale: Any


@dataclass(frozen=True)
class AlignmentHistologyData:
    """Histology and scale data derived from the active alignment."""

    histology: HistologyPlotData
    reference_histology: HistologyPlotData
    scale: ScaleFactorData


@dataclass(frozen=True)
class ChannelProjectionData:
    """Channel, tip, and reference-line geometry derived from an alignment."""

    channel_locations_ras: NDArray[Any]
    tip_location_ras: NDArray[Any]
    perpendicular_vectors: Any


@dataclass(frozen=True)
class NearbyBoundaryData:
    """Nearby-boundary curves derived from an alignment track."""

    x: Any
    y: Any
    colours: Any
    parent_x: Any
    parent_y: Any
    parent_colours: Any


class AlignmentDerivedDataService:
    """Computes non-Qt data derived from the active feature/track alignment."""

    def compute_histology(
        self,
        *,
        ephysalign: Any,
        feature: NDArray[Any],
        track: NDArray[Any],
        region_annotation_source: RegionAnnotationSource,
        region_fp: Any = None,
        region_label_fp: Any = None,
        region_colour_fp: Any = None,
    ) -> AlignmentHistologyData:
        """Compute scaled histology and scale factors for one alignment."""
        if region_annotation_source == "Allen":
            region, axis_label = ephysalign.scale_histology_regions(feature, track)
            colour = ephysalign.region_colour
            scale_region, scale = ephysalign.get_scale_factor(region)
            reference_region, reference_axis_label = ephysalign.scale_histology_regions(
                ephysalign.track_extent,
                ephysalign.track_extent,
            )
            reference_colour = ephysalign.region_colour
        elif region_annotation_source == "FranklinPaxinos":
            region, axis_label = ephysalign.scale_histology_regions(
                feature,
                track,
                region=region_fp,
                region_label=region_label_fp,
            )
            colour = region_colour_fp
            scale_region, scale = ephysalign.get_scale_factor(
                region,
                region_orig=region_fp,
            )
            reference_region, reference_axis_label = ephysalign.scale_histology_regions(
                ephysalign.track_extent,
                ephysalign.track_extent,
                region=region_fp,
                region_label=region_label_fp,
            )
            reference_colour = region_colour_fp
        else:
            raise ValueError(
                f"Unknown region annotation source: {region_annotation_source}"
            )

        return AlignmentHistologyData(
            histology=HistologyPlotData(
                region=region,
                axis_label=axis_label,
                colour=colour,
            ),
            reference_histology=HistologyPlotData(
                region=reference_region,
                axis_label=reference_axis_label,
                colour=reference_colour,
            ),
            scale=ScaleFactorData(region=scale_region, scale=scale),
        )

    def compute_channel_projection(
        self,
        *,
        ephysalign: Any,
        feature: NDArray[Any],
        track: NDArray[Any],
    ) -> ChannelProjectionData:
        """Compute channel/tip locations and perpendicular reference vectors."""
        return ChannelProjectionData(
            channel_locations_ras=ephysalign.get_channel_locations(feature, track),
            tip_location_ras=ephysalign.get_tip_location(feature, track),
            perpendicular_vectors=ephysalign.get_perp_vector(feature, track),
        )

    def compute_channel_locations(
        self,
        *,
        ephysalign: Any,
        feature: NDArray[Any],
        track: NDArray[Any],
    ) -> NDArray[Any]:
        """Compute only channel locations when plotting artifacts are not needed."""
        return np.asarray(ephysalign.get_channel_locations(feature, track))

    def compute_nearby_boundaries(
        self,
        *,
        ephysalign: Any,
        allen: Any,
        brain_atlas: Any,
        steps: int = 6,
    ) -> NearbyBoundaryData:
        """Compute nearby Allen region-boundary curves along an alignment track."""
        nearby_bounds = ephysalign.get_nearest_boundary(
            ephysalign.track_interpolation_ras,
            allen,
            steps=steps,
            brain_atlas=brain_atlas,
        )
        x, y, colours = ephysalign.arrange_into_regions(
            ephysalign.ephys_depths_along_track,
            nearby_bounds["id"],
            nearby_bounds["dist"],
            nearby_bounds["col"],
        )
        parent_x, parent_y, parent_colours = ephysalign.arrange_into_regions(
            ephysalign.ephys_depths_along_track,
            nearby_bounds["parent_id"],
            nearby_bounds["parent_dist"],
            nearby_bounds["parent_col"],
        )
        return NearbyBoundaryData(
            x=x,
            y=y,
            colours=colours,
            parent_x=parent_x,
            parent_y=parent_y,
            parent_colours=parent_colours,
        )
