"""Runtime initialization for shank alignment engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.ephys_alignment import EphysAlignment
from ephys_alignment_gui.shank_runtime import ShankRuntime


@dataclass(frozen=True)
class InitializedShankAlignmentRuntime:
    """Derived runtime state produced when a shank alignment engine is built."""

    ephysalign: Any
    feature_init: NDArray[Any]
    track_init: NDArray[Any]
    track_annos_and_ends_ras: NDArray[Any]
    region_fp: Any
    region_label_fp: Any
    region_colour_fp: Any


@dataclass(frozen=True)
class AlignmentRuntimeService:
    """Build and attach runtime alignment state for loaded shanks."""

    alignment_cls: Any = EphysAlignment

    def initialize_shank_runtime(
        self,
        shank_runtime: ShankRuntime,
        *,
        track_annotations_ras: Any,
        brain_atlas: Any,
        feature_prev: Any = None,
        track_prev: Any = None,
    ) -> InitializedShankAlignmentRuntime:
        """Build an alignment engine and attach derived state to shank runtime."""
        alignment_kwargs = {
            "track_annotations_ras": track_annotations_ras,
            "chn_depths": shank_runtime.chn_depths,
            "brain_atlas": brain_atlas,
        }
        if self._has_previous_alignment(feature_prev, track_prev):
            alignment_kwargs["feature_prev"] = feature_prev
            alignment_kwargs["track_prev"] = track_prev

        ephysalign = self.alignment_cls(**alignment_kwargs)
        region_fp, region_label_fp, region_colour_fp, _ = (
            self.alignment_cls.get_histology_regions(
                ephysalign.track_interpolation_ras,
                ephysalign.ephys_depths_along_track,
                brain_atlas,
            )
        )
        feature_init, track_init, track_annos_and_ends_ras = (
            ephysalign.get_track_and_feature()
        )

        initialized = InitializedShankAlignmentRuntime(
            ephysalign=ephysalign,
            feature_init=np.asarray(feature_init, dtype=float),
            track_init=np.asarray(track_init, dtype=float),
            track_annos_and_ends_ras=np.asarray(
                track_annos_and_ends_ras,
                dtype=float,
            ),
            region_fp=region_fp,
            region_label_fp=region_label_fp,
            region_colour_fp=region_colour_fp,
        )
        self._attach(shank_runtime, track_annotations_ras, initialized)
        return initialized

    @staticmethod
    def _attach(
        shank_runtime: ShankRuntime,
        track_annotations_ras: Any,
        initialized: InitializedShankAlignmentRuntime,
    ) -> None:
        shank_runtime.track_annotations_ras = np.asarray(
            track_annotations_ras,
            dtype=float,
        )
        shank_runtime.ephysalign = initialized.ephysalign
        shank_runtime.region_fp = initialized.region_fp
        shank_runtime.region_label_fp = initialized.region_label_fp
        shank_runtime.region_colour_fp = initialized.region_colour_fp
        shank_runtime.track_annos_and_ends_ras = initialized.track_annos_and_ends_ras
        shank_runtime.nearby_boundaries = None

    @staticmethod
    def _has_previous_alignment(feature_prev: Any, track_prev: Any) -> bool:
        if feature_prev is None or track_prev is None:
            return False
        try:
            return bool(np.any(feature_prev) and np.any(track_prev))
        except Exception:
            return False
