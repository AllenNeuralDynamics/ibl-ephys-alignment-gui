"""Build alignment output dictionaries for persistence."""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from typing import Any

import ants
import numpy as np
import pandas
from iblatlas.regions import BrainRegions
from iblutil.util import Bunch
from numpy.typing import NDArray

from ephys_alignment_gui.core.alignment_output import (
    AlignmentOutputInput,
    ChannelOutputIdentity,
)
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.services.histology_data import HistologyDataContext

logger = logging.getLogger(__name__)

ANTS_DIMENSION = 3
CCF_25UM_ML_BOUNDS_MM = (-5.739, 5.636)
CCF_ML_SAVE_MARGIN_MM = 1.0
AlignmentOutputResult = tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    bool,
]
AlignmentOutputInputLike = (
    AlignmentOutputInput
    | tuple[Any, Any]
    | tuple[Any, Any, ChannelOutputIdentity | None]
)


class AlignmentOutputService:
    """Compute histology-space and CCF-space channel output dictionaries."""

    def __init__(
        self,
        data_context: AlignmentDataContext,
        histology_context: HistologyDataContext,
    ) -> None:
        self.data_context = data_context
        self.histology_context = histology_context

    def get_alignment_results(
        self,
        channel_locations_ras: NDArray,
        chn_coords: NDArray,
        channel_identity: ChannelOutputIdentity | None = None,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        bool,
    ]:
        """Compute the histology-space + CCF channel dicts for a save."""
        results = self.get_alignment_results_batch(
            {
                "active": AlignmentOutputInput(
                    channel_locations_ras=channel_locations_ras,
                    channel_coordinates=chn_coords,
                    channel_identity=channel_identity,
                )
            }
        )
        return results["active"]

    def get_alignment_results_batch(
        self,
        alignments: Mapping[Hashable, AlignmentOutputInputLike],
    ) -> dict[Hashable, AlignmentOutputResult]:
        """Compute channel outputs for many alignments with one ANTs call."""
        logger.info("Saving channel locations locally")
        brain_atlas = self.histology_context.brain_atlas
        if brain_atlas is None:
            raise ValueError("Brain atlas not loaded, cannot save channel locations")
        multi_shank = self.data_context.n_shanks > 1

        channel_dicts: dict[Hashable, dict[str, dict[str, Any]]] = {}
        packed_points: list[NDArray] = []
        point_slices: dict[Hashable, slice] = {}
        start = 0
        for key, value in alignments.items():
            output_input = self._normalize_output_input(value)
            channel_locations_ras = output_input.channel_locations_ras
            logger.debug("Channels for %s: %s", key, channel_locations_ras)
            channel_dicts[key] = self._channel_dict_for_locations(
                channel_locations_ras,
                output_input.channel_coordinates,
                output_input.channel_identity,
            )
            stop = start + len(channel_locations_ras)
            point_slices[key] = slice(start, stop)
            packed_points.append(channel_locations_ras)
            start = stop

        if not packed_points:
            return {}

        all_channel_locations_ras = np.concatenate(packed_points, axis=0)
        all_ccf_xyz = self._transform_locations_to_ccf(all_channel_locations_ras)
        self._validate_ccf_xyz(
            all_ccf_xyz,
            expected_count=len(all_channel_locations_ras),
        )

        results: dict[Hashable, AlignmentOutputResult] = {}
        for key, channel_dict in channel_dicts.items():
            ccf_xyz = all_ccf_xyz[point_slices[key]]
            results[key] = (
                channel_dict,
                self._create_ccf_channel_dict(channel_dict, ccf_xyz),
                multi_shank,
            )
        return results

    def _channel_dict_for_locations(
        self,
        channel_locations_ras: NDArray,
        chn_coords: NDArray,
        channel_identity: ChannelOutputIdentity | None,
    ) -> dict[str, dict[str, Any]]:
        """Build the histology-space channel dict for one alignment."""
        brain_atlas = self.histology_context.brain_atlas
        if brain_atlas is None:
            raise ValueError("Brain atlas not loaded, cannot save channel locations")
        regions: BrainRegions = brain_atlas.regions
        brain_regions = regions.get(brain_atlas.get_labels(channel_locations_ras))
        # Persist xyz in SPIM-native coords so external tools reading the
        # output don't need to know about the GUI's display rotation.
        brain_regions["xyz"] = brain_atlas.unrotate_to_spim_native(
            channel_locations_ras
        )
        brain_regions["lateral"] = chn_coords[:, 0]
        brain_regions["axial"] = chn_coords[:, 1]
        identity = (channel_identity or ChannelOutputIdentity()).with_defaults(
            len(channel_locations_ras)
        )
        brain_regions["raw_ind"] = identity.raw_ind
        if identity.contact_id is not None:
            brain_regions["contact_id"] = identity.contact_id
        brain_regions["shank_idx"] = identity.shank_idx

        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        return self.create_channel_dict(brain_regions)

    def _transform_locations_to_ccf(
        self,
        channel_locations_ras: NDArray,
    ) -> NDArray:
        mouse_root = self.data_context.mouse_root
        brain_atlas = self.histology_context.brain_atlas
        if mouse_root is None or brain_atlas is None:
            raise RuntimeError(
                "Mouse root or brain atlas not loaded; cannot transform to CCF"
            )
        # Unrotate from the canonical (rotated) frame back to SPIM-native, then
        # use the pre-rotation (SPIM-native) sitk images for the index<->physical
        # math. The ANTs CCF chain was computed in SPIM-native coords and is
        # invalid for rotated inputs.
        channel_locations_ras_spim = brain_atlas.unrotate_to_spim_native(
            channel_locations_ras
        )
        histology_img = brain_atlas.intensity_sitk_image_spim_native
        pipeline_img = brain_atlas.pipeline_sitk_image_spim_native
        ras_to_lps = np.array([-1, -1, 1])
        # Convert IBL app world coordinates, RAS m, to ITK world coordinates, LPS mm
        channel_locations_lps_mm = 1e3 * ras_to_lps * channel_locations_ras_spim
        reg_pipeline_physical_points: list[list[float]] = []
        for point in channel_locations_lps_mm:
            index = histology_img.TransformPhysicalPointToContinuousIndex(point)
            pipeline_point = pipeline_img.TransformContinuousIndexToPhysicalPoint(index)
            reg_pipeline_physical_points.append(list(pipeline_point))

        reg_pipeline_physical_points_array = np.array(reg_pipeline_physical_points)

        logger.info("Warping to ccf")
        this_probe_df = pandas.DataFrame(
            reg_pipeline_physical_points_array, columns=list("xyz")
        )

        tx = mouse_root.transforms
        tx_list = [
            str(tx.image_to_template_affine),
            str(tx.image_to_template_warp),
            str(tx.template_to_ccf_affine),
            str(tx.template_to_ccf_warp),
        ]
        invert_list = [True, False, True, False]

        logger.info("applying transforms ...")
        ccf_coordinates_dataframe: pandas.DataFrame = ants.apply_transforms_to_points(
            ANTS_DIMENSION,
            this_probe_df,
            tx_list,
            whichtoinvert=invert_list,
        )
        logger.info("Done warping to ccf")

        return ccf_coordinates_dataframe.loc[:, ["x", "y", "z"]].to_numpy(
            dtype=np.float64
        )

    @staticmethod
    def _create_ccf_channel_dict(
        channel_dict: dict[str, dict[str, Any]],
        ccf_xyz: NDArray,
    ) -> dict[str, dict[str, Any]]:
        if len(ccf_xyz) != len(channel_dict):
            raise RuntimeError(
                "CCF transform returned a different number of points than the "
                f"channel output rows: {len(ccf_xyz)} != {len(channel_dict)}"
            )
        ccf_channel_dict: dict[str, dict[str, Any]] = {}
        for ch, (x, y, z) in zip(channel_dict.keys(), ccf_xyz):
            info = channel_dict[ch]
            ccf_channel_dict[ch] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "axial": info["axial"],
                "lateral": info["lateral"],
                "raw_ind": info["raw_ind"],
                "contact_id": info["contact_id"],
                "shank_idx": info["shank_idx"],
                "brain_region_id": info["brain_region_id"],
                "brain_region": info["brain_region"],
            }
        return ccf_channel_dict

    @staticmethod
    def _validate_ccf_xyz(ccf_xyz: NDArray, *, expected_count: int) -> None:
        ccf_xyz = np.asarray(ccf_xyz, dtype=np.float64)
        if ccf_xyz.shape != (expected_count, ANTS_DIMENSION):
            raise RuntimeError(
                "CCF transform returned coordinates with shape "
                f"{ccf_xyz.shape}, expected ({expected_count}, {ANTS_DIMENSION})"
            )
        if not np.all(np.isfinite(ccf_xyz)):
            raise RuntimeError("CCF transform returned non-finite coordinates")

        min_ml = CCF_25UM_ML_BOUNDS_MM[0] - CCF_ML_SAVE_MARGIN_MM
        max_ml = CCF_25UM_ML_BOUNDS_MM[1] + CCF_ML_SAVE_MARGIN_MM
        ml_values = ccf_xyz[:, 0]
        if np.any((ml_values < min_ml) | (ml_values > max_ml)):
            raise RuntimeError(
                "CCF transform returned ML coordinates outside Allen CCF bounds "
                f"plus {CCF_ML_SAVE_MARGIN_MM:g} mm margin: "
                f"range=({float(np.min(ml_values)):.3f}, "
                f"{float(np.max(ml_values)):.3f}) mm. This usually indicates "
                "an image orientation/origin mismatch at the anatomical-to-CCF "
                "save boundary."
            )

    @staticmethod
    def _normalize_output_input(value: Any) -> AlignmentOutputInput:
        if isinstance(value, AlignmentOutputInput):
            return value
        if len(value) == 2:
            channel_locations_ras, channel_coordinates = value
            return AlignmentOutputInput(
                channel_locations_ras=channel_locations_ras,
                channel_coordinates=channel_coordinates,
            )
        channel_locations_ras, channel_coordinates, channel_identity = value
        return AlignmentOutputInput(
            channel_locations_ras=channel_locations_ras,
            channel_coordinates=channel_coordinates,
            channel_identity=channel_identity,
        )

    @staticmethod
    def create_channel_dict(brain_regions: Bunch) -> dict[str, dict[str, Any]]:
        """Create the channel dictionary persisted to JSON output."""
        channel_dict: dict[str, dict[str, Any]] = {}

        for idx in range(brain_regions.id.size):
            channel = {
                "x": np.float64(brain_regions.xyz[idx, 0] * 1e6),
                "y": np.float64(brain_regions.xyz[idx, 1] * 1e6),
                "z": np.float64(brain_regions.xyz[idx, 2] * 1e6),
                "axial": np.float64(brain_regions.axial[idx]),
                "lateral": np.float64(brain_regions.lateral[idx]),
                "raw_ind": _json_scalar(
                    _optional_array_value(
                        getattr(brain_regions, "raw_ind", None),
                        idx,
                        default=idx,
                    )
                ),
                "contact_id": _optional_json_scalar(
                    getattr(brain_regions, "contact_id", None),
                    idx,
                ),
                "shank_idx": _json_scalar(
                    _optional_array_value(
                        getattr(brain_regions, "shank_idx", None),
                        idx,
                        default=0,
                    )
                ),
                "brain_region_id": int(brain_regions.id[idx]),
                "brain_region": brain_regions.acronym[idx],
            }
            channel_dict[f"channel_{idx}"] = channel

        return channel_dict


def _optional_json_scalar(values: Any | None, idx: int) -> Any | None:
    if values is None:
        return None
    return _json_scalar(values[idx])


def _optional_array_value(values: Any | None, idx: int, *, default: Any) -> Any:
    if values is None:
        return default
    return values[idx]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
