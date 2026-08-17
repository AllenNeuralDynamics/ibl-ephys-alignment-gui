"""Build alignment output dictionaries for persistence."""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
from iblatlas.regions import BrainRegions
from iblutil.util import Bunch
from numpy.typing import NDArray

from ephys_alignment_gui.core.alignment_output import (
    AlignmentOutputInput,
    CcfExportIssue,
    CcfExportStatus,
    ChannelOutputIdentity,
)
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.services.ants_points_transform import (
    AntsPointTransformCancelled,
    CancelTokenLike,
    apply_transforms_to_points,
)
from ephys_alignment_gui.services.ccf_transform_frame import (
    PIPELINE_FRAME,
    SPIM_NATIVE_FRAME,
)
from ephys_alignment_gui.services.histology_data import HistologyDataContext

logger = logging.getLogger(__name__)

ANTS_DIMENSION = 3
CCF_25UM_ML_BOUNDS_MM = (-5.739, 5.636)
CCF_ML_SAVE_MARGIN_MM = 1.0
CCF_ML_SAVE_BOUNDS_MM = (
    CCF_25UM_ML_BOUNDS_MM[0] - CCF_ML_SAVE_MARGIN_MM,
    CCF_25UM_ML_BOUNDS_MM[1] + CCF_ML_SAVE_MARGIN_MM,
)
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
        self.ccf_export_status_by_key: dict[Hashable, CcfExportStatus] = {}

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
        *,
        cancel_token: CancelTokenLike | None = None,
    ) -> dict[Hashable, AlignmentOutputResult]:
        """Compute channel outputs for many alignments with one ANTs call."""
        logger.info("Saving channel locations locally")
        self.ccf_export_status_by_key = {}
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
        all_ccf_xyz: NDArray | None
        try:
            all_ccf_xyz = self._transform_locations_to_ccf(
                all_channel_locations_ras,
                cancel_token=cancel_token,
            )
            all_ccf_xyz = self._validate_ccf_xyz_shape(
                all_ccf_xyz,
                expected_count=len(all_channel_locations_ras),
            )
        except AntsPointTransformCancelled:
            raise
        except Exception as exc:
            all_ccf_xyz = None
            message = (
                "CCF channel coordinate export failed before row-level validation; "
                "saving anatomical channel locations and alignment histories without "
                f"CCF channel coordinates: {exc}"
            )
            logger.warning(
                message,
                exc_info=True,
            )
            self._mark_all_ccf_failed(
                channel_dicts,
                reason="ccf_transform_failed",
                message=message,
            )

        results: dict[Hashable, AlignmentOutputResult] = {}
        for key, channel_dict in channel_dicts.items():
            if all_ccf_xyz is None:
                results[key] = (channel_dict, {}, multi_shank)
                continue
            ccf_xyz = all_ccf_xyz[point_slices[key]]
            valid_mask, status = self._ccf_validity_for_channel_dict(
                key,
                channel_dict,
                ccf_xyz,
            )
            self.ccf_export_status_by_key[key] = status
            results[key] = (
                channel_dict,
                self._create_ccf_channel_dict(
                    channel_dict,
                    ccf_xyz,
                    valid_mask=valid_mask,
                ),
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
        *,
        cancel_token: CancelTokenLike | None = None,
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
        transform_input_frame = getattr(
            brain_atlas,
            "ccf_transform_input_frame",
            PIPELINE_FRAME,
        )
        transform_input_frame_reason = getattr(
            brain_atlas,
            "ccf_transform_input_frame_reason",
            "legacy atlas without frame decision; defaulting to pipeline geometry",
        )
        logger.info(
            "Using %s CCF transform input frame for %d point(s): %s",
            transform_input_frame,
            len(channel_locations_ras),
            transform_input_frame_reason,
        )
        ras_to_lps = np.array([-1, -1, 1])
        # Convert IBL app world coordinates, RAS m, to ITK world coordinates, LPS mm
        channel_locations_lps_mm = 1e3 * ras_to_lps * channel_locations_ras_spim
        if transform_input_frame == PIPELINE_FRAME:
            transform_input_points: list[list[float]] = []
            for point in channel_locations_lps_mm:
                index = histology_img.TransformPhysicalPointToContinuousIndex(point)
                pipeline_point = pipeline_img.TransformContinuousIndexToPhysicalPoint(
                    index
                )
                transform_input_points.append(list(pipeline_point))
            transform_input_points_array = np.array(transform_input_points)
        elif transform_input_frame == SPIM_NATIVE_FRAME:
            transform_input_points_array = np.asarray(
                channel_locations_lps_mm,
                dtype=np.float64,
            )
        else:
            raise RuntimeError(
                "Unknown CCF transform input frame "
                f"{transform_input_frame!r}; expected {PIPELINE_FRAME!r} or "
                f"{SPIM_NATIVE_FRAME!r}"
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
        ccf_xyz = apply_transforms_to_points(
            transform_input_points_array,
            dimension=ANTS_DIMENSION,
            transforms=tx_list,
            whichtoinvert=invert_list,
            cancel_token=cancel_token,
        )
        logger.info("Done warping to ccf")
        return ccf_xyz

    @staticmethod
    def _create_ccf_channel_dict(
        channel_dict: dict[str, dict[str, Any]],
        ccf_xyz: NDArray,
        valid_mask: NDArray | None = None,
    ) -> dict[str, dict[str, Any]]:
        if len(ccf_xyz) != len(channel_dict):
            raise RuntimeError(
                "CCF transform returned a different number of points than the "
                f"channel output rows: {len(ccf_xyz)} != {len(channel_dict)}"
            )
        if valid_mask is None:
            valid_mask = np.ones(len(channel_dict), dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != (len(channel_dict),):
                raise RuntimeError(
                    "CCF validity mask has shape "
                    f"{valid_mask.shape}, expected ({len(channel_dict)},)"
                )
        ccf_channel_dict: dict[str, dict[str, Any]] = {}
        for ch, (x, y, z), is_valid in zip(channel_dict.keys(), ccf_xyz, valid_mask):
            if not is_valid:
                continue
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
    def _validate_ccf_xyz_shape(ccf_xyz: NDArray, *, expected_count: int) -> NDArray:
        ccf_xyz = np.asarray(ccf_xyz, dtype=np.float64)
        if ccf_xyz.shape != (expected_count, ANTS_DIMENSION):
            raise RuntimeError(
                "CCF transform returned coordinates with shape "
                f"{ccf_xyz.shape}, expected ({expected_count}, {ANTS_DIMENSION})"
            )
        return ccf_xyz

    def _ccf_validity_for_channel_dict(
        self,
        key: Hashable,
        channel_dict: dict[str, dict[str, Any]],
        ccf_xyz: NDArray,
    ) -> tuple[NDArray, CcfExportStatus]:
        if len(ccf_xyz) != len(channel_dict):
            raise RuntimeError(
                "CCF transform returned a different number of points than the "
                f"channel output rows: {len(ccf_xyz)} != {len(channel_dict)}"
            )
        if len(channel_dict) == 0:
            return np.zeros(0, dtype=bool), CcfExportStatus(
                status="complete",
                total_channel_count=0,
                ccf_channel_count=0,
                omitted_channel_count=0,
                in_brain_channel_count=0,
                bounds_ml_mm=CCF_ML_SAVE_BOUNDS_MM,
            )

        ccf_xyz = np.asarray(ccf_xyz, dtype=np.float64)
        finite_mask = np.all(np.isfinite(ccf_xyz), axis=1)
        in_brain_mask = self._in_brain_mask(channel_dict)
        ml_values = ccf_xyz[:, 0]
        ml_bounds_mask = (ml_values >= CCF_ML_SAVE_BOUNDS_MM[0]) & (
            ml_values <= CCF_ML_SAVE_BOUNDS_MM[1]
        )
        valid_mask = finite_mask & in_brain_mask & ml_bounds_mask
        issues = self._ccf_export_issues(
            in_brain_mask=in_brain_mask,
            finite_mask=finite_mask,
            ml_bounds_mask=ml_bounds_mask,
            ml_values=ml_values,
        )
        ccf_channel_count = int(np.count_nonzero(valid_mask))
        omitted_channel_count = int(len(channel_dict) - ccf_channel_count)
        status = "complete"
        if omitted_channel_count == len(channel_dict):
            status = "omitted"
        elif omitted_channel_count:
            status = "partial"

        in_brain_finite_ml = ml_values[in_brain_mask & finite_mask]
        export_status = CcfExportStatus(
            status=status,
            total_channel_count=len(channel_dict),
            ccf_channel_count=ccf_channel_count,
            omitted_channel_count=omitted_channel_count,
            in_brain_channel_count=int(np.count_nonzero(in_brain_mask)),
            bounds_ml_mm=CCF_ML_SAVE_BOUNDS_MM,
            in_brain_ml_range_mm=_range_tuple(in_brain_finite_ml),
            issues=tuple(issues),
        )

        if np.any(in_brain_mask & finite_mask & ~ml_bounds_mask):
            logger.warning(
                "CCF transform returned in-brain ML coordinates outside Allen "
                "CCF bounds plus %g mm margin for %s; omitting affected CCF "
                "rows. In-brain ML range: %s mm",
                CCF_ML_SAVE_MARGIN_MM,
                key,
                export_status.in_brain_ml_range_mm,
            )
        elif status == "omitted":
            logger.warning(
                "No valid in-brain CCF channel coordinates are available for %s; "
                "saving anatomical channel locations without CCF rows. Issues: %s",
                key,
                ", ".join(issue.reason for issue in issues) or "none",
            )
        elif status == "partial":
            logger.info(
                "Omitting %d of %d CCF channel rows for %s. Issues: %s",
                omitted_channel_count,
                len(channel_dict),
                key,
                ", ".join(issue.reason for issue in issues) or "none",
            )

        return valid_mask, export_status

    @staticmethod
    def _in_brain_mask(channel_dict: dict[str, dict[str, Any]]) -> NDArray:
        return np.asarray(
            [_is_in_brain_region(info) for info in channel_dict.values()],
            dtype=bool,
        )

    @staticmethod
    def _ccf_export_issues(
        *,
        in_brain_mask: NDArray,
        finite_mask: NDArray,
        ml_bounds_mask: NDArray,
        ml_values: NDArray,
    ) -> list[CcfExportIssue]:
        issues: list[CcfExportIssue] = []
        out_of_brain = ~in_brain_mask
        if np.any(out_of_brain):
            issues.append(
                CcfExportIssue(
                    reason="out_of_brain_channel_location",
                    message=(
                        "Channel location is outside the anatomical brain mask; "
                        "CCF coordinates are omitted because extrapolated CCF "
                        "transforms are not meaningful for these rows."
                    ),
                    channel_count=int(np.count_nonzero(out_of_brain)),
                    ml_range_mm=_range_tuple(ml_values[out_of_brain & finite_mask]),
                    bounds_ml_mm=CCF_ML_SAVE_BOUNDS_MM,
                )
            )
        in_brain_nonfinite = in_brain_mask & ~finite_mask
        if np.any(in_brain_nonfinite):
            issues.append(
                CcfExportIssue(
                    reason="nonfinite_ccf_coordinate",
                    message="CCF transform returned non-finite coordinates.",
                    channel_count=int(np.count_nonzero(in_brain_nonfinite)),
                    bounds_ml_mm=CCF_ML_SAVE_BOUNDS_MM,
                )
            )
        in_brain_out_of_bounds = in_brain_mask & finite_mask & ~ml_bounds_mask
        if np.any(in_brain_out_of_bounds):
            issues.append(
                CcfExportIssue(
                    reason="in_brain_ml_out_of_ccf_bounds",
                    message=(
                        "In-brain CCF ML coordinates are outside Allen CCF bounds "
                        "plus the save margin. This usually indicates an image "
                        "orientation/origin mismatch at the anatomical-to-CCF "
                        "save boundary."
                    ),
                    channel_count=int(np.count_nonzero(in_brain_out_of_bounds)),
                    ml_range_mm=_range_tuple(ml_values[in_brain_out_of_bounds]),
                    bounds_ml_mm=CCF_ML_SAVE_BOUNDS_MM,
                )
            )
        return issues

    def _mark_all_ccf_failed(
        self,
        channel_dicts: Mapping[Hashable, dict[str, dict[str, Any]]],
        *,
        reason: str,
        message: str,
    ) -> None:
        for key, channel_dict in channel_dicts.items():
            self.ccf_export_status_by_key[key] = CcfExportStatus(
                status="failed",
                total_channel_count=len(channel_dict),
                ccf_channel_count=0,
                omitted_channel_count=len(channel_dict),
                in_brain_channel_count=int(
                    np.count_nonzero(self._in_brain_mask(channel_dict))
                ),
                bounds_ml_mm=CCF_ML_SAVE_BOUNDS_MM,
                issues=(
                    CcfExportIssue(
                        reason=reason,
                        message=message,
                        channel_count=len(channel_dict),
                        bounds_ml_mm=CCF_ML_SAVE_BOUNDS_MM,
                    ),
                ),
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


def _is_in_brain_region(info: Mapping[str, Any]) -> bool:
    region_name = str(info.get("brain_region", "")).strip().lower()
    if region_name == "void":
        return False
    try:
        return int(info.get("brain_region_id", 0)) != 0
    except (TypeError, ValueError):
        return True


def _range_tuple(values: NDArray) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return (float(np.min(values)), float(np.max(values)))
