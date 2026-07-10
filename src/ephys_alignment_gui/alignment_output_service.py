"""Build alignment output dictionaries for persistence."""

from __future__ import annotations

import logging
import re
from typing import Any

import ants
import numpy as np
import pandas
from iblatlas.regions import BrainRegions
from iblutil.util import Bunch
from numpy.typing import NDArray

from ephys_alignment_gui.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.histology_data_service import HistologyDataContext

logger = logging.getLogger(__name__)

ANTS_DIMENSION = 3


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
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        bool,
    ]:
        """Compute the histology-space + CCF channel dicts for a save."""
        logger.info("Saving channel locations locally")
        logger.debug("Channels: %s", channel_locations_ras)
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

        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        channel_dict = self.create_channel_dict(brain_regions)
        ccf_channel_dict = self._transform_to_ccf(
            channel_locations_ras,
            channel_dict,
        )

        multi_shank = self.data_context.n_shanks > 1
        return channel_dict, ccf_channel_dict, multi_shank

    def _transform_to_ccf(
        self,
        channel_locations_ras: NDArray,
        channel_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
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

        ccf_channel_dict: dict[str, dict[str, Any]] = {}
        pattern = re.compile(r"channel_(\d+)")

        channel_indices = []
        channel_names = []
        for ch in channel_dict:
            match = pattern.match(ch)
            if match:
                channel_indices.append(int(match.group(1)))
                channel_names.append(ch)

        xyz_array = ccf_coordinates_dataframe.loc[
            channel_indices, ["x", "y", "z"]
        ].to_numpy(dtype=np.float64)

        for ch, (x, y, z) in zip(channel_names, xyz_array):
            info = channel_dict[ch]
            ccf_channel_dict[ch] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "axial": info["axial"],
                "lateral": info["lateral"],
                "brain_region_id": info["brain_region_id"],
                "brain_region": info["brain_region"],
            }
        return ccf_channel_dict

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
                "brain_region_id": int(brain_regions.id[idx]),
                "brain_region": brain_regions.acronym[idx],
            }
            channel_dict[f"channel_{idx}"] = channel

        return channel_dict
