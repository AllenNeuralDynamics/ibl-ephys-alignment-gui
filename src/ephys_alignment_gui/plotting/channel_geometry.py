"""Qt-free channel geometry value object used by plot-data builders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.geometry.channel_geometry import (
    n_shanks_from_geometry,
    rows_for_shank,
    valid_shank_indices,
)
from ephys_alignment_gui.services.ephys_data import ChannelCollectionView

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlotChannelGeometry:
    """Channel-table geometry derived for one plotted channel collection."""

    chn_coords_all: NDArray
    chn_raw_ind_all: NDArray
    chn_contact_id_all: NDArray | None
    chn_ind_all: NDArray
    chn_shank_ind_all: NDArray | None
    chn_rows: NDArray
    chn_coords: NDArray
    chn_ind: NDArray
    chn_min: float
    chn_max: float
    chn_diff: float
    chn_full: NDArray
    n_banks: int
    idx_full: NDArray


def build_plot_channel_geometry(
    data: Any,
    shank_idx: int,
    *,
    channel_collection: ChannelCollectionView | None = None,
) -> PlotChannelGeometry:
    """Build plot-channel geometry from runtime or legacy channel metadata."""
    if channel_collection is None:
        return _from_alf_data(data, shank_idx)
    return _from_channel_collection(channel_collection)


def _from_channel_collection(
    channel_collection: ChannelCollectionView,
) -> PlotChannelGeometry:
    """Build plot-channel geometry from a runtime channel-collection view."""
    channel_table = channel_collection.channel_table
    chn_coords_all = np.asarray(channel_table.local_coordinates)
    chn_raw_ind_all = (
        channel_table.raw_ind
        if channel_table.raw_ind is not None
        else np.arange(chn_coords_all.shape[0])
    ).astype(int)
    chn_contact_id_all = channel_table.contact_ids
    chn_ind_all = np.arange(chn_coords_all.shape[0], dtype=int)
    chn_shank_ind_all = channel_table.shank_indices
    chn_rows = np.asarray(channel_collection.rows, dtype=int).copy()
    if chn_rows.size == 0:
        logger.warning(
            "No channels found for shank %d; falling back to all channels",
            channel_collection.shank_idx,
        )
        chn_rows = chn_ind_all
    return _finalize_geometry(
        chn_coords_all=chn_coords_all,
        chn_raw_ind_all=chn_raw_ind_all,
        chn_contact_id_all=chn_contact_id_all,
        chn_ind_all=chn_ind_all,
        chn_shank_ind_all=chn_shank_ind_all,
        chn_rows=chn_rows,
    )


def _from_alf_data(data: Any, shank_idx: int) -> PlotChannelGeometry:
    """Build plot-channel geometry from legacy ALF channel metadata."""
    chn_coords_all = np.asarray(data["channels"]["localCoordinates"])
    chn_raw_ind_all = (
        data["channels"].get("rawInd", np.arange(chn_coords_all.shape[0])).astype(int)
    )
    chn_contact_id_all = data["channels"].get("contactId")
    chn_ind_all = np.arange(chn_coords_all.shape[0], dtype=int)
    chn_shank_ind_all = valid_shank_indices(
        data["channels"].get("shankInd"),
        chn_coords_all.shape[0],
    )

    n_shanks = n_shanks_from_geometry(chn_coords_all, chn_shank_ind_all)
    chn_rows = rows_for_shank(
        chn_coords_all,
        chn_shank_ind_all,
        shank_idx,
        n_shanks,
    )
    if chn_rows.size == 0:
        logger.warning(
            "No channels found for shank %d; falling back to all channels",
            shank_idx,
        )
        chn_rows = chn_ind_all
    return _finalize_geometry(
        chn_coords_all=chn_coords_all,
        chn_raw_ind_all=chn_raw_ind_all,
        chn_contact_id_all=chn_contact_id_all,
        chn_ind_all=chn_ind_all,
        chn_shank_ind_all=chn_shank_ind_all,
        chn_rows=chn_rows,
    )


def _finalize_geometry(
    *,
    chn_coords_all: NDArray,
    chn_raw_ind_all: NDArray,
    chn_contact_id_all: NDArray | None,
    chn_ind_all: NDArray,
    chn_shank_ind_all: NDArray | None,
    chn_rows: NDArray,
) -> PlotChannelGeometry:
    """Dedupe, sort, and derive plotting geometry for selected channel rows."""
    chn_coords = chn_coords_all[chn_rows, :]
    chn_ind = chn_rows.copy()

    _, unique_idx = np.unique(chn_coords, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    chn_coords = chn_coords[unique_idx]
    chn_ind = chn_ind[unique_idx]
    chn_rows = chn_rows[unique_idx]

    chn_min = float(np.min(chn_coords[:, 1]))
    chn_max = float(np.max(chn_coords[:, 1]))
    unique_depths = np.unique(chn_coords[:, 1])
    chn_diff = (
        float(np.min(np.abs(np.diff(unique_depths))))
        if unique_depths.size > 1
        else 1.0
    )
    chn_full = np.arange(chn_min, chn_max + chn_diff, chn_diff)

    chn_sort = np.argsort(chn_coords[:, 1])
    chn_coords = chn_coords[chn_sort]
    chn_ind = chn_ind[chn_sort]
    chn_rows = chn_rows[chn_sort]

    n_banks = int(len(np.unique(chn_coords[:, 0])))
    idx_full = np.where(np.isin(chn_full, chn_coords[:, 1]))[0]

    return PlotChannelGeometry(
        chn_coords_all=chn_coords_all,
        chn_raw_ind_all=chn_raw_ind_all,
        chn_contact_id_all=chn_contact_id_all,
        chn_ind_all=chn_ind_all,
        chn_shank_ind_all=chn_shank_ind_all,
        chn_rows=chn_rows,
        chn_coords=chn_coords,
        chn_ind=chn_ind,
        chn_min=chn_min,
        chn_max=chn_max,
        chn_diff=chn_diff,
        chn_full=chn_full,
        n_banks=n_banks,
        idx_full=idx_full,
    )
