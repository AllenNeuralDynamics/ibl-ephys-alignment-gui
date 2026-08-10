"""Probe-bank image layout for probe plot payloads."""

from __future__ import annotations

import logging

import numpy as np

from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry

logger = logging.getLogger(__name__)

BNK_SIZE = 10


def arrange_channels_to_banks(
    data,
    geometry: PlotChannelGeometry,
    *,
    bank_size: int = BNK_SIZE,
):
    """Arrange one value per selected channel into probe-bank image payloads."""
    bnk_data = []
    bnk_scale = np.empty((geometry.n_banks, 2))
    bnk_offset = np.empty((geometry.n_banks, 2))
    for i_x, x_coord in enumerate(np.unique(geometry.chn_coords[:, 0])):
        bnk_idx = np.where(geometry.chn_coords[:, 0] == x_coord)[0]

        bnk_ycoords = geometry.chn_coords[bnk_idx, 1]
        bnk_ycoords_unique = np.unique(bnk_ycoords)
        bnk_diff = np.min(np.abs(np.diff(bnk_ycoords_unique)))
        logger.debug(
            "x=%s: bnk_diff=%s, chn_diff=%s, n_chns=%s",
            x_coord,
            bnk_diff,
            geometry.chn_diff,
            len(bnk_ycoords),
        )
        bnk_full = np.arange(
            np.min(bnk_ycoords),
            np.max(bnk_ycoords) + bnk_diff,
            bnk_diff,
        )
        bnk_vals = np.full((bnk_full.shape[0]), np.nan)
        idx_full = np.where(np.isin(bnk_full, bnk_ycoords_unique))[0]
        bnk_vals[idx_full] = data[bnk_idx]

        bnk_data_current = bnk_vals[np.newaxis, :]

        bnk_yscale = (geometry.chn_max - geometry.chn_min) / bnk_data_current.shape[1]
        bnk_xscale = bank_size / bnk_data_current.shape[0]
        bnk_yoffset = np.min(bnk_ycoords)
        bnk_xoffset = bank_size * i_x

        bnk_data.append(bnk_data_current)
        bnk_scale[i_x, :] = np.array([bnk_xscale, bnk_yscale])
        bnk_offset[i_x, :] = np.array([bnk_xoffset, bnk_yoffset])

    return bnk_data, bnk_scale, bnk_offset
