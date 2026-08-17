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
    bnk_scale = []
    bnk_offset = []
    for i_x, x_coord in enumerate(np.unique(geometry.chn_coords[:, 0])):
        bnk_idx = np.where(geometry.chn_coords[:, 0] == x_coord)[0]

        bnk_ycoords = geometry.chn_coords[bnk_idx, 1]
        bnk_values = np.asarray(data)[bnk_idx]
        for segment_ycoords, segment_values, bnk_diff in _uniform_depth_segments(
            bnk_ycoords,
            bnk_values,
            fallback_pitch=float(geometry.chn_diff),
        ):
            logger.debug(
                "x=%s: bnk_diff=%s, chn_diff=%s, n_chns=%s",
                x_coord,
                bnk_diff,
                geometry.chn_diff,
                len(segment_ycoords),
            )
            bnk_full = _depth_grid(segment_ycoords, bnk_diff)
            bnk_vals = np.full((bnk_full.shape[0]), np.nan)
            idx_full = _grid_indices(
                bnk_full,
                segment_ycoords,
                bnk_diff,
            )
            bnk_vals[idx_full] = segment_values

            bnk_data_current = bnk_vals[np.newaxis, :]

            bnk_xscale = bank_size / bnk_data_current.shape[0]
            bnk_yoffset = np.min(segment_ycoords)
            bnk_xoffset = bank_size * i_x

            bnk_data.append(bnk_data_current)
            bnk_scale.append(np.array([bnk_xscale, bnk_diff]))
            bnk_offset.append(np.array([bnk_xoffset, bnk_yoffset]))

    return bnk_data, np.asarray(bnk_scale), np.asarray(bnk_offset)


def _uniform_depth_segments(
    ycoords,
    values,
    *,
    fallback_pitch: float,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Split one x column into affine-renderable depth segments.

    A pyqtgraph ImageItem has one affine transform, so a single strip can only
    be exact when its depths lie on one uniform grid. Mixed recording blocks can
    union channel maps into columns with internal gaps; drawing those as
    multiple strips preserves the physical depth of every contact without
    stretching over unsupported regions.
    """
    ycoords = np.asarray(ycoords, dtype=float)
    values = np.asarray(values)
    order = np.argsort(ycoords)
    ycoords = ycoords[order]
    values = values[order]
    if ycoords.size == 0:
        return []
    if ycoords.size == 1:
        return [(ycoords, values, fallback_pitch)]

    diffs = np.diff(ycoords)
    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    start = 0
    current_pitch = float(abs(diffs[0]))
    for idx in range(1, ycoords.size):
        incoming_pitch = float(abs(ycoords[idx] - ycoords[idx - 1]))
        if np.isclose(incoming_pitch, current_pitch):
            continue
        segments.append(
            _segment(
                ycoords[start:idx],
                values[start:idx],
                current_pitch,
                fallback_pitch=fallback_pitch,
            )
        )
        start = idx
        if idx < diffs.size:
            current_pitch = float(abs(diffs[idx]))
        else:
            current_pitch = fallback_pitch

    segments.append(
        _segment(
            ycoords[start:],
            values[start:],
            current_pitch,
            fallback_pitch=fallback_pitch,
        )
    )
    return segments


def _segment(
    ycoords: np.ndarray,
    values: np.ndarray,
    pitch: float,
    *,
    fallback_pitch: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if ycoords.size <= 1:
        return ycoords, values, fallback_pitch
    if pitch <= 0:
        pitch = fallback_pitch
    return ycoords, values, float(pitch)


def _depth_grid(ycoords: np.ndarray, pitch: float) -> np.ndarray:
    if ycoords.size <= 1:
        return np.asarray(ycoords, dtype=float)
    start = float(np.min(ycoords))
    stop = float(np.max(ycoords))
    return start + pitch * np.arange(round((stop - start) / pitch) + 1)


def _grid_indices(
    grid: np.ndarray,
    ycoords: np.ndarray,
    pitch: float,
) -> np.ndarray:
    if grid.size == ycoords.size and np.allclose(grid, ycoords):
        return np.arange(ycoords.size)
    start = float(grid[0])
    indices = np.rint((ycoords - start) / pitch).astype(int)
    if (
        np.any(indices < 0)
        or np.any(indices >= grid.size)
        or not np.allclose(grid[indices], ycoords)
    ):
        raise ValueError("Depth coordinates do not align to the inferred pitch grid")
    return indices
