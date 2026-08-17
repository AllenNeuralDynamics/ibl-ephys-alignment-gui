"""Tests for probe-bank probe-plot image layout."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry
from ephys_alignment_gui.plotting.probe_bank_layout import arrange_channels_to_banks


def test_split_bank_columns_use_physical_depth_pitch() -> None:
    """Columns that cover disjoint depth ranges should not be stretched."""
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 15.0],
            [0.0, 30.0],
            [32.0, 45.0],
            [32.0, 60.0],
            [32.0, 75.0],
        ]
    )
    values = np.arange(coords.shape[0], dtype=float)

    imgs, scales, offsets = arrange_channels_to_banks(values, _geometry(coords))

    assert len(imgs) == 2
    np.testing.assert_allclose(scales[:, 1], np.array([15.0, 15.0]))
    np.testing.assert_allclose(offsets[:, 1], np.array([0.0, 45.0]))
    np.testing.assert_array_equal([img.shape for img in imgs], [(1, 3), (1, 3)])

    drawn_extents = (
        offsets[:, 1] + np.array([img.shape[1] for img in imgs]) * scales[:, 1]
    )
    np.testing.assert_allclose(drawn_extents, np.array([45.0, 90.0]))


def test_union_column_with_depth_gap_is_split_into_supported_segments() -> None:
    """Block-unioned columns can contain gaps that should stay unsupported."""
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 15.0],
            [0.0, 30.0],
            [0.0, 90.0],
            [0.0, 105.0],
            [32.0, 0.0],
            [32.0, 15.0],
            [32.0, 30.0],
            [32.0, 45.0],
            [32.0, 60.0],
        ]
    )
    values = np.arange(coords.shape[0], dtype=float)

    imgs, scales, offsets = arrange_channels_to_banks(values, _geometry(coords))

    assert len(imgs) == 3
    np.testing.assert_allclose(scales[:, 1], np.array([15.0, 15.0, 15.0]))
    np.testing.assert_allclose(offsets[:, 1], np.array([0.0, 90.0, 0.0]))
    np.testing.assert_allclose(offsets[:, 0], np.array([0.0, 0.0, 10.0]))
    np.testing.assert_array_equal([img.shape for img in imgs], [(1, 3), (1, 2), (1, 5)])


def _geometry(coords: np.ndarray) -> PlotChannelGeometry:
    unique_depths = np.unique(coords[:, 1])
    chn_diff = float(np.min(np.abs(np.diff(unique_depths))))
    chn_min = float(np.min(coords[:, 1]))
    chn_max = float(np.max(coords[:, 1]))
    chn_full = np.arange(chn_min, chn_max + chn_diff, chn_diff)
    chn_ind = np.arange(coords.shape[0], dtype=int)
    return PlotChannelGeometry(
        chn_coords_all=coords,
        chn_raw_ind_all=chn_ind,
        chn_contact_id_all=None,
        chn_ind_all=chn_ind,
        chn_shank_ind_all=np.zeros(coords.shape[0], dtype=int),
        chn_rows=chn_ind,
        chn_coords=coords,
        chn_ind=chn_ind,
        chn_min=chn_min,
        chn_max=chn_max,
        chn_diff=chn_diff,
        chn_full=chn_full,
        n_banks=int(len(np.unique(coords[:, 0]))),
        idx_full=np.where(np.isin(chn_full, coords[:, 1]))[0],
    )
