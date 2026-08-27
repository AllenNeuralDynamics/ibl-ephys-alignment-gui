"""Build LFP correlation/coherency plot-data payloads."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry
from ephys_alignment_gui.plotting.level_policy import in_brain_depth_mask
from ephys_alignment_gui.plotting.phase_color import (
    MEASURED_FLOOR,
    phase_magnitude_rgb,
)

logger = logging.getLogger(__name__)


def _phase_rgba(phase: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """uint8 RGBA for a coherency block.

    The diagonal is coherent with itself by construction, so it carries no
    information and is painted with the floor rather than a saturated colour.
    """
    rgb = phase_magnitude_rgb(phase, magnitude)
    rgba = np.ones((*rgb.shape[:2], 4), dtype=np.float32)
    rgba[:, :, :3] = rgb.astype(np.float32)
    for channel in range(3):
        np.fill_diagonal(rgba[:, :, channel], MEASURED_FLOOR[channel])
    return (rgba * 255).astype(np.uint8)


@dataclass(frozen=True)
class LfpCorrelationPlotDataBuilder:
    """Load external LFP correlation matrices for one plotted shank."""

    probe_path: Path
    shank_idx: int
    geometry: PlotChannelGeometry
    in_brain_depths_um: Any = None

    def build(self) -> dict[str, Any]:
        """Load LFP correlation and coherency payloads from ``band_corr``."""
        lfp_corr_folder = self._get_lfp_correlation_folder()
        if lfp_corr_folder is None:
            return {}

        row_channels = self._load_row_channels(lfp_corr_folder)
        matrix_rows = self._matrix_rows_for_current_shank(row_channels)
        block_groups = self._unique_block_row_groups(row_channels)

        lfp_corr_files = self._correlation_files(lfp_corr_folder, matrix_rows)
        if not lfp_corr_files:
            logger.warning("No LFP correlation files found in %s", lfp_corr_folder)
            return {}

        all_data = self._load_correlation_files(
            lfp_corr_files, matrix_rows, block_groups
        )
        all_data.update(
            self._load_coherency_files(lfp_corr_folder, matrix_rows, block_groups)
        )
        data_img_lfp_corr = self._sort_lfp_correlation_keys(all_data)
        logger.debug(
            "LFP correlation data loaded: %d epoch_bands",
            len(data_img_lfp_corr),
        )
        return data_img_lfp_corr

    def _load_row_channels(self, lfp_corr_folder: Path) -> dict[str, Any] | None:
        """Load producer row-to-channel metadata for LFP band matrices."""
        path = lfp_corr_folder / "row_channels.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            logger.warning("Failed to load row_channels.json", exc_info=True)
            return None

    def _matrix_rows_for_current_shank(
        self,
        row_channels: dict[str, Any] | None,
        *,
        warn: bool = True,
    ) -> np.ndarray | None:
        """Return channel-table rows for the active shank matrix."""
        if not row_channels:
            return None
        shanks = row_channels.get("shanks", {})
        entry = shanks.get(str(self.shank_idx))
        if entry is None:
            entry = shanks.get(str(self.shank_idx + 1))
        if entry is None:
            if warn:
                logger.warning(
                    "row_channels.json has no entry for shank %d",
                    self.shank_idx,
                )
            return None
        rows = np.asarray(entry.get("rows", []), dtype=int)
        if rows.size == 0:
            if warn:
                logger.warning(
                    "row_channels.json entry for shank %d has no rows",
                    self.shank_idx,
                )
            return None
        return rows

    def _unique_block_row_groups(
        self,
        row_channels: dict[str, Any] | None,
    ) -> list[tuple[str, np.ndarray]] | None:
        """Depth-sorted per-block row arrays, deduplicated by row-set identity.

        Returns ``None`` when block metadata is absent or only one unique
        depth range exists — the single-image path is used in that case.
        """
        if not row_channels:
            return None
        shanks = row_channels.get("shanks", {})
        entry = shanks.get(str(self.shank_idx))
        if entry is None:
            entry = shanks.get(str(self.shank_idx + 1))
        if entry is None:
            return None
        blocks = entry.get("blocks", [])
        if not blocks:
            return None
        seen: dict[tuple, bool] = {}
        groups: list[tuple[str, np.ndarray]] = []
        for block_entry in blocks:
            key = tuple(block_entry["rows"])
            if key not in seen:
                seen[key] = True
                groups.append(
                    (
                        block_entry.get("label", "block"),
                        np.asarray(block_entry["rows"], dtype=int),
                    )
                )
        return groups if len(groups) > 1 else None

    def _slice_band_matrix(
        self,
        matrix: np.ndarray,
        rows: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Slice a full channel-table matrix to active-shank rows if needed."""
        if rows is None:
            return matrix, None
        if matrix.shape[0] == len(self.geometry.chn_coords_all):
            return matrix[np.ix_(rows, rows)], rows
        if matrix.shape[0] == len(rows):
            return matrix, rows
        logger.warning(
            "Skipping LFP matrix with shape %s; expected full channel count %d "
            "or shank row count %d",
            matrix.shape,
            len(self.geometry.chn_coords_all),
            len(rows),
        )
        return None, None

    def _matrix_row_depths(
        self,
        rows: np.ndarray | None,
        n_matrix: int,
    ) -> np.ndarray | None:
        """Per-row channel depths (um) for a band matrix, or None if unknown."""
        if rows is not None and len(rows) == n_matrix:
            return self.geometry.chn_coords_all[rows, 1]
        if n_matrix == len(self.geometry.chn_coords):
            return self.geometry.chn_coords[:, 1]
        return None

    def _matrix_depth_geometry(
        self,
        rows: np.ndarray | None,
        n_matrix: int,
    ) -> tuple[float, float, np.ndarray]:
        """Affine display geometry derived from matrix channel rows."""
        depths = self._matrix_row_depths(rows, n_matrix)
        if depths is None:
            depths = np.array(
                [self.geometry.chn_min, self.geometry.chn_max],
                dtype=float,
            )

        unique_depths = np.unique(depths)
        if unique_depths.size > 1:
            pitch = np.min(np.diff(unique_depths))
            scale = (unique_depths[-1] - unique_depths[0] + pitch) / n_matrix
            offset = unique_depths[0]
        else:
            scale = 1.0
            offset = float(unique_depths[0]) if unique_depths.size else 0.0
        return (
            float(scale),
            float(offset),
            np.array([offset, offset + scale * n_matrix]),
        )

    def _full_band_files(self, folder: Path, suffix: str) -> list[Path]:
        """Band files excluding legacy ``_shank<N>`` compatibility views."""
        shank_suffix = re.compile(r"_shank\d+$")
        files = []
        for file in folder.glob(f"*{suffix}.npy"):
            stem = file.stem.replace(suffix, "")
            if not shank_suffix.search(stem):
                files.append(file)
        return sorted(files)

    def _get_lfp_correlation_folder(self) -> Path | None:
        """Return the producer LFP-correlation folder, if present."""
        lfp_corr_folder = self.probe_path / "band_corr"
        logger.debug("LFP corr search: %s", lfp_corr_folder)
        if lfp_corr_folder.exists():
            return lfp_corr_folder
        return None

    def _correlation_files(
        self,
        lfp_corr_folder: Path,
        matrix_rows: np.ndarray | None,
    ) -> list[Path]:
        """Return correlation matrix files for the active shank."""
        if matrix_rows is not None:
            lfp_corr_files = self._full_band_files(lfp_corr_folder, "_mean_corr")
        else:
            lfp_corr_files = []
        shank_glob = f"*_shank{self.shank_idx + 1}_mean_corr.npy"
        if not lfp_corr_files:
            lfp_corr_files = list(lfp_corr_folder.glob(shank_glob))
        if not lfp_corr_files:
            lfp_corr_files = self._full_band_files(lfp_corr_folder, "_mean_corr")
        return lfp_corr_files

    def _coherency_files(
        self,
        lfp_corr_folder: Path,
        matrix_rows: np.ndarray | None,
    ) -> list[Path]:
        """Return coherency matrix files for the active shank."""
        if matrix_rows is not None:
            coherency_files = self._full_band_files(lfp_corr_folder, "_coherency")
        else:
            coherency_files = []
        coh_glob = f"*_shank{self.shank_idx + 1}_coherency.npy"
        if not coherency_files:
            coherency_files = list(lfp_corr_folder.glob(coh_glob))
        if not coherency_files:
            coherency_files = self._full_band_files(lfp_corr_folder, "_coherency")
        return coherency_files

    def available_keys(self) -> tuple[str, ...]:
        """Return cheaply discoverable correlation/coherency payload keys."""
        lfp_corr_folder = self._get_lfp_correlation_folder()
        if lfp_corr_folder is None:
            return ()

        row_channels = self._load_row_channels(lfp_corr_folder)
        matrix_rows = self._matrix_rows_for_current_shank(row_channels, warn=False)

        lfp_corr_files = [
            file
            for file in self._correlation_files(lfp_corr_folder, matrix_rows)
            if self._matrix_file_can_render(file, matrix_rows)
        ]
        if not lfp_corr_files:
            return ()

        keys = {self._band_key_from_file(file, "_mean_corr") for file in lfp_corr_files}
        keys.update(
            f"{self._band_key_from_file(file, '_coherency')}_phase"
            for file in self._coherency_files(lfp_corr_folder, matrix_rows)
            if self._matrix_file_can_render(file, matrix_rows)
        )
        return tuple(self._sort_lfp_correlation_keys(dict.fromkeys(keys)))

    @staticmethod
    def _band_key_from_file(file: Path, suffix: str) -> str:
        """Return the registry key represented by one LFP matrix filename."""
        shank_suffix = re.compile(r"_shank\d+$")
        return shank_suffix.sub("", file.stem.replace(suffix, ""))

    def _matrix_file_can_render(
        self,
        file: Path,
        matrix_rows: np.ndarray | None,
    ) -> bool:
        """Return whether a matrix file has a renderable shape without loading data."""
        try:
            matrix = np.load(file, mmap_mode="r")
        except Exception:
            logger.warning(
                "Skipping unreadable LFP matrix file: %s", file, exc_info=True
            )
            return False

        if matrix.ndim < 2:
            return False

        n_rows = matrix.shape[0]
        if matrix_rows is None:
            return n_rows > 0
        return n_rows in {len(self.geometry.chn_coords_all), len(matrix_rows)}

    def _load_correlation_files(
        self,
        files: list[Path],
        matrix_rows: np.ndarray | None,
        block_groups: list[tuple[str, np.ndarray]] | None = None,
    ) -> dict[str, Any]:
        """Load real-valued LFP correlation matrices."""
        all_data = {}
        for file in files:
            band_name = self._band_key_from_file(file, "_mean_corr")
            full_matrix = np.load(file)

            if block_groups is not None:
                # Multi-block path: one ImageItem per unique depth range.
                # Shared 95th-percentile colorscale computed across all blocks.
                shared_max = self._block_max_corr(full_matrix, block_groups)
                imgs, scales_list, offsets_list, xranges_list = [], [], [], []
                for _label, block_rows in block_groups:
                    block_corr, _ = self._slice_band_matrix(full_matrix, block_rows)
                    if block_corr is None:
                        continue
                    np.fill_diagonal(block_corr, 0.0)
                    n_block = block_corr.shape[0]
                    scale, offset_y, x_range = self._matrix_depth_geometry(
                        block_rows, n_block
                    )
                    imgs.append(block_corr)
                    scales_list.append(np.array([scale, scale]))
                    offsets_list.append(np.array([offset_y, offset_y]))
                    xranges_list.append(x_range)
                if not imgs:
                    continue
                overall_xrange = np.array(
                    [
                        min(r[0] for r in xranges_list),
                        max(r[1] for r in xranges_list),
                    ]
                )
                all_data[band_name] = {
                    "img": imgs,
                    "scale": scales_list,
                    "levels": np.array([-shared_max, shared_max]),
                    "offset": offsets_list,
                    "xrange": overall_xrange,
                    "cmap": "RdBu_r",
                    "title": f"LFP correlation ({band_name})",
                    "xaxis": "Distance from probe tip (µm)",
                }
                continue

            # Single-image path (no block metadata).
            this_corr, depth_rows = self._slice_band_matrix(full_matrix, matrix_rows)
            if this_corr is None:
                continue

            logger.debug(
                "LFP corr file: %s, shape: %s, range: [%.4f, %.4f]",
                file.name,
                this_corr.shape,
                this_corr.min(),
                this_corr.max(),
            )

            np.fill_diagonal(this_corr, 0.0)
            n_matrix = this_corr.shape[0]
            scale, offset_y, x_range = self._matrix_depth_geometry(
                depth_rows,
                n_matrix,
            )

            mask = ~np.eye(n_matrix, dtype=bool)
            inb = in_brain_depth_mask(
                self._matrix_row_depths(depth_rows, n_matrix),
                self.in_brain_depths_um,
            )
            if inb is not None and inb.shape[0] == n_matrix:
                mask &= np.outer(inb, inb)
            max_corr = (
                np.quantile(np.abs(this_corr[mask]), 0.95) if np.any(mask) else 1.0
            )
            logger.debug(
                "LFP corr %s: n=%d, off-diag q95=%.4f, scale=%.4f",
                band_name,
                n_matrix,
                max_corr,
                scale,
            )
            all_data[band_name] = {
                "img": this_corr,
                "scale": np.array([scale, scale]),
                "levels": np.array([-max_corr, max_corr]),
                "offset": np.array([offset_y, offset_y]),
                "xrange": x_range,
                "cmap": "RdBu_r",
                "title": f"LFP correlation ({band_name})",
                "xaxis": "Distance from probe tip (µm)",
            }
        return all_data

    def _block_max_corr(
        self,
        full_matrix: np.ndarray,
        block_groups: list[tuple[str, np.ndarray]],
    ) -> float:
        """Shared 95th-percentile |correlation| across all block sub-matrices."""
        vals = []
        for _label, block_rows in block_groups:
            block_corr = full_matrix[np.ix_(block_rows, block_rows)]
            n = block_corr.shape[0]
            mask = ~np.eye(n, dtype=bool)
            inb = in_brain_depth_mask(
                self._matrix_row_depths(block_rows, n),
                self.in_brain_depths_um,
            )
            if inb is not None and inb.shape[0] == n:
                mask &= np.outer(inb, inb)
            if np.any(mask):
                vals.append(np.abs(block_corr[mask]))
        if not vals:
            return 1.0
        return float(np.quantile(np.concatenate(vals), 0.95))

    def _load_coherency_files(
        self,
        lfp_corr_folder: Path,
        matrix_rows: np.ndarray | None,
        block_groups: list[tuple[str, np.ndarray]] | None = None,
    ) -> dict[str, Any]:
        """Load complex coherency matrices and render cyclic phase images."""
        all_data = {}
        for file in self._coherency_files(lfp_corr_folder, matrix_rows):
            try:
                band_name = self._band_key_from_file(file, "_coherency")
                full_matrix = np.load(file)

                if block_groups is not None:
                    payload = self._coherency_block_payload(
                        full_matrix, block_groups, band_name
                    )
                    if payload is not None:
                        all_data[f"{band_name}_phase"] = payload
                    continue

                coh, depth_rows = self._slice_band_matrix(full_matrix, matrix_rows)
                if coh is None:
                    continue

                magnitude = np.abs(coh)
                phase = np.angle(coh)

                off_diag = ~np.eye(coh.shape[0], dtype=bool)
                inb = in_brain_depth_mask(
                    self._matrix_row_depths(depth_rows, coh.shape[0]),
                    self.in_brain_depths_um,
                )
                if inb is not None and inb.shape[0] == coh.shape[0]:
                    off_diag &= np.outer(inb, inb)
                max_mag = (
                    np.quantile(magnitude[off_diag], 0.95) if np.any(off_diag) else 1.0
                )
                if max_mag > 0:
                    magnitude = magnitude / max_mag

                n_coh = coh.shape[0]
                coh_scale, coh_offset, coh_x_range = self._matrix_depth_geometry(
                    depth_rows,
                    n_coh,
                )

                all_data[f"{band_name}_phase"] = {
                    "img": _phase_rgba(phase, magnitude),
                    "scale": np.array([coh_scale, coh_scale]),
                    "levels": None,
                    "offset": np.array([coh_offset, coh_offset]),
                    "xrange": coh_x_range,
                    "cmap": None,
                    "title": f"LFP coherency phase ({band_name})",
                    "xaxis": "Distance from probe tip (µm)",
                }
            except Exception:
                logger.warning(
                    "Failed to load coherency file: %s",
                    file,
                    exc_info=True,
                )
        return all_data

    def _coherency_phase_rgba(
        self,
        coh: np.ndarray,
        rows: np.ndarray | None,
    ) -> np.ndarray:
        """Cyclic phase image (uint8 RGBA) for one coherency sub-matrix."""
        magnitude = np.abs(coh)
        phase = np.angle(coh)
        n = coh.shape[0]

        off_diag = ~np.eye(n, dtype=bool)
        inb = in_brain_depth_mask(
            self._matrix_row_depths(rows, n),
            self.in_brain_depths_um,
        )
        if inb is not None and inb.shape[0] == n:
            off_diag &= np.outer(inb, inb)
        max_mag = np.quantile(magnitude[off_diag], 0.95) if np.any(off_diag) else 1.0
        if max_mag > 0:
            magnitude = magnitude / max_mag

        return _phase_rgba(phase, magnitude)

    def _coherency_block_payload(
        self,
        full_matrix: np.ndarray,
        block_groups: list[tuple[str, np.ndarray]],
        band_name: str,
    ) -> dict[str, Any] | None:
        """Per-block coherency phase images, one ImageItem per depth range."""
        imgs, scales, offsets, xranges = [], [], [], []
        for _label, block_rows in block_groups:
            coh, _ = self._slice_band_matrix(full_matrix, block_rows)
            if coh is None:
                continue
            n_block = coh.shape[0]
            scale, offset_y, x_range = self._matrix_depth_geometry(block_rows, n_block)
            imgs.append(self._coherency_phase_rgba(coh, block_rows))
            scales.append(np.array([scale, scale]))
            offsets.append(np.array([offset_y, offset_y]))
            xranges.append(x_range)
        if not imgs:
            return None
        return {
            "img": imgs,
            "scale": scales,
            "levels": None,
            "offset": offsets,
            "xrange": np.array(
                [min(r[0] for r in xranges), max(r[1] for r in xranges)]
            ),
            "cmap": None,
            "title": f"LFP coherency phase ({band_name})",
            "xaxis": "Distance from probe tip (µm)",
        }

    @staticmethod
    def _sort_lfp_correlation_keys(all_data: dict[str, Any]) -> dict[str, Any]:
        """Sort LFP correlation data keys by epoch and frequency band."""
        epoch_order = ["spont", "opto", "diff"]
        band_order = ["delta", "theta", "alpha", "beta", "gamma"]

        expected_keys = []
        other_keys = []
        for epoch in epoch_order:
            for band in band_order:
                expected_key = f"{epoch}_{band}"
                if expected_key in all_data:
                    expected_keys.append(expected_key)

        for key in all_data.keys():
            if key not in expected_keys:
                other_keys.append(key)

        sorted_keys = expected_keys + other_keys
        lfp_dict = {key: all_data[key] for key in sorted_keys}
        return dict(sorted(lfp_dict.items()))
