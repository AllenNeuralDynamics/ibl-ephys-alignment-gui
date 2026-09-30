"""Interference-line amplitude image and line plot payload builder.

Amplitudes are shown in dB relative to each line's median within each block,
so a line's level drifting between recordings does not show as a seam, and
blocks recording the same channels are averaged after that. The fitted curve
is referenced the same way, so it overlays the data it was fitted to.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit

from ephys_alignment_gui.io.brain_surface import (
    BrainSurface,
    CandidateLine,
    SurfaceCandidate,
)
from ephys_alignment_gui.plotting.array_utils import (
    average_equal_depth_channels,
    safe_take,
)
from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry
from ephys_alignment_gui.plotting.level_policy import in_brain_depth_mask


@dataclass(frozen=True)
class InterferenceLine:
    """One detected line and its amplitude row in each block."""

    index: int
    freq_hz: float
    rows: dict[int, int]

    @property
    def label(self) -> str:
        """Menu label and payload key; detected lines are hertz apart."""
        return f"{self.freq_hz:.1f} Hz"


def interference_lines(
    n_rows: int, surface: BrainSurface | None
) -> tuple[InterferenceLine, ...]:
    """Lines in ascending frequency, grouping each line's per-block rows.

    Without a lines table matching the amplitude rows there are none: which
    rows share a line, and which block each came from, cannot be recovered
    from the amplitudes.
    """
    if surface is None or len(surface.lines) != n_rows:
        return ()
    grouped: dict[int, InterferenceLine] = {}
    for row, entry in enumerate(surface.lines):
        line = grouped.setdefault(
            entry.line_index,
            InterferenceLine(entry.line_index, entry.freq_hz, {}),
        )
        line.rows[entry.block_index] = row
    return tuple(sorted(grouped.values(), key=lambda line: line.freq_hz))


def _logistic(y: np.ndarray, centre: float, below: float, above: float):
    """Rise from 0 deep to 1 shallow, as the producer's fit defines it."""
    d = y - centre
    scale = np.where(d < 0, below, above)
    with np.errstate(over="ignore"):
        smooth = expit(d / np.where(scale > 0, scale, 1.0))
    return np.where(scale > 0, smooth, (d > 0).astype(float))


class LineAmplitudePlotDataBuilder:
    """Build interference-line amplitude image and line payloads."""

    def __init__(self, data, geometry: PlotChannelGeometry, shank_idx: int) -> None:
        self.data = data
        self.geometry = geometry
        self.shank_idx = shank_idx
        self._profiles: dict[int, tuple[np.ndarray, dict[int, float]]] = {}

    @property
    def _amps(self) -> np.ndarray | None:
        entry = self.data.get("line_amp")
        if not entry or not entry.get("exists", False) or "amps" not in entry:
            return None
        return np.asarray(entry["amps"])

    @property
    def _surface(self) -> BrainSurface | None:
        return self.data.get("brain_surface")

    @property
    def _depths(self) -> np.ndarray:
        return np.unique(self.geometry.chn_coords[:, 1])

    def lines(self) -> tuple[InterferenceLine, ...]:
        """Lines measured on this shank, ascending in frequency."""
        amps = self._amps
        if amps is None:
            return ()
        return tuple(
            line
            for line in interference_lines(amps.shape[0], self._surface)
            if np.isfinite(self._profile(line)[0]).any()
        )

    def line_keys(self) -> tuple[str, ...]:
        """Line labels with data on this shank, without building payloads."""
        return tuple(line.label for line in self.lines())

    def candidates(self) -> tuple[SurfaceCandidate, ...]:
        """This shank's surface candidates, strongest first."""
        surface = self._surface
        return () if surface is None else surface.shank_candidates(self.shank_idx)

    def _profile(self, line: InterferenceLine) -> tuple[np.ndarray, dict[int, float]]:
        """The line's referenced dB at each unique depth of this shank, and
        the median each block was referenced to."""
        if line.index in self._profiles:
            return self._profiles[line.index]
        amps = self._amps
        assert amps is not None
        per_block = []
        medians: dict[int, float] = {}
        for block, row in line.rows.items():
            db = self._db(amps[row])
            if not np.isfinite(db).any():
                continue
            medians[block] = float(np.nanmedian(db))
            per_block.append(db - medians[block])
        if per_block:
            profile = _nanmean(
                average_equal_depth_channels(
                    np.vstack(per_block), self.geometry.chn_coords[:, 1]
                )
            )
        else:
            profile = np.full(self._depths.size, np.nan)
        self._profiles[line.index] = (profile, medians)
        return self._profiles[line.index]

    def _db(self, amps_row: np.ndarray) -> np.ndarray:
        """One amplitude row on this shank's channels in dB, NaN off it."""
        on_shank = np.asarray(safe_take(amps_row, self.geometry.chn_ind), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 20.0 * np.log10(on_shank)
        db[~np.isfinite(db)] = np.nan
        return db

    def build_image(self, in_brain_depths_um=None):
        """All lines as columns over depth, ascending in frequency."""
        lines = self.lines()
        if not lines:
            return None
        img = np.vstack([self._profile(line)[0] for line in lines])
        img_full = np.full((len(lines), self.geometry.chn_full.shape[0]), np.nan)
        img_full[:, self.geometry.idx_full] = img

        col = in_brain_depth_mask(self._depths, in_brain_depths_um)
        level_src = img if col is None else img[:, col]
        finite = np.abs(level_src[np.isfinite(level_src)])
        max_abs = float(np.quantile(finite, 0.95)) if finite.size else 1.0
        yscale = (self.geometry.chn_max - self.geometry.chn_min) / img_full.shape[1]
        return {
            "img": img_full,
            "scale": np.array([1.0, yscale]),
            "levels": np.array([-max_abs, max_abs]),
            "offset": np.array([0.0, self.geometry.chn_min]),
            "cmap": "RdBu_r",
            "xrange": np.array([0.0, float(len(lines))]),
            "xaxis": "Line (ascending frequency)",
            "title": "Line amplitude (dB re block median)",
            "line_keys": tuple(line.label for line in lines),
            "surface_candidates": self._markers(),
        }

    def build_lines(self) -> dict[str, dict[str, Any]]:
        """One depth profile per line, keyed by label, with its fitted curve
        when the line was fitted."""
        return {line.label: self._build_line(line) for line in self.lines()}

    def _build_line(self, line: InterferenceLine) -> dict[str, Any]:
        profile, medians = self._profile(line)
        depths = self._depths
        fit = self._fitted_curve(line, medians)
        values = profile if fit is None else np.r_[profile, fit]
        finite = values[np.isfinite(values)]
        lo, hi = (
            (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
        )
        pad = max(0.05 * (hi - lo), 0.5)
        return {
            "x": profile,
            "y": depths,
            "xrange": np.array([lo - pad, hi + pad]),
            "xaxis": f"{line.label} amplitude (dB re block median)",
            "fit": None if fit is None else {"x": fit, "y": depths},
            "surface_candidates": self._markers(line, fit),
        }

    def _line_fits(
        self, line: InterferenceLine, method: str
    ) -> Iterator[tuple[SurfaceCandidate, CandidateLine]]:
        """This line's part in each candidate of one fitted model."""
        for candidate in self.candidates():
            if candidate.method != method:
                continue
            for entry in candidate.lines:
                if entry.line_index == line.index:
                    yield candidate, entry

    def _fitted_curve(
        self, line: InterferenceLine, medians: dict[int, float]
    ) -> np.ndarray | None:
        """The ``lines`` model for this line, referenced like the data."""
        fits = list(self._line_fits(line, "lines"))
        if not fits:
            return None
        _, first = fits[0]
        depths = self._depths
        steps = sum(
            entry.step_db
            * _logistic(
                depths,
                candidate.midpoint_um,
                entry.width_below_um,
                entry.width_above_um,
            )
            for candidate, entry in fits
        )
        curves = []
        for block, median in medians.items():
            if block not in first.baseline_db:
                continue
            curve = (
                first.baseline_db[block]
                + first.trend_db_per_mm * depths / 1000.0
                + steps
                - median
            )
            curves.append(np.where(self._block_depths(line, block), curve, np.nan))
        if not curves:
            return None
        return _nanmean(np.vstack(curves))

    def _block_depths(self, line: InterferenceLine, block: int) -> np.ndarray:
        """Unique depths of this shank where the block recorded the line."""
        amps = self._amps
        assert amps is not None
        recorded = np.isfinite(self._db(amps[line.rows[block]]))
        return np.isin(self._depths, self.geometry.chn_coords[recorded, 1])

    def _markers(
        self,
        line: InterferenceLine | None = None,
        fit: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Candidates as the marker overlay draws them; for one line, only
        those it was fitted in, with its fitted level at the midpoint and the
        10-90 % extent of its step."""
        markers = []
        for candidate in self.candidates():
            marker: dict[str, Any] = {
                "midpoint_um": candidate.midpoint_um,
                "width_um": candidate.width_um,
                "method": candidate.method,
            }
            if line is not None:
                entry = next(
                    (
                        e
                        for c, e in self._line_fits(line, candidate.method)
                        if c is candidate
                    ),
                    None,
                )
                if entry is None:
                    continue
                marker["agrees"] = entry.agrees
                marker["extent_um"] = entry.extent_um(candidate.midpoint_um)
                marker["curve_db"] = _level_at(
                    self._depths, fit, candidate.midpoint_um
                )
            markers.append(marker)
        return tuple(markers)


def _level_at(
    depths: np.ndarray, curve: np.ndarray | None, depth: float
) -> float | None:
    """The curve interpolated at *depth* over its finite span."""
    if curve is None:
        return None
    finite = np.isfinite(curve)
    if not finite.any():
        return None
    return float(np.interp(depth, depths[finite], curve[finite]))


def _nanmean(values: np.ndarray) -> np.ndarray:
    """Column mean over finite entries, ``NaN`` where there are none."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    return np.divide(
        sums, counts, out=np.full(values.shape[1], np.nan), where=counts > 0
    )
