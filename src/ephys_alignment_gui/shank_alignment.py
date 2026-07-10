"""Per-shank alignment state.

A single probe may carry several shanks (e.g. Neuropixels 2.0 4-shank). Each
shank is aligned independently and produces its own output files. This module
gives that independence a first-class home: one :class:`ShankAlignment` per
shank owns *all* state that varies from shank to shank, so switching shanks is a
matter of swapping which instance is active rather than re-filtering shared
mutable slots.

Keeping this state per-shank makes three former bug classes impossible by
construction:

* alignment histories can no longer cross-contaminate between shanks,
* the fit / undo buffer can no longer bleed from one shank into another,
* a freshly-selected shank can no longer inherit another shank's starting
  alignment.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ShankAlignment:
    """Owns every piece of state that belongs to a single shank.

    Parameters
    ----------
    shank_idx : int
        Zero-based index of this shank within the probe.
    max_idx : int
        Size of the cyclic fit/undo buffer (number of retained moves).
    """

    def __init__(self, shank_idx: int, max_idx: int = 10) -> None:
        self.shank_idx: int = shank_idx

        # -- Channel geometry for this shank --
        self.chn_coords: NDArray[Any] | None = None
        self.chn_depths: NDArray[np.floating[Any]] | None = None

        # -- Track / channel locations in atlas (RAS) space --
        self.track_annotations_ras: NDArray[np.floating[Any]] | None = None
        self.track_annos_and_ends_ras: NDArray[np.floating[Any]] | None = None
        self.channel_locations_ras: NDArray[np.floating[Any]] | None = None
        self.tip_location_ras: NDArray[np.floating[Any]] | None = None

        # -- Persistent alignment history (formerly on LoadDataLocal) --
        # Maps an ISO-timestamp key to ``[feature, track]`` control points.
        self.alignments: dict[str, list[list[float]]] = {}
        # Dropdown-ordered keys, newest first, with "original" appended.
        self.prev_align: list[str] = ["original"]

        # -- Cyclic fit / undo buffer (per shank) --
        self.max_idx: int = max_idx
        self.idx: int = 0
        self.current_idx: int = 0
        self.total_idx: int = 0
        self.last_idx: int = 0
        self.diff_idx: int = 0
        self.idx_prev: int = 0
        self.track: list[Any] = [0] * (max_idx + 1)
        self.features: list[Any] = [0] * (max_idx + 1)
        self.lin_fit_history: list[bool] = [True] * (max_idx + 1)

        # -- Currently-selected starting alignment --
        self.feature_prev: Any = None
        self.track_prev: Any = None

        # -- Alignment engine + derived region overlays for this shank --
        self.ephysalign: Any = None
        self.region_fp: Any = None
        self.region_label_fp: Any = None
        self.region_colour_fp: Any = None

        # -- Cached PlotData for this shank --
        # Built once per shank (filters the probe-wide spikes/channels to this
        # shank and memoizes its plot datasets); reused on revisit. Shares the
        # probe-wide ``data`` dict by reference, so this is cheap to keep.
        self.plotdata: Any = None

        # -- Cached atlas/histology slice for this shank --
        # Computed by get_slice_images() along the probe track; cached here so
        # revisiting a shank is instant. Keyed on the track it was built from
        # (_slice_track) so it recomputes if this shank's alignment changes.
        self.slice_data: Any = None
        self.fp_slice_data: Any = None
        self._slice_track: NDArray[np.floating[Any]] | None = None

        # -- Output dicts produced on save --
        self.channel_dict: dict[str, dict[str, Any]] = {}
        self.ccf_channel_dict: dict[str, dict[str, Any]] = {}

    # -- Alignment history helpers --

    def set_alignments(self, alignments: dict[str, list[list[float]]]) -> None:
        """Replace this shank's alignment history and rebuild the dropdown list."""
        self.alignments = alignments
        self.prev_align = self._ordered_keys(alignments)

    def add_alignment(self, feature: NDArray, track: NDArray) -> str:
        """Record a new alignment, keyed by the current timestamp.

        Returns the key used, and refreshes :attr:`prev_align`. The key keeps
        the historical second-resolution ISO format, but if a save already
        exists for the current second a ``.N`` disambiguator is appended so a
        rapid second save cannot silently overwrite the first.
        """
        base = datetime.now().replace(microsecond=0).isoformat()
        date = base
        n = 1
        while date in self.alignments:
            date = f"{base}.{n}"
            n += 1
        self.alignments[date] = [feature.tolist(), track.tolist()]
        self.prev_align = self._ordered_keys(self.alignments)
        return date

    def get_alignment_idx(self, idx: int) -> tuple[NDArray | None, NDArray | None]:
        """Return the ``(feature, track)`` for the alignment at dropdown ``idx``.

        ``("original")`` and out-of-range indices yield ``(None, None)``.
        """
        if len(self.prev_align) <= idx:
            return None, None
        alignment = self.prev_align[idx]
        if alignment == "original":
            return None, None
        feature = np.array(self.alignments[alignment][0])
        track = np.array(self.alignments[alignment][1])
        return feature, track

    @staticmethod
    def _ordered_keys(alignments: dict[str, Any]) -> list[str]:
        prev_align = sorted(alignments.keys(), reverse=True)
        prev_align.append("original")
        return prev_align

    # -- Cached atlas/histology slice --

    def cached_slice(self, track: NDArray) -> tuple[Any, Any] | None:
        """Return this shank's cached ``(slice_data, fp_slice_data)``.

        Returns ``None`` if nothing is cached yet, or if ``track`` differs from
        the track the cache was built for (i.e. the shank was re-aligned).
        """
        if self._slice_track is not None and np.array_equal(self._slice_track, track):
            return self.slice_data, self.fp_slice_data
        return None

    def set_slice(self, slice_data: Any, fp_slice_data: Any, track: NDArray) -> None:
        """Cache the slice built for ``track`` on this shank."""
        self.slice_data = slice_data
        self.fp_slice_data = fp_slice_data
        self._slice_track = track
