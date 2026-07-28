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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_edit_history import AlignmentEditHistory
from ephys_alignment_gui.alignment_state import AlignmentState
from ephys_alignment_gui.shank_runtime import ShankRuntime

logger = logging.getLogger(__name__)


def _edit_history_attr(name: str) -> property:
    """Delegate a compatibility attribute to ``edit_history``."""

    def _get(self: Any) -> Any:
        return getattr(self.edit_history, name)

    def _set(self: Any, value: Any) -> None:
        setattr(self.edit_history, name, value)

    return property(_get, _set)


def _runtime_attr(name: str) -> property:
    """Delegate a compatibility runtime attribute to ``runtime`` when attached."""

    fallback_name = f"_{name}"

    def _get(self: Any) -> Any:
        if self.runtime is not None:
            return getattr(self.runtime, name)
        return getattr(self, fallback_name)

    def _set(self: Any, value: Any) -> None:
        if self.runtime is not None:
            setattr(self.runtime, name, value)
            return
        setattr(self, fallback_name, value)

    return property(_get, _set)


class ShankAlignment:
    """Compatibility holder for per-shank alignment and derived runtime state.

    Pure editable state is owned by :class:`AlignmentState`. Runtime-derived
    compatibility fields still live here until the GUI/view split can consume
    them from runtime services directly.

    Parameters
    ----------
    shank_idx : int
        Zero-based index of this shank within the probe.
    max_idx : int
        Size of the transient cyclic fit/undo buffer.
    """

    # Compatibility accessors for the edit buffer. The storage lives on
    # AlignmentEditHistory; callers can keep using ``shank.idx`` etc while the
    # broader desktop view-session split proceeds.
    idx = _edit_history_attr("idx")
    current_idx = _edit_history_attr("current_idx")
    total_idx = _edit_history_attr("total_idx")
    last_idx = _edit_history_attr("last_idx")
    diff_idx = _edit_history_attr("diff_idx")
    idx_prev = _edit_history_attr("idx_prev")
    max_idx = _edit_history_attr("max_idx")
    track = _edit_history_attr("track")
    features = _edit_history_attr("features")
    lin_fit_history = _edit_history_attr("lin_fit_history")

    # Compatibility accessors for runtime-owned state. New code should prefer
    # ShankRuntime directly; these keep legacy plotting call sites working.
    chn_coords = _runtime_attr("chn_coords")
    chn_depths = _runtime_attr("chn_depths")
    track_annotations_ras = _runtime_attr("track_annotations_ras")
    track_annos_and_ends_ras = _runtime_attr("track_annos_and_ends_ras")
    channel_locations_ras = _runtime_attr("channel_locations_ras")
    tip_location_ras = _runtime_attr("tip_location_ras")
    ephysalign = _runtime_attr("ephysalign")
    region_fp = _runtime_attr("region_fp")
    region_label_fp = _runtime_attr("region_label_fp")
    region_colour_fp = _runtime_attr("region_colour_fp")

    @property
    def active_alignment(self) -> ActiveAlignment | None:
        """Current feature/track control points for this shank."""
        return self.alignment_state.active_alignment

    @active_alignment.setter
    def active_alignment(self, alignment: ActiveAlignment | None) -> None:
        self.alignment_state.active_alignment = alignment

    @property
    def edit_history(self) -> AlignmentEditHistory:
        """Transient edit buffer for the editable alignment state."""
        return self.alignment_state.edit_history

    @property
    def alignments(self) -> dict[str, list[list[float]]]:
        """Saved alignment history, delegated to AlignmentState."""
        return self.alignment_state.alignments

    @alignments.setter
    def alignments(self, alignments: dict[str, list[list[float]]]) -> None:
        self.alignment_state.set_alignments(alignments)

    @property
    def prev_align(self) -> list[str]:
        """Dropdown-ordered alignment keys, delegated to AlignmentState."""
        return self.alignment_state.prev_align

    @prev_align.setter
    def prev_align(self, prev_align: list[str]) -> None:
        self.alignment_state.prev_align = prev_align

    @property
    def feature_prev(self) -> Any:
        """Currently selected starting feature alignment."""
        return self.alignment_state.feature_prev

    @feature_prev.setter
    def feature_prev(self, feature_prev: Any) -> None:
        self.alignment_state.feature_prev = feature_prev

    @property
    def track_prev(self) -> Any:
        """Currently selected starting track alignment."""
        return self.alignment_state.track_prev

    @track_prev.setter
    def track_prev(self, track_prev: Any) -> None:
        self.alignment_state.track_prev = track_prev

    def __init__(self, shank_idx: int, max_idx: int = 10) -> None:
        self.shank_idx: int = shank_idx
        self.alignment_state = AlignmentState(max_idx=max_idx)
        self.runtime: ShankRuntime | None = None

        # -- Runtime fallback fields before a ShankRuntime is attached --
        self._chn_coords: NDArray[Any] | None = None
        self._chn_depths: NDArray[np.floating[Any]] | None = None

        self._track_annotations_ras: NDArray[np.floating[Any]] | None = None
        self._track_annos_and_ends_ras: NDArray[np.floating[Any]] | None = None
        self._channel_locations_ras: NDArray[np.floating[Any]] | None = None
        self._tip_location_ras: NDArray[np.floating[Any]] | None = None

        self._ephysalign: Any = None
        self._region_fp: Any = None
        self._region_label_fp: Any = None
        self._region_colour_fp: Any = None

        # -- Output dicts produced on save --
        self.channel_dict: dict[str, dict[str, Any]] = {}
        self.ccf_channel_dict: dict[str, dict[str, Any]] = {}

    def attach_runtime(self, runtime: ShankRuntime) -> None:
        """Attach runtime-owned shank data for compatibility accessors."""
        if runtime.shank_idx != self.shank_idx:
            raise ValueError(
                f"Cannot attach runtime for shank {runtime.shank_idx} "
                f"to shank {self.shank_idx}"
            )
        self.runtime = runtime

    # -- Alignment history helpers --

    def set_alignments(self, alignments: dict[str, list[list[float]]]) -> None:
        """Replace this shank's alignment history and rebuild the dropdown list."""
        self.alignment_state.set_alignments(alignments)

    def add_alignment(self, feature: NDArray, track: NDArray) -> str:
        """Record a new alignment, keyed by the current timestamp.

        Returns the key used, and refreshes :attr:`prev_align`. The key keeps
        the historical second-resolution ISO format, but if a save already
        exists for the current second a ``.N`` disambiguator is appended so a
        rapid second save cannot silently overwrite the first.
        """
        return self.alignment_state.add_alignment(feature, track)

    def get_alignment_idx(self, idx: int) -> tuple[NDArray | None, NDArray | None]:
        """Return the ``(feature, track)`` for the alignment at dropdown ``idx``.

        ``("original")`` and out-of-range indices yield ``(None, None)``.
        """
        return self.alignment_state.get_alignment_idx(idx)
