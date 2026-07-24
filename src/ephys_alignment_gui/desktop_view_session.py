"""Desktop view-session compatibility state.

This object owns the desktop-only state that is created/reset when loading a
new probe view. It is intentionally not part of the Qt-free workspace: it still
contains pyqtgraph item lifecycle, popup state, and compatibility projections
for older plot code.

Per-*shank* state does not live here directly. The session owns one
:class:`~ephys_alignment_gui.shank_alignment.ShankAlignment` per shank (see
:attr:`DesktopViewSession.shanks`) and exposes the *active* shank's fields through
:class:`_ShankAttr` descriptors, so switching shanks is just repointing
:attr:`active_shank`. Existing view/plot code can keep reading e.g.
``session.features[session.idx]`` unchanged; the read is transparently routed
to whichever shank is active. This keeps per-shank state single-sourced on the
shank (no separately-updated copies), which is the whole point of the refactor.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_alignment_gui.shank_alignment import ShankAlignment

logger = logging.getLogger(__name__)


class _ShankAttr:
    """Descriptor delegating an attribute to the session's active shank.

    Reading or writing ``session.<name>`` transparently reads/writes
    ``session.active_shank.<name>``. This keeps per-shank state physically on
    the :class:`ShankAlignment` while leaving the large body of view/plot code
    that refers to ``session.<name>`` untouched.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __set_name__(self, _owner: type, name: str) -> None:
        # Guard against typos: descriptor attribute name must match target.
        self._name = name

    def __get__(self, obj: Any, _objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return getattr(obj.active_shank, self._name)

    def __set__(self, obj: Any, value: Any) -> None:
        setattr(obj.active_shank, self._name, value)


class DesktopViewSession:
    """Owns desktop compatibility state for a single probe alignment view.

    Per-shank state is delegated to :attr:`active_shank`; the attributes below
    declared as :class:`_ShankAttr` live on the active :class:`ShankAlignment`,
    not on the session instance.
    """

    # -- Per-shank fields, delegated to the active ShankAlignment --
    # Fit / undo buffer
    idx = _ShankAttr("idx")
    current_idx = _ShankAttr("current_idx")
    total_idx = _ShankAttr("total_idx")
    last_idx = _ShankAttr("last_idx")
    diff_idx = _ShankAttr("diff_idx")
    idx_prev = _ShankAttr("idx_prev")
    max_idx = _ShankAttr("max_idx")
    track = _ShankAttr("track")
    features = _ShankAttr("features")
    lin_fit_history = _ShankAttr("lin_fit_history")
    active_alignment = _ShankAttr("active_alignment")
    # Track / channel locations
    chn_depths = _ShankAttr("chn_depths")
    track_annotations_ras = _ShankAttr("track_annotations_ras")
    track_annos_and_ends_ras = _ShankAttr("track_annos_and_ends_ras")
    # Selected starting alignment + engine + region overlays
    feature_prev = _ShankAttr("feature_prev")
    track_prev = _ShankAttr("track_prev")
    ephysalign = _ShankAttr("ephysalign")
    region_fp = _ShankAttr("region_fp")
    region_label_fp = _ShankAttr("region_label_fp")
    region_colour_fp = _ShankAttr("region_colour_fp")
    # Atlas/histology slice, cached per shank (see ShankAlignment.slice_data)
    slice_data = _ShankAttr("slice_data")
    fp_slice_data = _ShankAttr("fp_slice_data")

    _MAX_IDX = 10

    def __init__(self) -> None:
        # -- Shank container (must be set before any delegated attr access) --
        # ShankAlignment instances are created lazily on first access (see
        # ``active_shank``); the dict only holds shanks the user has visited.
        # ``_n_shanks`` bounds valid indices; ``init_shanks`` sets it once the
        # true shank count is known. Defaults describe a single-shank probe so
        # delegated attribute access is valid pre-load.
        self.shanks: dict[int, ShankAlignment] = {}
        self._n_shanks: int = 1
        self._active_shank_idx: int = 0

        # -- Probe geometry --
        self.probe_tip: int = 0
        self.probe_top: int = 3840
        self.probe_extra: int = 100
        self.view_total: list[int] = [-2000, 6000]
        self.depth: NDArray[np.signedinteger[Any]] = np.arange(
            self.view_total[0], self.view_total[1], 20
        )
        self.extend_feature: int = 1

        # -- Fit state (UI toggle; per-move history lives on the shank) --
        self.lin_fit: bool = True

        # -- UI toggle state --
        self.line_status: bool = True
        self.label_status: bool = True
        self.hist_bound_status: bool = True

        # -- Plot item caches --
        self.scale_regions: NDArray[Any] = np.empty((0, 1))
        self.hist_label_items: list[Any] = []
        self.hist_ref_label_items: list[Any] = []

        # -- Popups --
        self.cluster_popups: list[Any] = []
        self.label_popup: list[Any] = []
        self.popup_status: bool = True
        self.subj_win: Any = None

        # NOTE: fit history (track/features/lin_fit_history + idx cursors),
        # track/channel-location arrays, chn_depths, ephysalign, the selected
        # starting alignment (feature_prev/track_prev), region overlays, and
        # slice_data/fp_slice_data are per-shank; they live on the
        # active ShankAlignment and are reached via the _ShankAttr descriptors
        # declared at class scope.

        # -- Misc --
        self.nearby: Any = None

        # -- Per-probe track metadata --
        self.probe_path: Path | None = None
        self.sess_notes: str = ""

        # -- Large per-session objects (slice_data is per-shank) --
        self.ephys_stream: Any = None
        self.data: Any = None

        # -- Computed plot data --
        # Plot payloads resolve through plot_registry.py. ``img_raw_data`` is
        # not derived from ``plotdata`` in offline mode, so it stays a plain
        # attribute set explicitly by the view.
        self.img_raw_data: dict[str, Any] = {}

        # -- Plot items (per-session, have signal connections) --
        self.tip_pos: Any = None
        self.top_pos: Any = None
        self.hist_regions: Any = None
        self.hist_ref_regions: Any = None

        # -- Display state --
        self.scale_factor: Any = None
        self.selected_region: Any = None

        # -- Popup windows --
        self.clust_win: Any = None
        self.label_win: Any = None
        self.notes_win: Any = None
        self.nearby_win: Any = None
        self.nearby_table: Any = None
        self.region_win: Any = None

    # -- Shank management --

    def init_shanks(self, n_shanks: int) -> None:
        """Declare the probe's shank count and reset per-shank state.

        Called once the channel geometry has been read and the true shank count
        is known. Individual :class:`ShankAlignment` objects are created lazily
        on first access (see :attr:`active_shank`). Resets to shank 0 active.
        """
        self._n_shanks = max(1, int(n_shanks))
        self.shanks = {}
        self._active_shank_idx = 0

    @property
    def n_shanks(self) -> int:
        """Number of shanks on this probe (valid active-index range)."""
        return self._n_shanks

    def has_shank(self, idx: int) -> bool:
        """Whether ``idx`` is a valid shank index for this probe."""
        return 0 <= idx < self._n_shanks

    def _shank(self, idx: int) -> ShankAlignment:
        """Return the shank at ``idx``, creating it lazily on first access."""
        if not self.has_shank(idx):
            raise KeyError(
                f"Shank index {idx} out of range [0, {self._n_shanks}); "
                "call init_shanks() first."
            )
        shank = self.shanks.get(idx)
        if shank is None:
            shank = ShankAlignment(idx, max_idx=self._MAX_IDX)
            self.shanks[idx] = shank
        return shank

    @property
    def active_shank(self) -> ShankAlignment:
        """The shank whose state the delegated attributes currently expose."""
        return self._shank(self._active_shank_idx)

    @property
    def current_shank_idx(self) -> int:
        return self._active_shank_idx

    @current_shank_idx.setter
    def current_shank_idx(self, idx: int) -> None:
        if not self.has_shank(idx):
            raise KeyError(
                f"Shank index {idx} out of range [0, {self._n_shanks}); "
                "call init_shanks() first."
            )
        self._active_shank_idx = idx

    def detach(self, figures: dict[str, Any]) -> None:
        """Remove this session's plot items from the shared figures + disconnect
        its signals, WITHOUT nulling state or running gc.

        Used when swapping the displayed stream while keeping the outgoing
        session intact in the stream cache: it frees the shared figures for the
        incoming session but leaves this session's data/plotdata/shanks alive so
        a later switch-back is instant. :meth:`teardown` calls this then nulls.

        Parameters
        ----------
        figures : dict[str, Any]
            Map of figure names to pyqtgraph PlotItem / ViewBox widgets
            (e.g. ``{"img": fig_img, "line": fig_line, ...}``).
        """
        # -- Disconnect InfiniteLine signals (tip/top position markers) --
        for attr in ("tip_pos", "top_pos"):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    item.sigPositionChanged.disconnect()
                except TypeError:
                    pass

        for key in ("slice", "hist", "hist_ref", "hist_perp", "scale"):
            fig = figures.get(key)
            if fig is not None:
                fig.clear()

        # -- Disconnect and close popup windows --
        for popup_list in (self.cluster_popups, self.label_popup):
            for pop in popup_list:
                try:
                    pop.closed.disconnect()
                except (TypeError, AttributeError, RuntimeError):
                    pass
                try:
                    pop.moved.disconnect()
                except (TypeError, AttributeError, RuntimeError):
                    pass
                try:
                    pop.blockSignals(True)
                    pop.close()
                except RuntimeError:
                    pass

    def teardown(self, figures: dict[str, Any]) -> None:
        """Detach from figures, then null large references and gc.

        Use for a session that is being discarded (app reset, or evicting the
        stream cache) — NOT for a session that stays in the cache, which should
        only :meth:`detach`.
        """
        self.detach(figures)

        # -- Null large references (ephysalign/slice_data route to the
        # active shank via descriptors; this nulls the active shank's refs) --
        self.data = None
        if self.active_shank.runtime is not None:
            self.active_shank.runtime.plotdata = None
        self.ephysalign = None
        self.slice_data = None
        self.fp_slice_data = None

        # -- Force cycle collection --
        gc.collect()
