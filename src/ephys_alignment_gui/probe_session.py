"""Per-probe session state container.

All attributes that are created/reset when loading a new probe live here.
MainWindow delegates to ``self.session: ProbeSession | None``.

Per-*shank* state does not live here directly. Instead the session owns one
:class:`~ephys_alignment_gui.shank_alignment.ShankAlignment` per shank (see
:attr:`ProbeSession.shanks`) and exposes the *active* shank's fields through
:class:`_ShankAttr` descriptors, so switching shanks is just repointing
:attr:`active_shank`. Existing view/plot code can keep reading e.g.
``session.features[session.idx]`` unchanged; the read is transparently routed
to whichever shank is active.
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


class ProbeSession:
    """Owns all state for a single probe alignment session.

    Per-shank state is delegated to :attr:`active_shank`; the attributes below
    that are declared as :class:`_ShankAttr` descriptors live on the active
    :class:`ShankAlignment`, not on the session instance.
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
    # Track / channel locations
    chn_depths = _ShankAttr("chn_depths")
    track_annotations_ras = _ShankAttr("track_annotations_ras")
    track_annos_and_ends_ras = _ShankAttr("track_annos_and_ends_ras")
    channel_locations_ras = _ShankAttr("channel_locations_ras")
    tip_location_ras = _ShankAttr("tip_location_ras")
    # Selected starting alignment + engine + region overlays
    feature_prev = _ShankAttr("feature_prev")
    track_prev = _ShankAttr("track_prev")
    ephysalign = _ShankAttr("ephysalign")
    region_fp = _ShankAttr("region_fp")
    region_label_fp = _ShankAttr("region_label_fp")
    region_colour_fp = _ShankAttr("region_colour_fp")

    _MAX_IDX = 10

    def __init__(self) -> None:
        # -- Shank container (must be set before any delegated attr access) --
        # ShankAlignment instances are created lazily on first access (see
        # ``active_shank``): the dict only holds shanks the user has actually
        # visited. ``_n_shanks`` bounds the valid indices; ``init_shanks`` sets
        # it once the true shank count is known. Defaults describe a
        # single-shank probe so delegated attribute access is valid pre-load.
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
        self.channel_status: bool = True
        self.hist_bound_status: bool = True

        # -- Reference lines / points --
        self.lines_features: NDArray[Any] = np.empty((0, 3))
        self.lines_tracks: NDArray[Any] = np.empty((0, 1))
        self.points: NDArray[Any] = np.empty((0, 1))
        self.y_scale: float = 1
        self.x_scale: float = 1

        # -- Plot item caches --
        self.img_plots: list[Any] = []
        self.line_plots: list[Any] = []
        self.probe_plots: list[Any] = []
        self.img_cbars: list[Any] = []
        self.probe_cbars: list[Any] = []
        self.scale_regions: NDArray[Any] = np.empty((0, 1))
        self.slice_lines: list[Any] = []
        self.slice_items: list[Any] = []
        self.slice_chns: list[Any] = []
        self.slice_tip: Any = None
        self.probe_bounds: list[Any] = []

        # -- Popups --
        self.cluster_popups: list[Any] = []
        self.label_popup: list[Any] = []
        self.popup_status: bool = True
        self.subj_win: Any = None

        # -- Histology dicts --
        self.hist_data: dict[str, list[Any]] = {
            "region": [],
            "axis_label": [],
            "colour": [],
        }
        self.hist_data_ref: dict[str, list[Any]] = {
            "region": [],
            "axis_label": [],
            "colour": [],
        }
        self.scale_data: dict[str, list[Any]] = {"region": [], "scale": []}

        # -- Nearby boundary state --
        self.hist_nearby_x: Any = None
        self.hist_nearby_y: Any = None
        self.hist_nearby_col: Any = None
        self.hist_nearby_parent_x: Any = None
        self.hist_nearby_parent_y: Any = None
        self.hist_nearby_parent_col: Any = None
        self.hist_mapping: str = "Allen"

        # NOTE: fit history (track/features/lin_fit_history + idx cursors),
        # track/channel-location arrays, chn_depths, ephysalign, the selected
        # starting alignment (feature_prev/track_prev) and region overlays are
        # per-shank; they live on the active ShankAlignment and are reached via
        # the _ShankAttr descriptors declared at class scope.

        # -- Misc --
        self.nearby: Any = None

        # -- Per-probe track metadata --
        self.probe_path: Path | None = None
        self.sess_notes: str = ""

        # -- Large per-session objects --
        self.data: Any = None
        self.plotdata: Any = None
        self.slice_data: Any = None
        self.fp_slice_data: Any = None

        # -- Computed plot data --
        self.img_fr_data: Any = None
        self.img_spike_corr_data: Any = None
        self.img_rms_APdata: Any = None
        self.img_rms_LFPdata: Any = None
        self.img_rms_APdata_main: Any = None
        self.img_rms_LFPdata_main: Any = None
        self.img_lfp_data: Any = None
        self.img_lfp_data_main: Any = None
        self.img_lfp_corr_data: Any = None
        self.img_stim_data: Any = None
        self.img_raw_data: dict[str, Any] = {}
        self.line_fr_data: Any = None
        self.line_amp_data: Any = None
        self.probe_rms_APdata: Any = None
        self.probe_rms_LFPdata: Any = None
        self.probe_rms_APdata_main: Any = None
        self.probe_rms_LFPdata_main: Any = None
        self.probe_lfp_data: Any = None
        self.probe_lfp_data_main: Any = None
        self.probe_rfmap: Any = None
        self.rfmap_boundaries: Any = None
        self.scat_drift_data: Any = None
        self.scat_fr_data: Any = None
        self.scat_p2t_data: Any = None
        self.scat_amp_data: Any = None

        # -- Plot items (per-session, have signal connections) --
        self.tip_pos: Any = None
        self.top_pos: Any = None
        self.traj_line: Any = None
        self.data_plot: Any = None
        self.hist_regions: Any = None
        self.hist_ref_regions: Any = None

        # -- Display state --
        self.xrange: Any = None
        self.scale_factor: Any = None
        self.selected_line: Any = []
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

        Called once the channel geometry has been read and the true shank
        count is known. Individual :class:`ShankAlignment` objects are *not*
        built here; they are created lazily on first access (see
        :attr:`active_shank`). Resets to shank 0 as the active shank.
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

    def teardown(self, figures: dict[str, Any]) -> None:
        """Disconnect signals, remove plot items from figures, null references.

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

        # -- Disconnect signals on user-drawn reference lines --
        for arr in (self.lines_features, self.lines_tracks):
            for group in arr:
                for item in group if hasattr(group, "__iter__") else [group]:
                    try:
                        item.sigPositionChanged.disconnect()
                    except (TypeError, AttributeError, RuntimeError):
                        pass

        # -- Disconnect scatter click signal --
        data_plot = getattr(self, "data_plot", None)
        if data_plot is not None:
            try:
                data_plot.sigClicked.disconnect()
            except (TypeError, AttributeError):
                pass

        # -- Remove plot items from figures --
        if "img" in figures:
            for plot in self.img_plots:
                figures["img"].removeItem(plot)
            for cbar in self.img_cbars:
                figures["img"].removeItem(cbar)
        if "line" in figures:
            for plot in self.line_plots:
                figures["line"].removeItem(plot)
        if "probe" in figures:
            for plot in self.probe_plots:
                figures["probe"].removeItem(plot)
            for cbar in self.probe_cbars:
                figures["probe"].removeItem(cbar)
        for key in ("slice", "hist", "hist_ref", "scale"):
            fig = figures.get(key)
            if fig is not None:
                fig.clear()

        # -- Remove user-drawn reference lines and fit points from figures --
        for line_feature, line_track, point in zip(
            self.lines_features, self.lines_tracks, self.points
        ):
            if "img" in figures:
                figures["img"].removeItem(line_feature[0])
            if "line" in figures:
                figures["line"].removeItem(line_feature[1])
            if "probe" in figures:
                figures["probe"].removeItem(line_feature[2])
            if "hist" in figures:
                figures["hist"].removeItem(line_track[0])
            if "fit" in figures:
                figures["fit"].removeItem(point[0])

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

        # -- Null large references --
        self.data = None
        self.plotdata = None
        self.ephysalign = None
        self.slice_data = None
        self.fp_slice_data = None

        # -- Force cycle collection --
        gc.collect()
