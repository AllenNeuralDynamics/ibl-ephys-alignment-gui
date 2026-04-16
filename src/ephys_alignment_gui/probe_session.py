"""Per-probe session state container.

All attributes that are created/reset when loading a new probe live here.
MainWindow delegates to ``self.session: ProbeSession | None``.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ProbeSession:
    """Owns all state for a single probe alignment session."""

    def __init__(self) -> None:
        # -- Probe geometry --
        self.probe_tip: int = 0
        self.probe_top: int = 3840
        self.probe_extra: int = 100
        self.view_total: list[int] = [-2000, 6000]
        self.depth: NDArray[np.signedinteger[Any]] = np.arange(
            self.view_total[0], self.view_total[1], 20
        )
        self.extend_feature: int = 1

        # -- Fit state --
        self.lin_fit: bool = True
        self.max_idx: int = 10
        self.idx: int = 0
        self.current_idx: int = 0
        self.total_idx: int = 0
        self.last_idx: int = 0
        self.diff_idx: int = 0

        # -- UI toggle state --
        self.line_status: bool = True
        self.label_status: bool = True
        self.channel_status: bool = True
        self.hist_bound_status: bool = True

        # -- Reference lines / points --
        self.lines_features: NDArray[Any] = np.empty((0, 4))
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
        self.hist_label_items: list[Any] = []
        self.hist_ref_label_items: list[Any] = []

        # -- Perpendicular slice plot items --
        self.perp_image_item: Any = None
        self.perp_probe_line: Any = None
        self.perp_channel_dots: Any = None
        self.perp_tip_marker: Any = None
        self.slice_color_bar: Any = None
        self.slice_hist_levels: Any = None

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

        # -- Fit history --
        self.track: list[Any] = [0] * (self.max_idx + 1)
        self.features: list[Any] = [0] * (self.max_idx + 1)
        self.lin_fit_history: list[bool] = [True] * (self.max_idx + 1)

        # -- Misc --
        self.nearby: Any = None

        # -- Track / alignment arrays --
        self.track_annotations_ras: NDArray[np.floating[Any]] | None = None
        self.track_annos_and_ends_ras: NDArray[np.floating[Any]] | None = None
        self.channel_locations_ras: NDArray[np.floating[Any]] | None = None
        self.tip_location_ras: NDArray[np.floating[Any]] | None = None
        self.probe_path: Path | None = None
        self.chn_depths: NDArray[np.floating[Any]] | None = None
        self.sess_notes: str = ""

        # -- Large per-session objects --
        self.data: Any = None
        self.plotdata: Any = None
        self.ephysalign: Any = None
        self.slice_data: Any = None
        self.fp_slice_data: Any = None

        # -- Alignment state --
        self.feature_prev: Any = None
        self.track_prev: Any = None
        self.region_fp: Any = None
        self.region_label_fp: Any = None
        self.region_colour_fp: Any = None
        self.idx_prev: int = 0

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

        # -- Shank --
        self.current_shank_idx: int = 0

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
        for key in ("slice", "hist", "hist_ref", "hist_perp", "scale"):
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
            if "hist_perp" in figures and len(line_feature) > 3:
                figures["hist_perp"].removeItem(line_feature[3])
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
