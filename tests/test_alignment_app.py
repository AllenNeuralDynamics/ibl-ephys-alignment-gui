"""Tests for the UI-facing alignment app port."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.active_alignment import ActiveAlignment
from ephys_alignment_gui.alignment_derived_data_service import (
    AlignmentHistologyData,
    HistologyPlotData,
    ScaleFactorData,
)
from ephys_alignment_gui.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.alignment_events import AlignmentEdited, ShankChanged
from ephys_alignment_gui.alignment_repository import LoadedAlignmentHistory
from ephys_alignment_gui.app import AlignmentQueries
from ephys_alignment_gui.controller import (
    AlignmentChoicesUpdated,
    AlignmentEditApplied,
    NoPreviousAlignments,
    PreviousAlignmentSelected,
    ShankSelected,
)
from ephys_alignment_gui.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.slice_display_policy import SliceImageKind, SliceSelection
from ephys_alignment_gui.slice_runtime import SliceRuntime
from ephys_alignment_gui.workspace import AlignmentWorkspace


class FakePlotData:
    def __init__(self, label: str = "plot") -> None:
        self.label = label
        self.data = {
            "spikes": {"exists": True},
            "clusters": {"exists": False},
            "rms_AP": {"exists": False},
            "rms_AP_main": {"exists": False},
            "rms_LF": {"exists": False},
            "rms_LF_main": {"exists": False},
            "psd_lf": {"exists": False},
            "psd_lf_main": {"exists": False},
        }

    def cached(self, method: str, args: tuple = ()) -> Any:
        if method == "get_fr_img":
            return {"label": self.label}
        if method == "get_lfp_correlation_data_img":
            return {}
        if method == "get_passive_events":
            return {}
        if method == "get_lfp_spectrum_data":
            return None, {}
        if method == "get_rfmap_data":
            return {}, None
        return None


class FakeStreamRuntime:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.shank_runtime_by_idx = {}
        self.plotdata_by_shank = {
            1: FakePlotData("shank-1"),
            2: FakePlotData("shank-2"),
        }

    def plot_data_for_shank(self, shank_idx: int) -> FakePlotData:
        self.calls.append(shank_idx)
        return self.plotdata_by_shank[shank_idx]


class FakeAlignmentRepository:
    def __init__(self) -> None:
        self.loaded_alignments = None
        self.loaded_kwargs = None

    def load_previous_alignments(self, **kwargs):
        self.loaded_kwargs = kwargs
        if self.loaded_alignments is None:
            return None
        return LoadedAlignmentHistory(self.loaded_alignments)


class FakeDerivedDataService:
    def __init__(
        self,
        *,
        histology: Any = "histology",
        projection: Any = "projection",
    ) -> None:
        self.histology = histology
        self.projection = projection
        self.histology_kwargs = None
        self.projection_kwargs = None

    def compute_histology(self, **kwargs):
        self.histology_kwargs = kwargs
        return self.histology

    def compute_channel_projection(self, **kwargs):
        self.projection_kwargs = kwargs
        return self.projection


class FakeSliceService:
    def __init__(self) -> None:
        self.slice_set_calls: list[dict[str, Any]] = []
        self.perpendicular_calls: list[dict[str, Any]] = []

    def build_slice_set(self, **kwargs):
        self.slice_set_calls.append(kwargs)
        return {"ccf": np.array([[1.0]]), "scale": [1.0, 1.0], "offset": [0.0, 0.0]}

    def build_perpendicular_slice_image(self, **kwargs):
        self.perpendicular_calls.append(kwargs)
        n_perp_samples = kwargs["n_perp_samples"]
        n_depths = len(kwargs["feature_grid_m"])
        return np.ones((n_perp_samples, n_depths))


class FakeBrainAtlas:
    def __init__(self, dv_voxel_m: float = 20e-6) -> None:
        self.bc = SimpleNamespace(dxyz=[20e-6, 20e-6, dv_voxel_m])


class FakeFitAligner:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, Any]] = []

    def feature2track_lin(self, depths, feature, track):
        self.calls.append((depths, feature, track))
        return depths + 0.001


def _workspace_with_probe_state(
    *,
    shank_idx: int = 1,
    repo: FakeAlignmentRepository | None = None,
) -> AlignmentWorkspace:
    workspace = AlignmentWorkspace()
    if repo is not None:
        workspace.controller.alignment_repository = repo
    workspace.data_context.probe_info = SimpleNamespace(
        recording_id="rec",
        probe_name="probeA",
        ephys_collection="stream",
    )
    workspace.data_context.channel_table = SimpleNamespace(n_shanks=2)
    workspace.document.select_alignment_key(AlignmentKey("rec", "stream", shank_idx))
    return workspace


def _queries_with_cached_slice(
    *,
    slice_data: dict[str, Any],
    fp_slice_data: dict[str, Any] | None = None,
    derived: FakeDerivedDataService | None = None,
) -> tuple[AlignmentQueries, AlignmentKey, Any]:
    document = AlignmentDocument()
    key = AlignmentKey("rec", "stream", 1)
    state = document.select_alignment_key(key)
    active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    state.active_alignment = active_alignment
    track = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    slice_runtime = SliceRuntime()
    slice_runtime.set_coronal_slice(
        alignment_key=key,
        track_interpolation_ras=track,
        slice_data=slice_data,
        fp_slice_data=fp_slice_data,
    )
    ephysalign = SimpleNamespace(track_interpolation_ras=track)
    shank_runtime = SimpleNamespace(
        ephysalign=ephysalign,
        slice_runtime=slice_runtime,
        track_annos_and_ends_ras=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]]
        ),
    )
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=SimpleNamespace(
                shank_runtime_by_idx={1: shank_runtime}
            )
        ),
        derived_data_service=derived or FakeDerivedDataService(),
    )
    return queries, key, ephysalign


def test_workspace_exposes_app_port() -> None:
    workspace = AlignmentWorkspace()

    assert workspace.app.events is workspace.events
    assert workspace.app.queries.document is workspace.document
    assert workspace.app.queries.runtime is workspace.runtime


def test_commands_select_shank_updates_document_and_emits_event() -> None:
    workspace = AlignmentWorkspace()
    key0 = AlignmentKey("rec", "stream", 0)
    key1 = AlignmentKey("rec", "stream", 1)
    workspace.document.select_alignment_key(key0)
    events: list[ShankChanged] = []
    workspace.app.events.subscribe(ShankChanged, events.append)

    result = workspace.app.commands.select_shank(1, source="test")

    assert isinstance(result, ShankSelected)
    assert workspace.document.selected_alignment_key == key1
    assert len(events) == 1
    event = events[0]
    assert event.source == "test"
    assert event.previous_shank_idx == 0
    assert event.shank_idx == 1
    assert event.previous_key == key0
    assert event.active_key == key1
    assert not event.data_loaded


def test_commands_select_shank_captures_outgoing_reference_lines() -> None:
    workspace = AlignmentWorkspace()
    key0 = AlignmentKey("rec", "stream", 0)
    key1 = AlignmentKey("rec", "stream", 1)
    workspace.document.select_alignment_key(key0)
    workspace.document.mark_data_loaded(True)
    events: list[ShankChanged] = []
    workspace.app.events.subscribe(ShankChanged, events.append)

    result = workspace.app.commands.select_shank(
        1,
        outgoing_reference_lines=([10.0, 20.0], [11.0, 21.0]),
        source="test",
    )

    assert isinstance(result, ShankSelected)
    assert workspace.document.selected_alignment_key == key1
    pending = workspace.document.alignment_state_for(key0).pending_reference_lines
    assert pending is not None
    np.testing.assert_allclose(pending.feature_positions_um, [10.0, 20.0])
    np.testing.assert_allclose(pending.track_positions_um, [11.0, 21.0])
    assert workspace.document.alignment_state_for(key1).pending_reference_lines is None
    assert events[0].data_loaded


def test_commands_select_shank_clears_missing_outgoing_reference_lines() -> None:
    workspace = AlignmentWorkspace()
    key0 = AlignmentKey("rec", "stream", 0)
    workspace.document.select_alignment_key(key0)
    workspace.document.mark_data_loaded(True)
    workspace.document.active_set_pending_reference_lines([1.0], [2.0])

    result = workspace.app.commands.select_shank(1, outgoing_reference_lines=None)

    assert isinstance(result, ShankSelected)
    assert workspace.document.alignment_state_for(key0).pending_reference_lines is None


def test_commands_select_shank_without_line_state_leaves_pending_lines() -> None:
    workspace = AlignmentWorkspace()
    key0 = AlignmentKey("rec", "stream", 0)
    workspace.document.select_alignment_key(key0)
    workspace.document.mark_data_loaded(True)
    workspace.document.active_set_pending_reference_lines([1.0], [2.0])

    result = workspace.app.commands.select_shank(1)

    assert isinstance(result, ShankSelected)
    pending = workspace.document.alignment_state_for(key0).pending_reference_lines
    assert pending is not None
    np.testing.assert_allclose(pending.feature_positions_um, [1.0])
    np.testing.assert_allclose(pending.track_positions_um, [2.0])


def test_commands_load_previous_alignments_defaults_to_active_shank(tmp_path) -> None:
    repo = FakeAlignmentRepository()
    repo.loaded_alignments = {
        "auto": [[100.0], [200.0]],
        "saved": [[1.0], [2.0]],
    }
    workspace = _workspace_with_probe_state(shank_idx=1, repo=repo)

    result = workspace.app.commands.load_previous_alignments(
        folder=tmp_path,
        use_docdb=True,
    )

    assert isinstance(result, AlignmentChoicesUpdated)
    assert result.choices == ["saved", "original"]
    assert repo.loaded_kwargs["shank_idx"] == 1
    state = workspace.document.alignment_state_for(AlignmentKey("rec", "stream", 1))
    assert state.alignments == {"saved": [[1.0], [2.0]]}


def test_commands_load_previous_alignments_reports_missing_history(tmp_path) -> None:
    repo = FakeAlignmentRepository()
    workspace = _workspace_with_probe_state(repo=repo)

    result = workspace.app.commands.load_previous_alignments(
        folder=tmp_path,
        use_docdb=False,
    )

    assert isinstance(result, NoPreviousAlignments)


def test_commands_select_previous_alignment_defaults_to_active_shank() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    active_key = AlignmentKey("rec", "stream", 1)
    other_key = AlignmentKey("rec", "stream", 0)
    workspace.document.alignment_state_for(other_key).set_alignments(
        {"other": [[9.0], [10.0]]}
    )
    workspace.document.alignment_state_for(active_key).set_alignments(
        {"saved": [[1.0, 2.0], [3.0, 4.0]]}
    )

    result = workspace.app.commands.select_previous_alignment(0)

    assert isinstance(result, PreviousAlignmentSelected)
    assert result.choice == "saved"
    np.testing.assert_allclose(result.feature_prev, [1.0, 2.0])
    np.testing.assert_allclose(result.track_prev, [3.0, 4.0])
    active_state = workspace.document.alignment_state_for(active_key)
    np.testing.assert_allclose(active_state.feature_prev, [1.0, 2.0])
    assert workspace.document.alignment_state_for(other_key).feature_prev is None


def test_commands_offset_alignment_defaults_to_active_shank() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    active_key = AlignmentKey("rec", "stream", 1)
    state = workspace.document.alignment_state_for(active_key)
    events: list[AlignmentEdited] = []
    workspace.app.events.subscribe(AlignmentEdited, events.append)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
        lin_fit=True,
    )

    result = workspace.app.commands.offset_alignment_from_tip(
        tip_position_um=100.0,
        probe_tip_um=0.0,
        lin_fit=False,
    )

    assert isinstance(result, AlignmentEditApplied)
    np.testing.assert_allclose(result.alignment.track, [10.0001, 14.0001])
    assert result.lin_fit is False
    np.testing.assert_allclose(state.active_alignment.track, [10.0001, 14.0001])
    assert len(events) == 1
    assert events[0].edit_kind == "offset"
    assert events[0].active_key == active_key
    assert events[0].active_alignment == result.alignment
    assert events[0].lin_fit is False


def test_commands_previous_next_alignment_default_to_active_shank() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    active_key = AlignmentKey("rec", "stream", 1)
    state = workspace.document.alignment_state_for(active_key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
        lin_fit=True,
    )
    workspace.app.commands.offset_alignment_from_tip(
        tip_position_um=100.0,
        probe_tip_um=0.0,
        lin_fit=False,
    )

    previous_result = workspace.app.commands.go_previous_alignment()
    next_result = workspace.app.commands.go_next_alignment()

    assert isinstance(previous_result, AlignmentEditApplied)
    np.testing.assert_allclose(previous_result.alignment.track, [10.0, 14.0])
    assert previous_result.lin_fit is True
    assert isinstance(next_result, AlignmentEditApplied)
    np.testing.assert_allclose(next_result.alignment.track, [10.0001, 14.0001])
    assert next_result.lin_fit is False


def test_commands_do_not_emit_alignment_event_for_noop_edit() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    events: list[AlignmentEdited] = []
    workspace.app.events.subscribe(AlignmentEdited, events.append)

    result = workspace.app.commands.go_next_alignment()

    assert not isinstance(result, AlignmentEditApplied)
    assert events == []


def test_commands_reset_alignment_to_initial_clears_pending_lines() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    active_key = AlignmentKey("rec", "stream", 1)
    state = workspace.document.alignment_state_for(active_key)
    pending_lines_at_event: list[object] = []
    workspace.app.events.subscribe(
        AlignmentEdited,
        lambda _event: pending_lines_at_event.append(state.pending_reference_lines),
    )
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
    )
    workspace.document.active_set_pending_reference_lines([10.0], [11.0])
    shank_runtime = SimpleNamespace(
        shank_idx=1,
        ephysalign=SimpleNamespace(
            feature_init=np.array([1.0, 3.0]),
            track_init=np.array([2.0, 4.0]),
        ),
    )

    result = workspace.app.commands.reset_alignment_to_initial(
        shank_runtime,
        lin_fit=False,
    )

    assert isinstance(result, AlignmentEditApplied)
    np.testing.assert_allclose(result.alignment.feature, [1.0, 3.0])
    np.testing.assert_allclose(result.alignment.track, [2.0, 4.0])
    assert state.pending_reference_lines is None
    assert pending_lines_at_event == [None]


def test_queries_return_active_shank_selection_state() -> None:
    document = AlignmentDocument()
    key = AlignmentKey("rec", "stream", 2)
    document.select_alignment_key(key)
    document.mark_data_loaded(True)
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    state = queries.active_shank_selection()

    assert state.shank_idx == 2
    assert state.shank_id == 3
    assert state.alignment_key == key
    assert state.data_loaded


def test_queries_identify_loaded_stream_shank() -> None:
    document = AlignmentDocument()
    document.select_alignment_key(AlignmentKey("rec", "stream", 1))
    document.mark_data_loaded(True)
    stream_runtime = SimpleNamespace(
        stream_key=("rec", "stream"),
        current_shank_idx=1,
    )
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=stream_runtime,
            current_stream_key=("rec", "stream"),
        ),
    )

    assert queries.is_loaded_stream_shank(("rec", "stream"), 1)


def test_queries_reject_loaded_stream_shank_mismatches() -> None:
    document = AlignmentDocument()
    document.select_alignment_key(AlignmentKey("rec", "stream", 1))
    document.mark_data_loaded(True)
    stream_runtime = SimpleNamespace(
        stream_key=("rec", "stream"),
        current_shank_idx=1,
    )
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=stream_runtime,
            current_stream_key=("rec", "stream"),
        ),
    )

    assert not queries.is_loaded_stream_shank(("rec", "other-stream"), 1)
    assert not queries.is_loaded_stream_shank(("rec", "stream"), 0)
    assert not queries.is_loaded_stream_shank(None, 1)
    document.mark_data_loaded(False)
    assert not queries.is_loaded_stream_shank(("rec", "stream"), 1)


def test_queries_build_plot_menu_state_from_active_runtime_shank() -> None:
    document = AlignmentDocument()
    document.select_alignment_key(
        AlignmentKey(
            recording_id="rec",
            ephys_collection="stream",
            shank_idx=2,
        )
    )
    stream_runtime = FakeStreamRuntime()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=stream_runtime),
    )

    state = queries.active_plot_menu_state()

    assert state.group("image").selected_key == "image.fr"
    assert stream_runtime.calls == [2]


def test_queries_resolve_plot_payload_from_active_runtime_shank() -> None:
    document = AlignmentDocument(selected_shank=1)
    stream_runtime = FakeStreamRuntime()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=stream_runtime),
    )

    payload = queries.active_plot_payload("image.fr")

    assert payload == {"label": "shank-1"}
    assert stream_runtime.calls == [1]


def test_queries_can_resolve_raw_payload_without_plotdata() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    state = queries.active_plot_menu_state(
        previous_selected_keys={"image": "image.raw.raw_ap"},
        raw_image_payloads={"raw_ap": "raw-image"},
    )
    payload = queries.active_plot_payload(
        "image.raw.raw_ap",
        raw_image_payloads={"raw_ap": "raw-image"},
    )

    assert state.group("image").selected_key == "image.raw.raw_ap"
    assert payload == "raw-image"


def test_queries_fail_closed_without_plotdata_or_raw_payloads() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    state = queries.active_plot_menu_state()

    assert not state.group("image").enabled
    assert queries.active_plot_payload("image.fr") is None


def test_queries_build_active_alignment_render_state_from_document_runtime() -> None:
    document = AlignmentDocument()
    key = AlignmentKey("rec", "stream", 1)
    state = document.select_alignment_key(key)
    active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    state.active_alignment = active_alignment
    display_state = AlignmentDisplayState(
        region_annotation_source="FranklinPaxinos"
    )
    shank_runtime = SimpleNamespace(
        ephysalign="aligner",
        region_fp="region-fp",
        region_label_fp="region-label-fp",
        region_colour_fp="region-colour-fp",
    )
    derived = FakeDerivedDataService()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=SimpleNamespace(
                shank_runtime_by_idx={1: shank_runtime}
            )
        ),
        display_state=display_state,
        derived_data_service=derived,
    )

    render_state = queries.active_alignment_render_state()

    assert render_state is not None
    assert render_state.key == key
    np.testing.assert_allclose(
        render_state.active_alignment.feature,
        active_alignment.feature,
    )
    np.testing.assert_allclose(
        render_state.active_alignment.track,
        active_alignment.track,
    )
    assert render_state.histology == "histology"
    assert render_state.projection == "projection"
    assert derived.histology_kwargs["ephysalign"] == "aligner"
    assert derived.histology_kwargs["feature"] is render_state.active_alignment.feature
    assert derived.histology_kwargs["track"] is render_state.active_alignment.track
    assert (
        derived.histology_kwargs["region_annotation_source"]
        == "FranklinPaxinos"
    )
    assert derived.histology_kwargs["region_fp"] == "region-fp"
    assert derived.histology_kwargs["region_label_fp"] == "region-label-fp"
    assert derived.histology_kwargs["region_colour_fp"] == "region-colour-fp"
    assert derived.projection_kwargs["ephysalign"] == "aligner"
    assert derived.projection_kwargs["feature"] is render_state.active_alignment.feature
    assert derived.projection_kwargs["track"] is render_state.active_alignment.track


def test_queries_build_histology_and_scale_panel_states() -> None:
    document = AlignmentDocument()
    key = AlignmentKey("rec", "stream", 1)
    state = document.select_alignment_key(key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 0.005]),
        np.array([0.0, 0.004]),
    )
    histology = AlignmentHistologyData(
        histology=HistologyPlotData(
            region=np.array([[0.0, 100.0]]),
            axis_label=np.array([[50.0, "VISp"]], dtype=object),
            colour=np.array([[1, 2, 3]]),
        ),
        reference_histology=HistologyPlotData(
            region=np.array([[0.0, 120.0]]),
            axis_label=np.array([[60.0, "VISp"]], dtype=object),
            colour=np.array([[4, 5, 6]]),
        ),
        scale=ScaleFactorData(
            region=np.array([[0.0, 100.0]]),
            scale=np.array([1.1]),
        ),
    )
    derived = FakeDerivedDataService(histology=histology)
    shank_runtime = SimpleNamespace(
        ephysalign="aligner",
        region_fp=None,
        region_label_fp=None,
        region_colour_fp=None,
    )
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=SimpleNamespace(
                shank_runtime_by_idx={1: shank_runtime}
            )
        ),
        derived_data_service=derived,
    )

    histology_state = queries.active_histology_panel_state(
        probe_tip_um=0.0,
        probe_top_um=3840.0,
        probe_extra_um=100.0,
    )
    scale_state = queries.active_scale_factor_state(
        probe_tip_um=0.0,
        probe_top_um=3840.0,
        probe_extra_um=100.0,
    )

    assert histology_state is not None
    assert histology_state.key == key
    assert histology_state.histology is histology
    assert histology_state.probe_extent.tip_bounds_um == (1.0, 1159.0)
    assert histology_state.probe_extent.top_bounds_um == (3841.0, 4999.0)
    assert scale_state is not None
    assert scale_state.key == key
    assert scale_state.region is histology.scale.region
    assert scale_state.scale is histology.scale.scale


def test_queries_build_fit_plot_state() -> None:
    document = AlignmentDocument()
    key = AlignmentKey("rec", "stream", 1)
    state = document.select_alignment_key(key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 0.001, 0.002, 0.003, 0.004]),
        np.array([0.0, 0.0015, 0.002, 0.0035, 0.004]),
    )
    ephysalign = FakeFitAligner()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=SimpleNamespace(
                shank_runtime_by_idx={
                    1: SimpleNamespace(ephysalign=ephysalign)
                }
            )
        ),
    )

    fit_state = queries.active_fit_plot_state(
        depth_um=np.array([0.0, 20.0]),
        lin_fit=True,
    )

    assert fit_state is not None
    assert fit_state.key == key
    np.testing.assert_allclose(
        fit_state.feature_um,
        [0.0, 1000.0, 2000.0, 3000.0, 4000.0],
    )
    np.testing.assert_allclose(
        fit_state.track_um,
        [0.0, 1500.0, 2000.0, 3500.0, 4000.0],
    )
    np.testing.assert_allclose(fit_state.linear_feature_um, [0.0, 20.0])
    np.testing.assert_allclose(fit_state.linear_track_um, [1000.0, 1020.0])
    assert len(ephysalign.calls) == 1


def test_queries_active_alignment_render_state_fails_closed_without_runtime() -> None:
    document = AlignmentDocument()
    state = document.select_alignment_key(AlignmentKey("rec", "stream", 1))
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    assert queries.active_alignment_render_state() is None


def test_queries_ensure_active_slice_data_state_uses_runtime_cache() -> None:
    document = AlignmentDocument()
    key = AlignmentKey("rec", "stream", 1)
    state = document.select_alignment_key(key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    track = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    shank_runtime = SimpleNamespace(
        ephysalign=SimpleNamespace(track_interpolation_ras=track),
        slice_runtime=SliceRuntime(),
    )
    slice_service = FakeSliceService()
    histology_context = SimpleNamespace(
        brain_atlas=FakeBrainAtlas(),
        histology_images={},
        lazy_channel_paths={},
    )
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=SimpleNamespace(
                shank_runtime_by_idx={1: shank_runtime}
            )
        ),
        histology_context=histology_context,
        slice_service=slice_service,
    )

    first = queries.ensure_active_slice_data_state()
    second = queries.ensure_active_slice_data_state()

    assert first is not None
    assert second is not None
    assert first.key == key
    assert second.slice_data is first.slice_data
    assert len(slice_service.slice_set_calls) == 1
    call = slice_service.slice_set_calls[0]
    assert call["brain_atlas"] is histology_context.brain_atlas
    assert call["histology_images"] == {}
    assert call["lazy_channel_paths"] == {}
    assert call["track_interpolation_ras"] is track
    assert queries.active_slice_data_by_attr()["slice_data"] is first.slice_data


def test_queries_build_active_slice_menu_state_with_fallback_selection() -> None:
    slice_data = {
        "ccf": np.zeros((2, 2)),
        "label": np.zeros((2, 2, 3)),
        "histology_registration": np.ones((2, 2)),
        "scale": np.array([1.0, 2.0]),
        "offset": np.array([3.0, 4.0]),
    }
    queries, key, _ephysalign = _queries_with_cached_slice(
        slice_data=slice_data,
        fp_slice_data={"label": np.zeros((2, 2, 3))},
    )

    restored = queries.active_slice_menu_state(
        offline=True,
        previous_selection=SliceSelection("slice_data", "histology_registration"),
    )
    fallback = queries.active_slice_menu_state(
        offline=True,
        previous_selection=SliceSelection("slice_data", "missing"),
    )

    assert restored is not None
    assert fallback is not None
    assert restored.key == key
    assert [item.label for item in restored.items] == [
        "CCF",
        "Annotation",
        "Annotation FP",
        "histology_registration",
    ]
    assert restored.default_selection == SliceSelection(
        "slice_data",
        "histology_registration",
    )
    assert restored.selection.selection == SliceSelection(
        "slice_data",
        "histology_registration",
    )
    assert restored.selection.used_previous
    assert fallback.selection.selection == restored.default_selection
    assert not fallback.selection.used_previous


def test_queries_build_active_slice_render_state_for_selection() -> None:
    image = np.arange(4.0).reshape(2, 2)
    slice_data = {
        "ccf": np.zeros((2, 2)),
        "label": np.zeros((2, 2, 3)),
        "annotation_ids": np.array([[1, 1], [0, 1]]),
        "histology_registration": image,
        "scale": np.array([1.0, 2.0]),
        "offset": np.array([3.0, 4.0]),
    }
    derived = FakeDerivedDataService(projection="projection")
    queries, key, ephysalign = _queries_with_cached_slice(
        slice_data=slice_data,
        derived=derived,
    )

    render_state = queries.active_slice_render_state(
        SliceSelection("slice_data", "histology_registration")
    )

    assert render_state is not None
    assert render_state.key == key
    assert render_state.selection == SliceSelection(
        "slice_data",
        "histology_registration",
    )
    assert render_state.image is image
    np.testing.assert_allclose(render_state.scale, [1.0, 2.0])
    np.testing.assert_allclose(render_state.offset, [3.0, 4.0])
    assert render_state.decision.kind is SliceImageKind.SCALAR
    assert render_state.scalar_channel == "histology_registration"
    np.testing.assert_allclose(
        render_state.track_annos_and_ends_ras,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]],
    )
    assert render_state.projection == "projection"
    assert derived.projection_kwargs["ephysalign"] is ephysalign
    np.testing.assert_allclose(
        derived.projection_kwargs["feature"],
        [0.0, 1.0],
    )
    np.testing.assert_allclose(
        derived.projection_kwargs["track"],
        [2.0, 3.0],
    )


def test_queries_build_active_perpendicular_slice_state_from_runtime_cache() -> None:
    document = AlignmentDocument()
    key = AlignmentKey("rec", "stream", 1)
    state = document.select_alignment_key(key)
    active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    state.active_alignment = active_alignment
    histology = AlignmentHistologyData(
        histology=HistologyPlotData(
            region=np.array([-200.0, 5000.0]),
            axis_label=[],
            colour=[],
        ),
        reference_histology=HistologyPlotData(region=[], axis_label=[], colour=[]),
        scale=ScaleFactorData(region=[], scale=[]),
    )
    track_interpolation = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [2.0, 0.0, 4.0]]
    )
    ephysalign = SimpleNamespace(
        track_interpolation_ras=track_interpolation,
        ephys_depths_along_track=np.array([0.0, 1.0, 2.0]),
    )
    shank_runtime = SimpleNamespace(
        ephysalign=ephysalign,
        slice_runtime=SliceRuntime(),
        chn_depths=np.array([0.0, 100.0]),
        region_fp=None,
        region_label_fp=None,
        region_colour_fp=None,
    )
    slice_service = FakeSliceService()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=SimpleNamespace(
                shank_runtime_by_idx={1: shank_runtime}
            )
        ),
        derived_data_service=FakeDerivedDataService(histology=histology),
        histology_context=SimpleNamespace(
            brain_atlas=FakeBrainAtlas(dv_voxel_m=20e-6),
            histology_images={},
            lazy_channel_paths={},
        ),
        slice_service=slice_service,
    )

    first = queries.active_perpendicular_slice_state("ccf")
    second = queries.active_perpendicular_slice_state("ccf")

    assert first is not None
    assert second is not None
    assert first.key == key
    assert first.channel_name == "ccf"
    assert first.extent_um == 500.0
    assert first.feature_min_um == -200.0
    assert first.feature_max_um == 5000.0
    assert first.n_perp_samples == 51
    assert first.n_depths == 261
    assert first.image.shape == (51, 261)
    assert second.image is first.image
    assert len(slice_service.perpendicular_calls) == 1
    call = slice_service.perpendicular_calls[0]
    assert call["brain_atlas"] is queries.histology_context.brain_atlas
    assert call["ephysalign"] is ephysalign
    assert call["channel_name"] == "ccf"
    assert call["n_perp_samples"] == 51
    np.testing.assert_allclose(call["feature_ref"], active_alignment.feature)
    np.testing.assert_allclose(call["track_ref"], active_alignment.track)
    np.testing.assert_allclose(first.channel_depths_um, [0.0, 100.0])
