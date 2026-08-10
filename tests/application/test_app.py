"""Tests for the UI-facing alignment app port."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.application.queries import AlignmentQueries
from ephys_alignment_gui.application.results import (
    ActiveStreamDetached,
    AlignmentChoicesUpdated,
    AlignmentEditApplied,
    AlignmentEditNoop,
    CachedEphysDataActivated,
    LoadDataAlreadyActiveResult,
    LoadDataCachedActivated,
    LoadDataFreshCompleted,
    LoadDataFreshPrepared,
    LoadDataFreshRequiredResult,
    LoadDataPrepared,
    LoadedShankPrepared,
    PendingReferenceLinesUpdated,
    PreviousAlignmentSelected,
    ShankSelected,
    StreamCacheEvicted,
    VisitedAlignmentOutputsSaved,
)
from ephys_alignment_gui.application.results.alignment_persistence import (
    NoPreviousAlignments,
)
from ephys_alignment_gui.application.results.metadata import (
    MouseRootLoaded,
    ProbeSelected,
    RecordingSelected,
)
from ephys_alignment_gui.application.results.path import (
    OutputDirectoryDerived,
    OutputRootSet,
)
from ephys_alignment_gui.application.workflow import Blocked, Failed, Ok
from ephys_alignment_gui.application.workspace import AlignmentWorkspace
from ephys_alignment_gui.core.active_alignment import ActiveAlignment
from ephys_alignment_gui.core.alignment_display_state import AlignmentDisplayState
from ephys_alignment_gui.core.alignment_events import AlignmentEdited, ShankChanged
from ephys_alignment_gui.core.alignment_read_models import (
    ActiveAlignmentEditScreenState,
    ActiveReferenceLineRenderState,
)
from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.core.slice_display_policy import SliceImageKind, SliceSelection
from ephys_alignment_gui.io.datapackage_loader import MouseRoot, ProbeInfo
from ephys_alignment_gui.io.load_data_job import (
    LoadDataJobCompleted,
    LoadDataJobRequest,
)
from ephys_alignment_gui.runtime.histology_loader import HistologyDataLoaded
from ephys_alignment_gui.runtime.session import (
    LoadDataAlreadyActive,
    LoadDataCachedStreamAvailable,
    LoadDataFreshRequired,
    SessionRuntime,
)
from ephys_alignment_gui.runtime.slice import SliceRuntime
from ephys_alignment_gui.services.alignment_derived_data import (
    AlignmentHistologyData,
    HistologyPlotData,
    ScaleFactorData,
)
from ephys_alignment_gui.services.alignment_repository import (
    LoadedAlignmentHistory,
    SavedAlignmentOutputs,
)
from ephys_alignment_gui.services.ephys_data import ChannelTable, EphysStreamData


class FakePlotPayloadCache:
    def __init__(self, label: str = "plot") -> None:
        self.label = label
        self.filtered_subsets: list[str] = []
        self.in_brain_depths_um = np.array([20.0, 40.0])
        self.chn_min = 5.0
        self.chn_max = 200.0
        self.t_autocorr = np.array([0.0, 1.0, 2.0])
        self.t_template = np.array([0.0, 0.5])
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

    def filter_units(self, subset: str) -> None:
        self.filtered_subsets.append(subset)

    def get_autocorr(self, cluster_idx: int):
        return np.array(
            [cluster_idx, cluster_idx + 1, cluster_idx + 2]
        ), cluster_idx + 10

    def get_template_wf(self, cluster_idx: int):
        return np.array([cluster_idx + 0.5, cluster_idx + 1.5])


class FakeStreamRuntime:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.shank_runtime_by_idx = {}
        self.plot_payload_cache_by_shank = {
            1: FakePlotPayloadCache("shank-1"),
            2: FakePlotPayloadCache("shank-2"),
        }

    def plot_payload_cache_for_shank(self, shank_idx: int) -> FakePlotPayloadCache:
        self.calls.append(shank_idx)
        return self.plot_payload_cache_by_shank[shank_idx]

    def filtered_plot_payload_cache_for_shank(
        self,
        shank_idx: int,
        *,
        unit_filter: str,
    ) -> FakePlotPayloadCache:
        payload_cache = self.plot_payload_cache_for_shank(shank_idx)
        payload_cache.filter_units(unit_filter)
        return payload_cache

    def visited_shank_runtimes(self):
        return self.shank_runtime_by_idx


class FakeAlignmentRepository:
    def __init__(self) -> None:
        self.loaded_alignments = None
        self.loaded_kwargs = None
        self.saved_kwargs: list[dict[str, Any]] = []

    def load_previous_alignments(self, **kwargs):
        self.loaded_kwargs = kwargs
        if self.loaded_alignments is None:
            return None
        return LoadedAlignmentHistory(self.loaded_alignments)

    def save_alignment_outputs(self, **kwargs):
        self.saved_kwargs.append(kwargs)
        return SavedAlignmentOutputs(
            channel_results_path=kwargs["output_directory"] / "channels.json",
            previous_alignments_path=kwargs["output_directory"] / "alignments.json",
            ccf_channel_results_path=kwargs["output_directory"] / "ccf.json",
            docdb_probe_name="probeA_0" if kwargs["use_docdb"] else None,
        )


class FakeBatchOutputBuilder:
    def __init__(self) -> None:
        self.batched_alignments = None

    def get_alignment_results_batch(self, alignments):
        self.batched_alignments = alignments
        return {
            key: (
                {"channel": key.shank_idx},
                {"ccf": key.shank_idx},
                True,
            )
            for key in alignments
        }


class FakeRuntimeInitializer:
    def __init__(self) -> None:
        self.calls = []

    def initialize_shank_runtime(self, shank_runtime, **kwargs):
        self.calls.append((shank_runtime, kwargs))
        shank_runtime.track_annotations_ras = kwargs["track_annotations_ras"]
        return SimpleNamespace(
            feature_init=np.array([1.0, 2.0]),
            track_init=np.array([3.0, 4.0]),
            track_annos_and_ends_ras=np.array([[1.0, 2.0, 3.0]]),
        )


class FakeProbeTrackService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def load_track_annotations(self, **kwargs):
        self.calls.append(kwargs)
        return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])


class FakeDerivedDataService:
    def __init__(
        self,
        *,
        histology: Any = "histology",
        projection: Any = "projection",
    ) -> None:
        self.histology = histology
        self.projection = projection
        self.nearby_boundaries = SimpleNamespace(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            colours=["red", "blue"],
            parent_x=np.array([5.0, 6.0]),
            parent_y=np.array([7.0, 8.0]),
            parent_colours=["pink", "cyan"],
        )
        self.histology_kwargs = None
        self.projection_kwargs = None
        self.nearby_kwargs: list[dict[str, Any]] = []
        self.channel_location_calls: list[dict[str, Any]] = []

    def compute_histology(self, **kwargs):
        self.histology_kwargs = kwargs
        return self.histology

    def compute_channel_projection(self, **kwargs):
        self.projection_kwargs = kwargs
        return self.projection

    def compute_nearby_boundaries(self, **kwargs):
        self.nearby_kwargs.append(kwargs)
        return self.nearby_boundaries

    def compute_channel_locations(self, **kwargs):
        self.channel_location_calls.append(kwargs)
        return np.array([[1.0, 2.0, 3.0]])


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


class FakeEditEphysAlignment:
    feature_init = np.array([1.0, 3.0])
    track_init = np.array([2.0, 4.0])

    def __init__(self) -> None:
        self.feature2track_calls: list[tuple[Any, Any, Any]] = []
        self.uniform_calls: list[tuple[Any, Any]] = []
        self.linear_calls: list[tuple[Any, Any, Any]] = []

    def feature2track(self, depths_track, feature_ref, track_ref):
        self.feature2track_calls.append((depths_track, feature_ref, track_ref))
        return np.asarray(depths_track, dtype=float) + 0.1

    def adjust_extremes_uniform(self, feature, track):
        self.uniform_calls.append((feature, track))
        return np.asarray(track, dtype=float) + 0.01

    def adjust_extremes_linear(self, feature, track, extend_feature=1):
        self.linear_calls.append((feature, track, extend_feature))
        return (
            np.asarray(feature, dtype=float) + extend_feature,
            np.asarray(track, dtype=float) + extend_feature,
        )


class FakeEphysDataService:
    def __init__(self) -> None:
        self.loaded_probe: ProbeInfo | None = None

    def load_channel_table(self, probe: ProbeInfo) -> ChannelTable:
        self.loaded_probe = probe
        return ChannelTable(
            local_coordinates=np.array([[0.0, 0.0], [250.0, 0.0]]),
            shank_indices=np.array([0, 1]),
        )


class FakeLoadDataJob:
    def __init__(self, result: Any | None = None) -> None:
        self.result = result or LoadDataJobCompleted(
            ephys=SimpleNamespace(stream=_ephys_stream()),
            histology=HistologyDataLoaded(),
        )
        self.calls: list[LoadDataJobRequest] = []

    def run(self, request: LoadDataJobRequest):
        self.calls.append(request)
        return self.result


def _probe_info() -> ProbeInfo:
    return ProbeInfo(
        probe_id="probe-id",
        probe_name="probeA",
        recording_id="rec",
        logical_probe="probeA",
        ephys_collection="stream",
        num_shanks=2,
        ephys_dir=Path("/tmp/ephys"),
        channel_table=None,
        xyz_picks=(),
    )


def _mouse_root_with_probe(probe: ProbeInfo | None = None) -> MouseRoot:
    probe = probe or _probe_info()
    return MouseRoot(
        root=Path("/tmp/mouse"),
        schema_version="3.1.0",
        mouse_id="mouse",
        transforms=None,
        histology=None,
        probes={probe.recording_id: {probe.probe_name: probe}},
    )


def _ephys_stream(ephys_collection: str = "stream") -> EphysStreamData:
    return EphysStreamData(
        recording_id="rec",
        ephys_collection=ephys_collection,
        ephys_dir=Path("/tmp/ephys"),
        channel_table=ChannelTable(
            local_coordinates=np.array([[0.0, 0.0], [250.0, 0.0]]),
            shank_indices=np.array([0, 1]),
        ),
        alf_data={},
        session_notes="notes",
    )


def _workspace_with_probe_state(
    *,
    shank_idx: int = 1,
    repo: FakeAlignmentRepository | None = None,
) -> AlignmentWorkspace:
    workspace = AlignmentWorkspace()
    if repo is not None:
        workspace.persistence_commands.alignment_repository = repo
    workspace.data_context.probe_info = SimpleNamespace(
        recording_id="rec",
        probe_name="probeA",
        ephys_collection="stream",
    )
    workspace.data_context.channel_table = SimpleNamespace(n_shanks=2)
    workspace.alignment_key_context.set_from_probe(
        workspace.data_context.probe_info,
        n_shanks=2,
    )
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
        track_annos_and_ends_ras=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]]),
    )
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=SimpleNamespace(
                shank_runtime_by_idx={1: shank_runtime}
            )
        ),
        derived_data_service=derived or FakeDerivedDataService(),
        histology_context=SimpleNamespace(
            brain_atlas=FakeBrainAtlas(),
            histology_images={},
            lazy_channel_paths={},
        ),
        slice_service=FakeSliceService(),
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

    result = workspace.app.commands.shanks.select_shank(1, source="test")

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

    result = workspace.app.commands.shanks.select_shank(
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

    result = workspace.app.commands.shanks.select_shank(
        1, outgoing_reference_lines=None
    )

    assert isinstance(result, ShankSelected)
    assert workspace.document.alignment_state_for(key0).pending_reference_lines is None


def test_commands_select_shank_without_line_state_leaves_pending_lines() -> None:
    workspace = AlignmentWorkspace()
    key0 = AlignmentKey("rec", "stream", 0)
    workspace.document.select_alignment_key(key0)
    workspace.document.mark_data_loaded(True)
    workspace.document.active_set_pending_reference_lines([1.0], [2.0])

    result = workspace.app.commands.shanks.select_shank(1)

    assert isinstance(result, ShankSelected)
    pending = workspace.document.alignment_state_for(key0).pending_reference_lines
    assert pending is not None
    np.testing.assert_allclose(pending.feature_positions_um, [1.0])
    np.testing.assert_allclose(pending.track_positions_um, [2.0])


def test_commands_capture_active_reference_lines_updates_document_state() -> None:
    workspace = AlignmentWorkspace()
    key = AlignmentKey("rec", "stream", 0)
    workspace.document.select_alignment_key(key)
    workspace.document.mark_data_loaded(True)

    result = workspace.app.commands.shanks.capture_active_reference_lines(
        ([10.0, 20.0], [11.0, 21.0])
    )

    assert isinstance(result, PendingReferenceLinesUpdated)
    pending = workspace.document.alignment_state_for(key).pending_reference_lines
    assert pending is not None
    np.testing.assert_allclose(pending.feature_positions_um, [10.0, 20.0])
    np.testing.assert_allclose(pending.track_positions_um, [11.0, 21.0])


def test_commands_capture_active_reference_lines_clears_missing_lines() -> None:
    workspace = AlignmentWorkspace()
    key = AlignmentKey("rec", "stream", 0)
    workspace.document.select_alignment_key(key)
    workspace.document.mark_data_loaded(True)
    workspace.document.active_set_pending_reference_lines([1.0], [2.0])

    result = workspace.app.commands.shanks.capture_active_reference_lines(None)

    assert isinstance(result, PendingReferenceLinesUpdated)
    assert workspace.document.alignment_state_for(key).pending_reference_lines is None


def test_commands_capture_active_reference_lines_noops_when_data_unloaded() -> None:
    workspace = AlignmentWorkspace()

    result = workspace.app.commands.shanks.capture_active_reference_lines(None)

    assert isinstance(result, Ok)


def test_queries_active_reference_line_state_prefers_pending_lines() -> None:
    workspace = AlignmentWorkspace()
    key = AlignmentKey("rec", "stream", 0)
    state = workspace.document.select_alignment_key(key)
    state.feature_prev = np.array([0.0, 1.0, 2.0])
    workspace.document.active_set_pending_reference_lines([10.0], [11.0])

    result = workspace.app.queries.workspace.active_reference_line_state(0)

    assert isinstance(result, ActiveReferenceLineRenderState)
    np.testing.assert_allclose(result.feature_positions_um, [10.0])
    np.testing.assert_allclose(result.track_positions_um, [11.0])


def test_queries_active_reference_line_state_falls_back_to_previous_feature() -> None:
    workspace = AlignmentWorkspace()
    key = AlignmentKey("rec", "stream", 0)
    state = workspace.document.select_alignment_key(key)
    state.feature_prev = np.array([-0.001, 0.001, 0.002, 0.003])

    result = workspace.app.queries.workspace.active_reference_line_state(0)

    assert isinstance(result, ActiveReferenceLineRenderState)
    np.testing.assert_allclose(result.feature_positions_um, [1000.0, 2000.0])
    assert result.track_positions_um is None


def test_queries_active_reference_line_state_rejects_mismatched_shank() -> None:
    workspace = AlignmentWorkspace()
    workspace.document.select_alignment_key(AlignmentKey("rec", "stream", 0))
    workspace.document.active_set_pending_reference_lines([10.0], [11.0])

    assert workspace.app.queries.workspace.active_reference_line_state(1) is None


def test_commands_prepare_fresh_ephys_load_marks_unloaded_and_evicts_stale() -> None:
    workspace = AlignmentWorkspace()
    workspace.document.mark_data_loaded(True)
    workspace.display_state.set_unit_filter("KS good")
    stream_runtime = workspace.runtime.cache_loaded_stream_data(
        _ephys_stream(),
        workspace.plot_payload_cache_factory,
        shank_idx=1,
    )

    result = workspace.app.commands.load.prepare_fresh_ephys_load(("rec", "stream"))

    assert isinstance(result, LoadDataPrepared)
    assert result.preserve_plot_selection
    assert not workspace.document.data_loaded
    assert workspace.runtime.active_stream_runtime is None
    assert workspace.runtime.current_stream_key is None
    assert ("rec", "stream") not in workspace.runtime.stream_cache
    assert stream_runtime.stream_key == ("rec", "stream")
    assert workspace.display_state.unit_filter == "all"


def test_commands_begin_load_data_noops_when_stream_shank_already_active() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    workspace.data_context.mouse_root = _mouse_root_with_probe()
    workspace.runtime.cache_loaded_stream_data(
        _ephys_stream(),
        workspace.plot_payload_cache_factory,
        shank_idx=1,
    )
    workspace.document.mark_data_loaded(True)
    state = workspace.document.active_alignment_state

    result = workspace.app.commands.load.begin_load_data(
        recording_id="rec",
        probe_name="probeA",
        target_shank=1,
        outgoing_reference_lines=([10.0], [11.0]),
    )

    assert isinstance(result, LoadDataAlreadyActiveResult)
    assert result.stream_key == ("rec", "stream")
    assert result.shank_idx == 1
    assert state is not None
    assert state.pending_reference_lines is None
    assert workspace.document.data_loaded


def test_commands_begin_load_data_activates_cached_stream_and_captures_lines() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    workspace.data_context.mouse_root = _mouse_root_with_probe()
    old_key = AlignmentKey("rec", "stream", 1)
    old_state = workspace.document.alignment_state_for(old_key)
    workspace.document.mark_data_loaded(True)
    stream_runtime = workspace.runtime.cache_loaded_stream_data(
        _ephys_stream(),
        workspace.plot_payload_cache_factory,
        shank_idx=1,
    )
    workspace.runtime.clear_active_stream()

    result = workspace.app.commands.load.begin_load_data(
        recording_id="rec",
        probe_name="probeA",
        target_shank=0,
        outgoing_reference_lines=([10.0], [11.0]),
    )

    assert isinstance(result, LoadDataCachedActivated)
    assert result.stream_key == ("rec", "stream")
    assert result.activated.stream_runtime is stream_runtime
    assert result.activated.shank_idx == 0
    assert workspace.runtime.active_stream_runtime is stream_runtime
    assert workspace.document.selected_alignment_key == AlignmentKey("rec", "stream", 0)
    assert old_state.pending_reference_lines is not None
    np.testing.assert_allclose(
        old_state.pending_reference_lines.feature_positions_um,
        [10.0],
    )


def test_commands_begin_load_data_prepares_fresh_load_and_captures_lines() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    workspace.data_context.mouse_root = _mouse_root_with_probe()
    old_key = AlignmentKey("rec", "stream", 1)
    old_state = workspace.document.alignment_state_for(old_key)
    workspace.document.mark_data_loaded(True)
    workspace.display_state.set_unit_filter("KS good")

    result = workspace.app.commands.load.begin_load_data(
        recording_id="rec",
        probe_name="probeA",
        target_shank=0,
        outgoing_reference_lines=([10.0], [11.0]),
    )

    assert isinstance(result, LoadDataFreshPrepared)
    assert result.stream_key == ("rec", "stream")
    assert result.shank_idx == 0
    assert result.preserve_plot_selection
    assert not workspace.document.data_loaded
    assert workspace.document.selected_alignment_key == AlignmentKey("rec", "stream", 0)
    assert workspace.runtime.active_stream_runtime is None
    assert workspace.display_state.unit_filter == "all"
    assert old_state.pending_reference_lines is not None


def test_commands_complete_fresh_load_data_returns_typed_transaction_result() -> None:
    workspace = _workspace_with_probe_state(shank_idx=0)
    load_data_job = FakeLoadDataJob()
    workspace.load_data_commands.load_data_job = load_data_job
    prepared = LoadDataFreshPrepared(
        stream_key=("rec", "stream"),
        shank_idx=1,
        preserve_plot_selection=True,
    )

    result = workspace.app.commands.load.complete_fresh_load_data(prepared)

    assert isinstance(result, LoadDataFreshCompleted)
    assert result.stream_key == ("rec", "stream")
    assert result.preserve_plot_selection
    assert result.ephys.shank_idx == 1
    assert isinstance(result.histology, HistologyDataLoaded)
    assert load_data_job.calls == [LoadDataJobRequest(1)]
    assert workspace.document.data_loaded


def test_commands_activate_cached_probe_selection_reports_fresh_required() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    workspace.data_context.mouse_root = _mouse_root_with_probe()
    workspace.document.mark_data_loaded(True)

    result = workspace.app.commands.load.activate_cached_probe_selection(
        recording_id="rec",
        probe_name="probeA",
        target_shank=0,
    )

    assert isinstance(result, LoadDataFreshRequiredResult)
    assert result.stream_key == ("rec", "stream")
    assert result.shank_idx == 0
    assert workspace.document.data_loaded
    assert workspace.runtime.active_stream_runtime is None


def test_commands_detach_active_stream_preserves_cache_and_resets_display() -> None:
    workspace = AlignmentWorkspace()
    workspace.display_state.set_unit_filter("KS good")
    stream_runtime = workspace.runtime.cache_loaded_stream_data(
        _ephys_stream(),
        workspace.plot_payload_cache_factory,
        shank_idx=1,
    )

    result = workspace.app.commands.load.detach_active_stream()

    assert isinstance(result, ActiveStreamDetached)
    assert result.cached_stream_count == 1
    assert workspace.runtime.active_stream_runtime is None
    assert workspace.runtime.current_stream_key is None
    assert workspace.runtime.stream_cache[("rec", "stream")] is stream_runtime
    assert workspace.display_state.unit_filter == "all"


def test_commands_evict_stream_cache_clears_cache_and_resets_display() -> None:
    workspace = AlignmentWorkspace()
    workspace.display_state.set_unit_filter("KS good")
    workspace.runtime.cache_loaded_stream_data(
        _ephys_stream("streamA"),
        workspace.plot_payload_cache_factory,
        shank_idx=0,
    )
    workspace.runtime.cache_loaded_stream_data(
        _ephys_stream("streamB"),
        workspace.plot_payload_cache_factory,
        shank_idx=0,
    )

    result = workspace.app.commands.load.evict_stream_cache()

    assert isinstance(result, StreamCacheEvicted)
    assert result.evicted_stream_count == 2
    assert workspace.runtime.stream_cache == {}
    assert workspace.runtime.active_stream_runtime is None
    assert workspace.runtime.current_stream_key is None
    assert workspace.display_state.unit_filter == "all"


def test_commands_path_operations_update_document_and_context(tmp_path) -> None:
    workspace = AlignmentWorkspace()
    loaded_root = SimpleNamespace(root=tmp_path / "mouse", mouse_id="mouse")
    loaded_root.root.mkdir()
    probe = _probe_info()
    data_context = SimpleNamespace(
        mouse_root=None,
        probe_info=probe,
        set_mouse_root=lambda path: loaded_root,
    )
    workspace.metadata_commands.data_context = data_context
    workspace.path_commands.data_context = data_context

    mouse_result = workspace.app.commands.metadata.set_mouse_root(loaded_root.root)
    workspace.document.select_probe(probe.recording_id, probe.probe_name)
    root_result = workspace.app.commands.paths.set_output_root(tmp_path / "results")
    derived_result = workspace.app.commands.paths.derive_output_directory()

    assert isinstance(mouse_result, MouseRootLoaded)
    assert mouse_result.mouse_root.root == loaded_root.root
    assert workspace.document.mouse_root == loaded_root.root
    assert isinstance(root_result, OutputRootSet)
    assert root_result.output_directory == tmp_path / "results" / "rec" / "probeA"
    assert isinstance(derived_result, OutputDirectoryDerived)
    assert derived_result.output_directory == tmp_path / "results" / "rec" / "probeA"
    assert workspace.document.output_directory == derived_result.output_directory


def test_commands_can_load_data_delegates_to_workflow_policy() -> None:
    workspace = AlignmentWorkspace()

    result = workspace.app.commands.load.can_load_data()

    assert isinstance(result, Blocked)
    assert result.first.code == "probe_required"


def test_queries_expose_active_paths_and_output_state(tmp_path) -> None:
    workspace = AlignmentWorkspace()
    queries = workspace.app.queries

    assert queries.workspace.active_mouse_root_path() is None
    assert not queries.workspace.mouse_root_loaded()
    assert queries.workspace.active_output_root() is None
    assert not queries.workspace.has_output_directory()

    mouse_root = _mouse_root_with_probe()
    output_root = tmp_path / "results"
    output_directory = output_root / "rec" / "probe"
    workspace.data_context.mouse_root = mouse_root
    workspace.document.set_output_root(output_root)
    workspace.document.set_output_directory(output_directory)

    assert queries.workspace.active_mouse_root_path() == mouse_root.root
    assert queries.workspace.mouse_root_loaded()
    assert queries.workspace.active_output_root() == output_root
    assert queries.workspace.has_output_directory()


def test_commands_clear_histology_context() -> None:
    workspace = AlignmentWorkspace()
    workspace.histology_context.runtime_data = object()

    result = workspace.app.commands.metadata.clear_histology_context()

    assert isinstance(result, Ok)
    assert workspace.histology_context.runtime_data is None


def test_commands_select_probe_metadata_loads_channel_info() -> None:
    ephys_data_service = FakeEphysDataService()
    workspace = AlignmentWorkspace(ephys_data_service=ephys_data_service)
    workspace.data_context.mouse_root = _mouse_root_with_probe()

    result = workspace.app.commands.metadata.select_probe_metadata("rec", "probeA")

    assert isinstance(result, ProbeSelected)
    assert ephys_data_service.loaded_probe is not None
    assert ephys_data_service.loaded_probe.probe_name == "probeA"
    assert result.shanks == ["1/2", "2/2"]
    assert workspace.document.channel_info_loaded
    assert workspace.document.selected_alignment_key == AlignmentKey("rec", "stream", 0)


def test_commands_select_recording_metadata_clears_probe_selection() -> None:
    workspace = AlignmentWorkspace()
    workspace.data_context.mouse_root = _mouse_root_with_probe()

    result = workspace.app.commands.metadata.select_recording_metadata("rec")

    assert isinstance(result, RecordingSelected)
    assert result.probes == ["probeA"]
    assert workspace.document.selected_probe is None


def test_commands_activate_cached_ephys_data_uses_explicit_shank() -> None:
    workspace = AlignmentWorkspace()
    workspace.data_context.mouse_root = _mouse_root_with_probe()
    stream_runtime = workspace.runtime.cache_loaded_stream_data(
        _ephys_stream(),
        workspace.plot_payload_cache_factory,
        shank_idx=1,
    )
    workspace.runtime.clear_active_stream()

    result = workspace.app.commands.load.activate_cached_ephys_data(
        recording_id="rec",
        probe_name="probeA",
        stream_key=("rec", "stream"),
        shank_idx=0,
    )

    assert isinstance(result, CachedEphysDataActivated)
    assert result.stream_runtime is stream_runtime
    assert result.shank_idx == 0
    assert result.probe.shanks == ["1/2", "2/2"]
    assert workspace.runtime.active_stream_runtime is stream_runtime
    assert workspace.runtime.current_stream_key == ("rec", "stream")
    assert stream_runtime.current_shank_idx == 0
    assert workspace.document.data_loaded
    assert workspace.document.selected_alignment_key == AlignmentKey("rec", "stream", 0)


def test_commands_activate_cached_ephys_data_reports_missing_cache() -> None:
    workspace = AlignmentWorkspace()
    workspace.data_context.mouse_root = _mouse_root_with_probe()

    result = workspace.app.commands.load.activate_cached_ephys_data(
        recording_id="rec",
        probe_name="probeA",
        stream_key=("rec", "missing"),
        shank_idx=0,
    )

    assert isinstance(result, Failed)
    assert "Cached stream not found" in result.message
    assert not workspace.document.data_loaded
    assert workspace.runtime.active_stream_runtime is None


def test_commands_activate_cached_ephys_data_failure_does_not_mark_loaded() -> None:
    workspace = AlignmentWorkspace()
    workspace.data_context.mouse_root = _mouse_root_with_probe()
    stream_runtime = workspace.runtime.cache_loaded_stream_data(
        _ephys_stream(),
        workspace.plot_payload_cache_factory,
        shank_idx=1,
    )
    workspace.runtime.clear_active_stream()

    result = workspace.app.commands.load.activate_cached_ephys_data(
        recording_id="rec",
        probe_name="probeA",
        stream_key=("rec", "stream"),
        shank_idx=3,
    )

    assert isinstance(result, Failed)
    assert "Failed to restore cached stream runtime" in result.message
    assert not workspace.document.data_loaded
    assert workspace.runtime.active_stream_runtime is None
    assert workspace.runtime.current_stream_key is None
    assert stream_runtime.current_shank_idx == 1


def test_commands_save_visited_alignment_outputs_batches_active_shanks(
    tmp_path,
) -> None:
    repo = FakeAlignmentRepository()
    output_builder = FakeBatchOutputBuilder()
    derived = FakeDerivedDataService()
    workspace = AlignmentWorkspace()
    workspace.persistence_commands.alignment_repository = repo
    workspace.persistence_commands.output_builder = output_builder
    workspace.persistence_commands.derived_data_service = derived
    workspace.document.output_directory = tmp_path
    workspace.data_context.probe_info = SimpleNamespace(
        recording_id="rec",
        probe_name="probeA",
        ephys_collection="stream",
    )
    active_key = AlignmentKey("rec", "stream", 1)
    active_state = workspace.document.select_alignment_key(active_key)
    active_state.active_alignment = ActiveAlignment(
        feature=np.array([1.0, 2.0]),
        track=np.array([3.0, 4.0]),
    )
    workspace.runtime.active_stream_runtime = FakeStreamRuntime()
    workspace.runtime.active_stream_runtime.shank_runtime_by_idx = {
        1: SimpleNamespace(
            ephysalign="aligner",
            chn_coords=np.array([[10.0, 20.0]]),
        )
    }

    result = workspace.app.commands.persistence.save_visited_alignment_outputs(
        use_docdb=True
    )

    assert isinstance(result, VisitedAlignmentOutputsSaved)
    assert result.saved_count == 1
    assert result.active_choices == active_state.prev_align
    assert list(result.saved_outputs) == [active_key]
    assert list(output_builder.batched_alignments) == [active_key]
    np.testing.assert_allclose(
        output_builder.batched_alignments[active_key][0],
        [[1.0, 2.0, 3.0]],
    )
    np.testing.assert_allclose(
        output_builder.batched_alignments[active_key][1],
        [[10.0, 20.0]],
    )
    assert len(derived.channel_location_calls) == 1
    assert derived.channel_location_calls[0]["ephysalign"] == "aligner"
    np.testing.assert_allclose(
        derived.channel_location_calls[0]["feature"],
        active_state.active_alignment.feature,
    )
    np.testing.assert_allclose(
        derived.channel_location_calls[0]["track"],
        active_state.active_alignment.track,
    )
    assert len(active_state.alignments) == 1
    assert repo.saved_kwargs[0]["use_docdb"]
    assert repo.saved_kwargs[0]["shank_idx"] == 1
    assert repo.saved_kwargs[0]["previous_alignments"] == active_state.alignments


def test_commands_load_previous_alignments_defaults_to_active_shank(tmp_path) -> None:
    repo = FakeAlignmentRepository()
    repo.loaded_alignments = {
        "auto": [[100.0], [200.0]],
        "saved": [[1.0], [2.0]],
    }
    workspace = _workspace_with_probe_state(shank_idx=1, repo=repo)

    result = workspace.app.commands.persistence.load_previous_alignments(
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

    result = workspace.app.commands.persistence.load_previous_alignments(
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

    result = workspace.app.commands.persistence.select_previous_alignment(0)

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

    result = workspace.app.commands.edit.offset_alignment_from_tip(
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


def test_commands_fit_active_alignment_uses_document_pending_lines() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    active_key = AlignmentKey("rec", "stream", 1)
    state = workspace.document.alignment_state_for(active_key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 0.004]),
        np.array([0.010, 0.014]),
        lin_fit=True,
    )
    workspace.document.active_set_pending_reference_lines([1000.0], [11000.0])
    workspace.display_state.edit_settings.set_lin_fit(False)
    workspace.display_state.edit_settings.extend_feature = 7
    ephysalign = FakeEditEphysAlignment()
    shank_runtime = SimpleNamespace(
        shank_idx=1,
        ephysalign=ephysalign,
        chn_depths=np.array([0.0, 4000.0]),
    )
    workspace.runtime.active_stream_runtime = SimpleNamespace(
        shank_runtime_by_idx={1: shank_runtime}
    )

    result = (
        workspace.app.commands.edit.fit_active_alignment_from_pending_reference_lines()
    )

    assert isinstance(result, AlignmentEditApplied)
    assert result.lin_fit is False
    assert len(ephysalign.feature2track_calls) == 1
    depths_track, previous_feature, previous_track = ephysalign.feature2track_calls[0]
    np.testing.assert_allclose(depths_track, [0.010, 0.011, 0.014])
    np.testing.assert_allclose(previous_feature, [0.0, 0.004])
    np.testing.assert_allclose(previous_track, [0.010, 0.014])
    np.testing.assert_allclose(result.alignment.feature, [0.0, 0.001, 0.004])
    assert ephysalign.linear_calls == []


def test_commands_fit_active_alignment_reports_missing_runtime() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)

    result = (
        workspace.app.commands.edit.fit_active_alignment_from_pending_reference_lines()
    )

    assert isinstance(result, Failed)
    assert result.message == "Cannot fit alignment: active shank runtime is not loaded"


def test_commands_offset_active_alignment_uses_display_settings() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    active_key = AlignmentKey("rec", "stream", 1)
    state = workspace.document.alignment_state_for(active_key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
        lin_fit=True,
    )
    workspace.display_state.depth_view.probe_tip_um = 20.0
    workspace.display_state.edit_settings.set_lin_fit(False)

    result = workspace.app.commands.edit.offset_active_alignment_from_tip(
        tip_position_um=120.0,
    )

    assert isinstance(result, AlignmentEditApplied)
    assert result.lin_fit is False
    np.testing.assert_allclose(result.alignment.track, [10.0001, 14.0001])


def test_commands_nudge_active_alignment_respects_channel_depth_bounds() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    active_key = AlignmentKey("rec", "stream", 1)
    state = workspace.document.alignment_state_for(active_key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 0.004]),
        np.array([0.0, 0.004]),
    )
    workspace.runtime.active_stream_runtime = SimpleNamespace(
        shank_runtime_by_idx={
            1: SimpleNamespace(
                shank_idx=1,
                chn_depths=np.array([0.0, 3840.0]),
            )
        }
    )

    applied = workspace.app.commands.edit.nudge_active_alignment_from_tip(
        tip_position_um=0.0,
        track_shift_m=-50 / 1e6,
    )
    blocked = workspace.app.commands.edit.nudge_active_alignment_from_tip(
        tip_position_um=0.0,
        track_shift_m=-500 / 1e6,
    )

    assert isinstance(applied, AlignmentEditApplied)
    np.testing.assert_allclose(applied.alignment.track, [-50e-6, 0.00395])
    assert isinstance(blocked, AlignmentEditNoop)


def test_commands_reset_active_alignment_uses_runtime_and_display_settings() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    active_key = AlignmentKey("rec", "stream", 1)
    state = workspace.document.alignment_state_for(active_key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
    )
    workspace.document.active_set_pending_reference_lines([10.0], [11.0])
    workspace.display_state.edit_settings.set_lin_fit(False)
    workspace.runtime.active_stream_runtime = SimpleNamespace(
        shank_runtime_by_idx={
            1: SimpleNamespace(
                shank_idx=1,
                ephysalign=SimpleNamespace(
                    feature_init=np.array([1.0, 3.0]),
                    track_init=np.array([2.0, 4.0]),
                ),
            )
        }
    )

    result = workspace.app.commands.edit.reset_active_alignment_to_initial()

    assert isinstance(result, AlignmentEditApplied)
    assert result.lin_fit is False
    assert state.pending_reference_lines is None
    np.testing.assert_allclose(result.alignment.feature, [1.0, 3.0])
    np.testing.assert_allclose(result.alignment.track, [2.0, 4.0])


def test_commands_previous_next_alignment_default_to_active_shank() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    active_key = AlignmentKey("rec", "stream", 1)
    state = workspace.document.alignment_state_for(active_key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 4.0]),
        np.array([10.0, 14.0]),
        lin_fit=True,
    )
    workspace.app.commands.edit.offset_alignment_from_tip(
        tip_position_um=100.0,
        probe_tip_um=0.0,
        lin_fit=False,
    )

    previous_result = workspace.app.commands.edit.go_previous_alignment()
    next_result = workspace.app.commands.edit.go_next_alignment()

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

    result = workspace.app.commands.edit.go_next_alignment()

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

    result = workspace.app.commands.edit.reset_alignment_to_initial(
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

    state = queries.workspace.active_shank_selection()

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
    runtime = SessionRuntime(
        active_stream_runtime=stream_runtime,
        stream_cache={("rec", "stream"): stream_runtime},
        current_stream_key=("rec", "stream"),
    )
    queries = AlignmentQueries(
        document=document,
        runtime=runtime,
    )

    assert queries.workspace.is_loaded_stream_shank(("rec", "stream"), 1)


def test_queries_reject_loaded_stream_shank_mismatches() -> None:
    document = AlignmentDocument()
    document.select_alignment_key(AlignmentKey("rec", "stream", 1))
    document.mark_data_loaded(True)
    stream_runtime = SimpleNamespace(
        stream_key=("rec", "stream"),
        current_shank_idx=1,
    )
    runtime = SessionRuntime(
        active_stream_runtime=stream_runtime,
        stream_cache={("rec", "stream"): stream_runtime},
        current_stream_key=("rec", "stream"),
    )
    queries = AlignmentQueries(
        document=document,
        runtime=runtime,
    )

    assert not queries.workspace.is_loaded_stream_shank(("rec", "other-stream"), 1)
    assert not queries.workspace.is_loaded_stream_shank(("rec", "stream"), 0)
    assert not queries.workspace.is_loaded_stream_shank(None, 1)
    document.mark_data_loaded(False)
    assert not queries.workspace.is_loaded_stream_shank(("rec", "stream"), 1)


def test_queries_plan_load_data_delegates_to_runtime_cache_plan() -> None:
    document = AlignmentDocument()
    document.select_alignment_key(AlignmentKey("rec", "stream", 1))
    document.mark_data_loaded(True)
    stream_runtime = SimpleNamespace(
        stream_key=("rec", "stream"),
        current_shank_idx=1,
    )
    runtime = SessionRuntime(
        active_stream_runtime=stream_runtime,
        stream_cache={("rec", "stream"): stream_runtime},
        current_stream_key=("rec", "stream"),
    )
    queries = AlignmentQueries(document=document, runtime=runtime)

    active_plan = queries.workspace.plan_load_data(("rec", "stream"), 1)
    cached_plan = queries.workspace.plan_load_data(("rec", "stream"), 0)
    fresh_plan = queries.workspace.plan_load_data(("rec", "other-stream"), 0)

    assert isinstance(active_plan, LoadDataAlreadyActive)
    assert isinstance(cached_plan, LoadDataCachedStreamAvailable)
    assert cached_plan.cached_shank_idx == 1
    assert isinstance(fresh_plan, LoadDataFreshRequired)


def test_queries_resolve_stream_key_through_data_context() -> None:
    data_context = SimpleNamespace(
        stream_key_for_selection=lambda recording_id, probe_name: (
            recording_id,
            f"{probe_name}.ap",
        )
    )
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SessionRuntime(),
        data_context=data_context,
    )

    assert queries.workspace.stream_key_for_selection("rec", "probeA") == (
        "rec",
        "probeA.ap",
    )


def test_queries_report_histology_loaded_from_context() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SessionRuntime(),
        histology_context=SimpleNamespace(brain_atlas="atlas"),
    )

    assert queries.workspace.histology_data_loaded()


def test_queries_stream_key_resolution_failure_returns_none() -> None:
    def fail(_recording_id, _probe_name):
        raise RuntimeError("missing")

    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SessionRuntime(),
        data_context=SimpleNamespace(stream_key_for_selection=fail),
    )

    assert queries.workspace.stream_key_for_selection("rec", "probeA") is None


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

    state = queries.ephys.active_plot_menu_state()

    assert state.group("image").selected_key == "image.fr"
    assert stream_runtime.calls == [2]


def test_queries_resolve_plot_payload_from_active_runtime_shank() -> None:
    document = AlignmentDocument(selected_shank=1)
    stream_runtime = FakeStreamRuntime()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=stream_runtime),
    )

    payload = queries.ephys.active_plot_payload("image.fr")

    assert payload == {"label": "shank-1"}
    assert stream_runtime.calls == [1]


def test_commands_set_unit_filter_updates_display_state() -> None:
    workspace = AlignmentWorkspace()
    workspace.document.set_selected_shank(1)
    stream_runtime = FakeStreamRuntime()
    workspace.runtime.active_stream_runtime = stream_runtime

    result = workspace.app.commands.edit.set_unit_filter("KS good")

    assert isinstance(result, Ok)
    assert workspace.display_state.unit_filter == "KS good"
    assert workspace.app.queries.ephys.active_unit_filter() == "KS good"
    assert stream_runtime.plot_payload_cache_by_shank[1].filtered_subsets == ["KS good"]
    assert stream_runtime.calls == [1]


def test_commands_set_unit_filter_does_not_require_loaded_runtime() -> None:
    workspace = AlignmentWorkspace()

    result = workspace.app.commands.edit.set_unit_filter("KS good")

    assert isinstance(result, Ok)
    assert workspace.display_state.unit_filter == "KS good"


def test_display_commands_update_app_owned_display_settings() -> None:
    workspace = AlignmentWorkspace()

    assert workspace.app.commands.display.toggle_reference_lines_visible() is False
    assert workspace.app.commands.display.toggle_histology_boundaries_visible() is False
    assert workspace.app.commands.display.toggle_region_annotation_source() == (
        "FranklinPaxinos"
    )
    assert workspace.app.commands.display.set_linear_fit_enabled(False) is False

    assert workspace.display_state.reference_lines_visible is False
    assert workspace.display_state.histology_boundaries_visible is False
    assert workspace.display_state.region_annotation_source == "FranklinPaxinos"
    assert workspace.app.queries.workspace.linear_fit_enabled() is False


def test_queries_return_output_and_alignment_screen_read_models(tmp_path) -> None:
    workspace = AlignmentWorkspace()
    workspace.document.set_output_directory(tmp_path / "session" / "probeA")
    workspace.document.set_selected_shank(2)
    state = workspace.document.select_alignment_key(AlignmentKey("rec1", "streamA", 2))
    state.feature_prev = np.array([0.0, 0.001, 0.002, 0.003])
    state.edit_history.set_current_alignment(
        ActiveAlignment(
            feature=np.array([0.0, 0.001]),
            track=np.array([0.0, 0.0015]),
        )
    )
    state.edit_history.total_idx = 1

    assert workspace.app.queries.workspace.active_plot_export_directory() == (
        tmp_path / "session" / "probeA" / "Plots_Shank_3"
    )
    edit_state = workspace.app.queries.workspace.active_alignment_edit_screen_state()

    assert isinstance(edit_state, ActiveAlignmentEditScreenState)
    assert edit_state.current_idx == 0
    assert edit_state.total_idx == 1
    np.testing.assert_allclose(
        edit_state.previous_feature_positions_um,
        [1000.0, 2000.0],
    )


def test_commands_prepare_loaded_shank_without_histology() -> None:
    workspace = _workspace_with_probe_state(shank_idx=1)
    stream_runtime = workspace.runtime.cache_loaded_stream_data(
        _ephys_stream(),
        workspace.plot_payload_cache_factory,
        shank_idx=0,
    )
    workspace.document.mark_data_loaded(True)

    result = workspace.app.commands.loaded_shank.prepare_loaded_shank(1)

    assert isinstance(result, LoadedShankPrepared)
    assert result.shank_idx == 1
    assert 1 in stream_runtime.shank_runtime_by_idx
    assert result.n_channels == 1
    assert not result.histology_available
    assert result.alignment_choices is None
    assert workspace.document.active_alignment_state is not None
    assert workspace.document.active_alignment_state.active_alignment is None


def test_commands_prepare_loaded_shank_initializes_histology_runtime() -> None:
    runtime_initializer = FakeRuntimeInitializer()
    probe_track_service = FakeProbeTrackService()
    workspace = AlignmentWorkspace(
        alignment_runtime_service=runtime_initializer,
        probe_track_service=probe_track_service,
    )
    workspace.data_context.probe_info = SimpleNamespace(
        recording_id="rec",
        probe_name="probeA",
        ephys_collection="stream",
    )
    workspace.data_context.channel_table = SimpleNamespace(n_shanks=2)
    workspace.document.select_alignment_key(AlignmentKey("rec", "stream", 1))
    workspace.document.mark_data_loaded(True)
    workspace.histology_context.runtime_data = SimpleNamespace(brain_atlas="atlas")
    stream_runtime = workspace.runtime.cache_loaded_stream_data(
        _ephys_stream(),
        workspace.plot_payload_cache_factory,
        shank_idx=1,
    )

    result = workspace.app.commands.loaded_shank.prepare_loaded_shank(1)

    assert isinstance(result, LoadedShankPrepared)
    assert result.histology_available
    assert result.alignment_choices == ["original"]
    assert probe_track_service.calls == [
        {
            "probe": workspace.data_context.probe_info,
            "shank_idx": 1,
            "brain_atlas": "atlas",
        }
    ]
    shank_runtime = stream_runtime.shank_runtime_by_idx[1]
    assert runtime_initializer.calls[0][0] is shank_runtime
    assert runtime_initializer.calls[0][1]["brain_atlas"] == "atlas"
    np.testing.assert_allclose(
        runtime_initializer.calls[0][1]["track_annotations_ras"],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]],
    )
    active_state = workspace.document.active_alignment_state
    assert active_state is not None
    assert active_state.active_alignment is not None
    np.testing.assert_allclose(active_state.active_alignment.feature, [1.0, 2.0])
    np.testing.assert_allclose(active_state.active_alignment.track, [3.0, 4.0])

    second = workspace.app.commands.loaded_shank.prepare_loaded_shank(1)

    assert isinstance(second, LoadedShankPrepared)
    assert len(probe_track_service.calls) == 1
    assert len(runtime_initializer.calls) == 2


def test_prepare_loaded_shank_preserves_explicit_original_selection() -> None:
    runtime_initializer = FakeRuntimeInitializer()
    probe_track_service = FakeProbeTrackService()
    workspace = AlignmentWorkspace(
        alignment_runtime_service=runtime_initializer,
        probe_track_service=probe_track_service,
    )
    workspace.data_context.probe_info = SimpleNamespace(
        recording_id="rec",
        probe_name="probeA",
        ephys_collection="stream",
    )
    workspace.data_context.channel_table = SimpleNamespace(n_shanks=2)
    active_key = AlignmentKey("rec", "stream", 1)
    state = workspace.document.select_alignment_key(active_key)
    state.set_alignments({"saved": [[10.0, 20.0], [30.0, 40.0]]})
    selected = workspace.app.commands.persistence.select_previous_alignment(1)
    assert isinstance(selected, PreviousAlignmentSelected)
    assert selected.choice == "original"
    workspace.document.mark_data_loaded(True)
    workspace.histology_context.runtime_data = SimpleNamespace(brain_atlas="atlas")
    workspace.runtime.cache_loaded_stream_data(
        _ephys_stream(),
        workspace.plot_payload_cache_factory,
        shank_idx=1,
    )

    result = workspace.app.commands.loaded_shank.prepare_loaded_shank(
        1,
        select_default_alignment_if_empty=False,
    )

    assert isinstance(result, LoadedShankPrepared)
    assert result.alignment_choices == ["saved", "original"]
    assert state.feature_prev is None
    assert state.track_prev is None
    assert state.active_alignment is not None
    np.testing.assert_allclose(state.active_alignment.feature, [1.0, 2.0])
    np.testing.assert_allclose(state.active_alignment.track, [3.0, 4.0])
    assert runtime_initializer.calls[0][1]["feature_prev"] is None
    assert runtime_initializer.calls[0][1]["track_prev"] is None


def test_queries_return_default_unit_filter() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    assert queries.ephys.active_unit_filter() == "all"


def test_queries_can_resolve_raw_payload_without_payload_cache() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    state = queries.ephys.active_plot_menu_state(
        previous_selected_keys={"image": "image.raw.raw_ap"},
        raw_image_payloads={"raw_ap": "raw-image"},
    )
    payload = queries.ephys.active_plot_payload(
        "image.raw.raw_ap",
        raw_image_payloads={"raw_ap": "raw-image"},
    )

    assert state.group("image").selected_key == "image.raw.raw_ap"
    assert payload == "raw-image"


def test_queries_fail_closed_without_payload_cache_or_raw_payloads() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    state = queries.ephys.active_plot_menu_state()

    assert not state.group("image").enabled
    assert queries.ephys.active_plot_payload("image.fr") is None


def test_queries_return_active_in_brain_depths_from_runtime_payload_cache() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(selected_shank=1),
        runtime=SimpleNamespace(active_stream_runtime=FakeStreamRuntime()),
    )

    np.testing.assert_array_equal(
        queries.ephys.active_in_brain_depths_um(),
        [20.0, 40.0],
    )


def test_queries_prepare_shank_plot_data_state_filters_runtime_payload_cache() -> None:
    document = AlignmentDocument(selected_shank=1)
    stream_runtime = FakeStreamRuntime()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=stream_runtime),
        display_state=AlignmentDisplayState(unit_filter="KS good"),
    )

    state = queries.ephys.prepare_active_shank_plot_data_state()

    assert state is not None
    assert state.shank_idx == 1
    assert state.unit_filter == "KS good"
    assert state.channel_min_um == 5.0
    assert state.channel_max_um == 200.0
    assert stream_runtime.plot_payload_cache_by_shank[1].filtered_subsets == ["KS good"]
    assert stream_runtime.plot_payload_cache_by_shank[1].in_brain_depths_um is None


def test_queries_build_active_shank_screen_state_from_runtime_menus() -> None:
    document = AlignmentDocument(data_loaded=True)
    document.select_alignment_key(AlignmentKey("rec", "stream", 2))
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=FakeStreamRuntime()),
        display_state=AlignmentDisplayState(unit_filter="KS good"),
    )

    state = queries.active_shank.active_shank_screen_state(
        preserve_plot_selection=True,
        previous_ephys_plot_keys={"image": "image.raw.raw_ap"},
        raw_image_payloads={"raw_ap": "raw"},
        previous_slice_selection=SliceSelection("slice_data", "missing"),
        offline=True,
    )

    assert state.shank_idx == 2
    assert state.shank_id == 3
    assert state.alignment_key == AlignmentKey("rec", "stream", 2)
    assert state.data_loaded
    assert state.preserve_plot_selection
    assert state.unit_filter == "KS good"
    assert state.plot_menu.group("image").selected_key == "image.raw.raw_ap"
    assert state.slice_menu is None


def test_queries_prepare_active_shank_screen_state_materializes_runtime() -> None:
    document = AlignmentDocument(data_loaded=True)
    document.select_alignment_key(AlignmentKey("rec", "stream", 2))
    stream_runtime = FakeStreamRuntime()
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=stream_runtime),
        display_state=AlignmentDisplayState(unit_filter="KS good"),
    )

    prepared = queries.active_shank.prepare_active_shank_screen_state(
        histology_available=False,
        preserve_plot_selection=True,
        previous_ephys_plot_keys={"image": "image.raw.raw_ap"},
        raw_image_payloads={"raw_ap": "raw"},
        previous_slice_selection=SliceSelection("slice_data", "missing"),
        offline=True,
    )

    assert not prepared.missing_plot_data
    assert not prepared.missing_required_slice_data
    assert prepared.plot_data is not None
    assert prepared.plot_data.shank_idx == 2
    assert prepared.screen is not None
    assert prepared.screen.plot_menu.group("image").selected_key == "image.raw.raw_ap"
    assert stream_runtime.plot_payload_cache_by_shank[2].filtered_subsets == ["KS good"]


def test_queries_prepare_active_shank_screen_state_reports_required_slice_gap() -> None:
    document = AlignmentDocument(data_loaded=True)
    document.select_alignment_key(AlignmentKey("rec", "stream", 2))
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(active_stream_runtime=FakeStreamRuntime()),
        display_state=AlignmentDisplayState(unit_filter="KS good"),
    )

    prepared = queries.active_shank.prepare_active_shank_screen_state(
        histology_available=True,
        preserve_plot_selection=True,
        previous_ephys_plot_keys=None,
        raw_image_payloads=None,
        previous_slice_selection=None,
        offline=True,
    )

    assert not prepared.missing_plot_data
    assert prepared.missing_required_slice_data
    assert prepared.screen is None


def test_queries_build_cluster_detail_from_runtime_payload_cache() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(selected_shank=1),
        runtime=SimpleNamespace(active_stream_runtime=FakeStreamRuntime()),
    )

    detail = queries.ephys.active_cluster_detail(3)

    assert detail is not None
    assert detail.cluster_no == 13
    np.testing.assert_array_equal(detail.autocorr, [3, 4, 5])
    np.testing.assert_array_equal(detail.t_autocorr, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(detail.template_waveform, [3.5, 4.5])
    np.testing.assert_array_equal(detail.t_template, [0.0, 0.5])


def test_queries_cluster_detail_fails_closed_without_active_runtime() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    assert queries.ephys.active_cluster_detail(3) is None


def test_queries_active_session_notes_returns_loaded_stream_notes() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(
            active_stream_runtime=SimpleNamespace(
                stream=SimpleNamespace(session_notes="session notes")
            )
        ),
    )

    assert queries.ephys.active_session_notes() == "session notes"


def test_queries_active_session_notes_fails_closed_without_runtime() -> None:
    queries = AlignmentQueries(
        document=AlignmentDocument(),
        runtime=SimpleNamespace(active_stream_runtime=None),
    )

    assert queries.ephys.active_session_notes() == ""


def test_queries_active_histology_region_id_reads_active_shank_runtime() -> None:
    document = AlignmentDocument()
    document.select_alignment_key(AlignmentKey("rec", "stream", 1))
    queries = AlignmentQueries(
        document=document,
        runtime=SimpleNamespace(
            active_stream_runtime=SimpleNamespace(
                shank_runtime_by_idx={
                    1: SimpleNamespace(
                        ephysalign=SimpleNamespace(
                            region_id=np.array([[10], [42], [84]])
                        )
                    )
                }
            )
        ),
    )

    assert queries.alignment_render.active_histology_region_id(1) == 42
    assert queries.alignment_render.active_histology_region_id(9) is None


def test_queries_build_active_alignment_render_state_from_document_runtime() -> None:
    document = AlignmentDocument()
    key = AlignmentKey("rec", "stream", 1)
    state = document.select_alignment_key(key)
    active_alignment = ActiveAlignment(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
    )
    state.active_alignment = active_alignment
    display_state = AlignmentDisplayState(region_annotation_source="FranklinPaxinos")
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

    render_state = queries.alignment_render.active_alignment_render_state()

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
    assert derived.histology_kwargs["region_annotation_source"] == "FranklinPaxinos"
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

    histology_state = queries.alignment_render.active_histology_panel_state(
        probe_tip_um=0.0,
        probe_top_um=3840.0,
        probe_extra_um=100.0,
    )
    scale_state = queries.alignment_render.active_scale_factor_state(
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


def test_queries_build_nearby_boundary_state_from_runtime_cache() -> None:
    document = AlignmentDocument()
    key = AlignmentKey("rec", "stream", 1)
    state = document.select_alignment_key(key)
    state.active_alignment = ActiveAlignment(
        np.array([0.0, 0.005]),
        np.array([0.0, 0.004]),
    )
    derived = FakeDerivedDataService()
    shank_runtime = SimpleNamespace(
        ephysalign="aligner",
        nearby_boundaries=None,
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

    first_state = queries.alignment_render.active_nearby_boundary_state(
        probe_tip_um=0.0,
        probe_top_um=3840.0,
        probe_extra_um=100.0,
        allen="allen-table",
        brain_atlas="atlas",
    )
    second_state = queries.alignment_render.active_nearby_boundary_state(
        probe_tip_um=0.0,
        probe_top_um=3840.0,
        probe_extra_um=100.0,
        allen="allen-table",
        brain_atlas="atlas",
    )

    assert first_state is not None
    assert first_state.key == key
    np.testing.assert_array_equal(first_state.x, [1.0, 2.0])
    np.testing.assert_array_equal(first_state.y, [3.0, 4.0])
    assert first_state.colours == ["red", "blue"]
    np.testing.assert_array_equal(first_state.parent_x, [5.0, 6.0])
    np.testing.assert_array_equal(first_state.parent_y, [7.0, 8.0])
    assert first_state.parent_colours == ["pink", "cyan"]
    assert second_state is not None
    assert shank_runtime.nearby_boundaries is derived.nearby_boundaries
    assert len(derived.nearby_kwargs) == 1
    assert derived.nearby_kwargs[0]["ephysalign"] == "aligner"
    assert derived.nearby_kwargs[0]["allen"] == "allen-table"
    assert derived.nearby_kwargs[0]["brain_atlas"] == "atlas"


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
                shank_runtime_by_idx={1: SimpleNamespace(ephysalign=ephysalign)}
            )
        ),
    )

    fit_state = queries.alignment_render.active_fit_plot_state(
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

    assert queries.alignment_render.active_alignment_render_state() is None


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

    first = queries.slices.ensure_active_slice_data_state()
    second = queries.slices.ensure_active_slice_data_state()

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
    assert queries.slices.active_slice_data_by_attr()["slice_data"] is first.slice_data


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

    restored = queries.slices.active_slice_menu_state(
        offline=True,
        previous_selection=SliceSelection("slice_data", "histology_registration"),
    )
    fallback = queries.slices.active_slice_menu_state(
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

    render_state = queries.slices.active_slice_render_state(
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
    track_interpolation = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [2.0, 0.0, 4.0]])
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

    first = queries.slices.active_perpendicular_slice_state("ccf")
    second = queries.slices.active_perpendicular_slice_state("ccf")

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
