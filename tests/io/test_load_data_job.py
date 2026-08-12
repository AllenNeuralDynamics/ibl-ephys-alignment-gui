"""Tests for the Qt-free fresh load-data job boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.io.load_data_job import (
    LoadDataCancelToken,
    LoadDataJob,
    LoadDataJobCancelled,
    LoadDataJobCompleted,
    LoadDataJobProgress,
    LoadDataJobRequest,
)
from ephys_alignment_gui.runtime.histology_loader import (
    HistologyDataLoaded,
    HistologyDataUnavailable,
)


class FakeEphysStreamLoader:
    def __init__(self, result: Any | None = None, *, error: Exception | None = None):
        self.result = result or _loaded_ephys(Path("/tmp/ephys"))
        self.error = error
        self.calls: list[int] = []

    def load_target(self, target):
        self.calls.append(target.shank_idx)
        if self.error is not None:
            raise self.error
        return self.result


class FakeHistologyRuntimeLoader:
    def __init__(self, result: Any | None = None):
        self.result = result or HistologyDataLoaded()
        self.calls = 0

    def load_for_mouse_root(self, mouse_root, *, store: bool):
        self.calls += 1
        self.mouse_root = mouse_root
        self.store = store
        return self.result


def _loaded_ephys(ephys_dir: Path | None) -> SimpleNamespace:
    return SimpleNamespace(stream=SimpleNamespace(ephys_dir=ephys_dir))


def _target(shank_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        recording_id="rec",
        probe_name="probeA",
        stream_key=("rec", "stream"),
        shank_idx=shank_idx,
        mouse_root="mouse-root",
    )


def test_load_data_job_runs_ephys_then_histology() -> None:
    ephys = _loaded_ephys(Path("/tmp/ephys"))
    ephys_loader = FakeEphysStreamLoader(result=ephys)
    histology_loader = FakeHistologyRuntimeLoader()
    job = LoadDataJob(ephys_loader, histology_loader)

    result = job.run(LoadDataJobRequest(_target(shank_idx=2)))

    assert isinstance(result, LoadDataJobCompleted)
    assert result.target.shank_idx == 2
    assert result.ephys is ephys
    assert isinstance(result.histology, HistologyDataLoaded)
    assert result.warnings == ()
    assert ephys_loader.calls == [2]
    assert histology_loader.calls == 1
    assert histology_loader.mouse_root == "mouse-root"
    assert histology_loader.store is False


def test_load_data_job_reports_progress() -> None:
    progress: list[LoadDataJobProgress] = []
    job = LoadDataJob(FakeEphysStreamLoader(), FakeHistologyRuntimeLoader())

    result = job.run(LoadDataJobRequest(_target()), progress=progress.append)

    assert isinstance(result, LoadDataJobCompleted)
    assert [(event.phase, event.status) for event in progress] == [
        ("ephys", "started"),
        ("ephys", "completed"),
        ("histology", "started"),
        ("histology", "completed"),
        ("complete", "completed"),
    ]


def test_load_data_job_returns_failed_when_ephys_load_raises() -> None:
    ephys_loader = FakeEphysStreamLoader(error=RuntimeError("boom"))
    histology_loader = FakeHistologyRuntimeLoader()
    job = LoadDataJob(ephys_loader, histology_loader)

    result = job.run(LoadDataJobRequest(_target(shank_idx=1)))

    assert isinstance(result, Failed)
    assert result.message == "Failed to load ephys data: boom"
    assert ephys_loader.calls == [1]
    assert histology_loader.calls == 0


def test_load_data_job_returns_failed_when_ephys_dir_is_missing() -> None:
    ephys_loader = FakeEphysStreamLoader(result=_loaded_ephys(None))
    histology_loader = FakeHistologyRuntimeLoader()
    job = LoadDataJob(ephys_loader, histology_loader)

    result = job.run(LoadDataJobRequest(_target(shank_idx=1)))

    assert isinstance(result, Failed)
    assert result.message == "Failed to load ephys data"
    assert histology_loader.calls == 0


def test_load_data_job_keeps_histology_unavailable_nonfatal() -> None:
    histology = HistologyDataUnavailable("no histology")
    job = LoadDataJob(
        FakeEphysStreamLoader(),
        FakeHistologyRuntimeLoader(result=histology),
    )

    result = job.run(LoadDataJobRequest(_target(shank_idx=0)))

    assert isinstance(result, LoadDataJobCompleted)
    assert result.histology is histology
    assert len(result.warnings) == 1
    assert result.warnings[0].message == "no histology"


def test_load_data_job_cancels_before_ephys_starts() -> None:
    token = LoadDataCancelToken()
    token.cancel("new probe selected")
    ephys_loader = FakeEphysStreamLoader()
    histology_loader = FakeHistologyRuntimeLoader()
    progress: list[LoadDataJobProgress] = []
    job = LoadDataJob(ephys_loader, histology_loader)

    result = job.run(
        LoadDataJobRequest(_target(shank_idx=0)),
        progress=progress.append,
        cancel_token=token,
    )

    assert isinstance(result, LoadDataJobCancelled)
    assert result.reason == "new probe selected"
    assert ephys_loader.calls == []
    assert histology_loader.calls == 0
    assert [(event.phase, event.status) for event in progress] == [
        ("cancelled", "cancelled")
    ]


def test_load_data_job_cancels_after_ephys_completes() -> None:
    token = LoadDataCancelToken()
    ephys_loader = FakeEphysStreamLoader()
    histology_loader = FakeHistologyRuntimeLoader()

    def cancel_after_ephys(event: LoadDataJobProgress) -> None:
        if event.phase == "ephys" and event.status == "completed":
            token.cancel("target no longer foreground")

    job = LoadDataJob(ephys_loader, histology_loader)

    result = job.run(
        LoadDataJobRequest(_target(shank_idx=0)),
        progress=cancel_after_ephys,
        cancel_token=token,
    )

    assert isinstance(result, LoadDataJobCancelled)
    assert result.reason == "target no longer foreground"
    assert ephys_loader.calls == [0]
    assert histology_loader.calls == 0
