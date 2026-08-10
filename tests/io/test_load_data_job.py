"""Tests for the Qt-free fresh load-data job boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.application.workflow import Failed
from ephys_alignment_gui.io.load_data_job import (
    LoadDataJob,
    LoadDataJobCompleted,
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

    def load(self, shank_idx: int):
        self.calls.append(shank_idx)
        if self.error is not None:
            raise self.error
        return self.result


class FakeHistologyRuntimeLoader:
    def __init__(self, result: Any | None = None):
        self.result = result or HistologyDataLoaded()
        self.calls = 0

    def load_if_needed(self):
        self.calls += 1
        return self.result


def _loaded_ephys(ephys_dir: Path | None) -> SimpleNamespace:
    return SimpleNamespace(stream=SimpleNamespace(ephys_dir=ephys_dir))


def test_load_data_job_runs_ephys_then_histology() -> None:
    ephys = _loaded_ephys(Path("/tmp/ephys"))
    ephys_loader = FakeEphysStreamLoader(result=ephys)
    histology_loader = FakeHistologyRuntimeLoader()
    job = LoadDataJob(ephys_loader, histology_loader)

    result = job.run(LoadDataJobRequest(shank_idx=2))

    assert isinstance(result, LoadDataJobCompleted)
    assert result.ephys is ephys
    assert isinstance(result.histology, HistologyDataLoaded)
    assert ephys_loader.calls == [2]
    assert histology_loader.calls == 1


def test_load_data_job_returns_failed_when_ephys_load_raises() -> None:
    ephys_loader = FakeEphysStreamLoader(error=RuntimeError("boom"))
    histology_loader = FakeHistologyRuntimeLoader()
    job = LoadDataJob(ephys_loader, histology_loader)

    result = job.run(LoadDataJobRequest(shank_idx=1))

    assert isinstance(result, Failed)
    assert result.message == "Failed to load ephys data: boom"
    assert ephys_loader.calls == [1]
    assert histology_loader.calls == 0


def test_load_data_job_returns_failed_when_ephys_dir_is_missing() -> None:
    ephys_loader = FakeEphysStreamLoader(result=_loaded_ephys(None))
    histology_loader = FakeHistologyRuntimeLoader()
    job = LoadDataJob(ephys_loader, histology_loader)

    result = job.run(LoadDataJobRequest(shank_idx=1))

    assert isinstance(result, Failed)
    assert result.message == "Failed to load ephys data"
    assert histology_loader.calls == 0


def test_load_data_job_keeps_histology_unavailable_nonfatal() -> None:
    histology = HistologyDataUnavailable("no histology")
    job = LoadDataJob(
        FakeEphysStreamLoader(),
        FakeHistologyRuntimeLoader(result=histology),
    )

    result = job.run(LoadDataJobRequest(shank_idx=0))

    assert isinstance(result, LoadDataJobCompleted)
    assert result.histology is histology
