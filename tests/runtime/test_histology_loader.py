"""Tests for subject-level histology runtime loading."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from ephys_alignment_gui.runtime.histology_loader import (
    HistologyDataAlreadyLoaded,
    HistologyDataLoaded,
    HistologyDataUnavailable,
    HistologyRuntimeLoader,
)


class FakeHistologyService:
    def __init__(self, data=None, exc: Exception | None = None) -> None:
        self.data = data or _histology_data("atlas")
        self.exc = exc
        self.calls: list[object] = []

    def load(self, mouse_root):
        self.calls.append(mouse_root)
        if self.exc is not None:
            raise self.exc
        return self.data


class BlockingHistologyService:
    def __init__(self, data=None) -> None:
        self.data = data or _histology_data("atlas")
        self.calls: list[object] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def load(self, mouse_root):
        self.calls.append(mouse_root)
        self.started.set()
        self.release.wait(timeout=2)
        return self.data


def _histology_data(brain_atlas):
    return SimpleNamespace(
        brain_atlas=brain_atlas,
        histology_images={},
        lazy_channel_paths={},
    )


def test_load_if_needed_skips_when_histology_is_already_loaded() -> None:
    data_context = SimpleNamespace(mouse_root=object())
    histology_context = SimpleNamespace(
        brain_atlas="atlas",
        set=lambda _data: None,
    )
    service = FakeHistologyService()
    loader = HistologyRuntimeLoader(data_context, service, histology_context)

    result = loader.load_if_needed()

    assert isinstance(result, HistologyDataAlreadyLoaded)
    assert service.calls == []


def test_load_if_needed_loads_and_stores_histology_data() -> None:
    mouse_root = object()
    histology_data = _histology_data("atlas")
    stored: list[object] = []
    histology_context = SimpleNamespace(
        brain_atlas=None,
        set=stored.append,
    )
    service = FakeHistologyService(data=histology_data)
    loader = HistologyRuntimeLoader(
        SimpleNamespace(mouse_root=mouse_root),
        service,
        histology_context,
    )

    result = loader.load_if_needed()

    assert isinstance(result, HistologyDataLoaded)
    assert service.calls == [mouse_root]
    assert stored == [histology_data]


def test_load_if_needed_reports_missing_mouse_root_as_non_fatal() -> None:
    histology_context = SimpleNamespace(
        brain_atlas=None,
        set=lambda _data: None,
    )
    service = FakeHistologyService()
    loader = HistologyRuntimeLoader(
        SimpleNamespace(mouse_root=None),
        service,
        histology_context,
    )

    result = loader.load_if_needed()

    assert isinstance(result, HistologyDataUnavailable)
    assert result.message == "Failed to load atlas/histology: No mouse root loaded"
    assert service.calls == []


def test_load_if_needed_reports_service_exception_as_non_fatal() -> None:
    histology_context = SimpleNamespace(
        brain_atlas=None,
        set=lambda _data: None,
    )
    service = FakeHistologyService(exc=RuntimeError("boom"))
    loader = HistologyRuntimeLoader(
        SimpleNamespace(mouse_root=object()),
        service,
        histology_context,
    )

    result = loader.load_if_needed()

    assert isinstance(result, HistologyDataUnavailable)
    assert result.message == "Failed to load atlas/histology: boom"


def test_load_for_mouse_root_joins_inflight_warmup() -> None:
    mouse_root = SimpleNamespace(root="/tmp/mouse")
    service = BlockingHistologyService()
    loader = HistologyRuntimeLoader(
        SimpleNamespace(mouse_root=mouse_root),
        service,
        SimpleNamespace(brain_atlas=None, set=lambda _data: None),
    )
    result_holder: list[object] = []

    assert loader.start_warmup_for_mouse_root(mouse_root)
    assert service.started.wait(timeout=2)

    waiter = threading.Thread(
        target=lambda: result_holder.append(
            loader.load_for_mouse_root(mouse_root, store=False)
        )
    )
    waiter.start()

    assert service.calls == [mouse_root]
    service.release.set()
    waiter.join(timeout=2)

    assert len(result_holder) == 1
    assert isinstance(result_holder[0], HistologyDataLoaded)
    assert service.calls == [mouse_root]


def test_load_for_mouse_root_reuses_completed_warmup() -> None:
    mouse_root = SimpleNamespace(root="/tmp/mouse")
    service = BlockingHistologyService()
    loader = HistologyRuntimeLoader(
        SimpleNamespace(mouse_root=mouse_root),
        service,
        SimpleNamespace(brain_atlas=None, set=lambda _data: None),
    )

    assert loader.start_warmup_for_mouse_root(mouse_root)
    assert service.started.wait(timeout=2)
    service.release.set()
    first = loader.load_for_mouse_root(mouse_root, store=False)
    second = loader.load_for_mouse_root(mouse_root, store=False)

    assert isinstance(first, HistologyDataLoaded)
    assert second is first
    assert service.calls == [mouse_root]


def test_clear_warmup_results_discards_completed_warmup() -> None:
    mouse_root = SimpleNamespace(root="/tmp/mouse")
    service = FakeHistologyService()
    loader = HistologyRuntimeLoader(
        SimpleNamespace(mouse_root=mouse_root),
        service,
        SimpleNamespace(brain_atlas=None, set=lambda _data: None),
    )

    assert loader.start_warmup_for_mouse_root(mouse_root)
    warm_result = loader.load_for_mouse_root(mouse_root, store=False)
    loader.clear_warmup_results()
    fresh_result = loader.load_for_mouse_root(mouse_root, store=False)

    assert isinstance(warm_result, HistologyDataLoaded)
    assert isinstance(fresh_result, HistologyDataLoaded)
    assert service.calls == [mouse_root, mouse_root]
