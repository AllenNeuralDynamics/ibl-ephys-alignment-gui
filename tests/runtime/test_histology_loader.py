"""Tests for subject-level histology runtime loading."""

from __future__ import annotations

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
