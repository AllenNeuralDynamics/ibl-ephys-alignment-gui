"""Tests for Qt-free plot payload cache warmup jobs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.plotting.payload_warmup import (
    PlotPayloadWarmupJob,
    PlotPayloadWarmupRequest,
)


class FakePayloadCache:
    def __init__(self) -> None:
        self.data = {
            "spikes": {"exists": True},
            "clusters": {"exists": True},
            "rms_AP": {"exists": False},
            "rms_AP_main": {"exists": False},
            "rms_LF": {"exists": False},
            "rms_LF_main": {"exists": False},
            "psd_lf": {"exists": False},
            "psd_lf_main": {"exists": False},
        }
        self.filtered_subsets: list[str] = []
        self.cache: dict[Any, Any] = {}

    def get_or_build_payload(self, key: tuple[Any, ...], build):
        if key not in self.cache:
            self.cache[key] = build()
        return self.cache[key]

    def filter_units(self, subset: str) -> None:
        self.filtered_subsets.append(subset)

    def get_fr_img(self) -> Any:
        return "fr-img"

    def get_depth_data_scatter(self) -> Any:
        return "depth-scatter"

    def get_spike_correlation_data_img(self) -> Any:
        return "spike-corr-img"

    def get_fr_p2t_data_scatter(self) -> Any:
        return "cluster-fr", "cluster-duration", "cluster-amp"

    def get_fr_amp_data_line(self) -> Any:
        return "line-fr", "line-amp"

    def get_lfp_correlation_keys(self) -> tuple[str, ...]:
        return ()

    def get_passive_event_keys(self) -> tuple[str, ...]:
        return ()

    def get_lfp_spectrum_probe_keys(self, _format: str) -> tuple[str, ...]:
        return ()

    def get_rfmap_keys(self) -> tuple[str, ...]:
        return ()


class FakePayloadCacheFactory:
    def __init__(self) -> None:
        self.payload_cache = FakePayloadCache()
        self.calls: list[tuple[Any, int]] = []

    def build_for_stream(self, stream, shank_idx: int):
        self.calls.append((stream, shank_idx))
        return self.payload_cache


def test_plot_payload_warmup_filters_menu_availability_and_safe_payloads() -> None:
    factory = FakePayloadCacheFactory()
    stream = SimpleNamespace(name="stream")
    request = PlotPayloadWarmupRequest(
        stream_key=("rec", "probeA"),
        stream=stream,
        shank_idx=0,
        unit_filter="unitrefine_neural",
    )

    result = PlotPayloadWarmupJob(factory).run(request)

    assert factory.calls == [(stream, 0)]
    assert result.payload_cache is factory.payload_cache
    assert result.unit_filter == "unitrefine_neural"
    assert result.warmed_spec_keys == ("line.fr",)
    assert factory.payload_cache.filtered_subsets == ["unitrefine_neural"]
    assert ("available_plot_specs_for_menu", "image") in factory.payload_cache.cache
    assert ("available_plot_specs_for_menu", "line") in factory.payload_cache.cache
    assert ("available_plot_specs_for_menu", "probe") in factory.payload_cache.cache
    assert ("fr_amp_data_line",) in factory.payload_cache.cache
