"""Tests for declarative ephys plot registry."""

from __future__ import annotations

import logging
from typing import Any

from ephys_alignment_gui.plotting.registry import (
    available_plot_specs_for_menu,
    default_plot_spec,
    mapping_plot_specs,
    plot_spec,
    plot_specs_for_menu,
    resolve_plot_bounds,
    resolve_plot_payload,
)


class FakePayloadCache:
    def __init__(self, passive_events: dict[str, Any] | None = None) -> None:
        self.calls = []
        self.passive_events = passive_events if passive_events is not None else {}

    def get_or_build_payload(self, key: tuple[Any, ...], build):
        self.calls.append(key)
        return build()

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

    def get_rms_data_img_probe(self, _format: str) -> Any:
        return "image-rms", "probe-rms"

    def get_lfp_correlation_data_img(self) -> Any:
        return {"theta": "corr-img"}

    def get_passive_events(self) -> Any:
        return self.passive_events

    def get_lfp_spectrum_data(self, _format: str) -> Any:
        return "lfp-img", {"0 - 4 Hz": "probe-lfp"}

    def get_rfmap_data(self) -> Any:
        return {"left": "rfmap"}, "bounds"


class FakeOptionalDependencyMissingPayloadCache(FakePayloadCache):
    def get_passive_events(self) -> Any:
        raise ModuleNotFoundError(
            "No module named 'brainbox'",
            name="brainbox",
        )


class FakeDataEntry:
    def __init__(self, exists: bool) -> None:
        self.exists = exists

    def __getitem__(self, key: str) -> Any:
        if key == "exists":
            return self.exists
        raise KeyError(key)


def test_default_specs_are_registered_for_static_menus() -> None:
    assert default_plot_spec("image").key == "image.fr"
    assert default_plot_spec("line").key == "line.fr"
    assert default_plot_spec("probe").key == "probe.rms_ap"


def test_specs_are_grouped_by_menu_in_order() -> None:
    image_keys = [spec.key for spec in plot_specs_for_menu("image")]

    assert image_keys[:3] == [
        "image.fr",
        "scatter.amplitude",
        "image.spike_correlation",
    ]
    assert "line.fr" not in image_keys


def test_resolve_plot_payload_uses_typed_cache_key_and_index() -> None:
    payload_cache = FakePayloadCache()

    assert resolve_plot_payload(payload_cache, "probe.rms_ap") == "probe-rms"

    assert payload_cache.calls == [("rms_data_img_probe", "AP")]


def test_plot_spec_contains_renderer_for_view_dispatch() -> None:
    assert plot_spec("scatter.cluster_amp").renderer == "scatter"
    assert plot_spec("image.fr").renderer == "image"


def test_available_plot_specs_include_present_dynamic_image_payloads() -> None:
    payload_cache = FakePayloadCache()

    specs = available_plot_specs_for_menu(payload_cache, "image")
    keys = [spec.key for spec in specs]

    assert "image.lfp_correlation.theta" in keys
    assert not any(key.startswith("image.passive_event.") for key in keys)


def test_available_plot_specs_hide_static_entries_for_missing_datasets() -> None:
    payload_cache = FakePayloadCache()
    payload_cache.data = {
        "spikes": {"exists": False},
        "clusters": {"exists": True},
        "rms_AP": {"exists": False},
        "rms_AP_main": {"exists": False},
        "rms_LF": FakeDataEntry(True),
        "rms_LF_main": {"exists": False},
        "psd_lf": {"exists": False},
        "psd_lf_main": {"exists": False},
    }

    image_keys = [spec.key for spec in available_plot_specs_for_menu(payload_cache, "image")]
    probe_keys = [spec.key for spec in available_plot_specs_for_menu(payload_cache, "probe")]
    line_keys = [spec.key for spec in available_plot_specs_for_menu(payload_cache, "line")]

    assert "image.fr" not in image_keys
    assert "image.rms_ap" not in image_keys
    assert "image.rms_lfp" in image_keys
    assert "probe.rms_lfp" in probe_keys
    assert line_keys == []


def test_available_plot_specs_include_dynamic_probe_payloads_and_bounds() -> None:
    payload_cache = FakePayloadCache()

    specs = available_plot_specs_for_menu(payload_cache, "probe")
    spec_by_key = {spec.key: spec for spec in specs}

    assert resolve_plot_payload(
        payload_cache,
        spec_by_key["probe.lfp_spectrum.0 - 4 Hz"],
    ) == "probe-lfp"
    assert resolve_plot_payload(payload_cache, spec_by_key["probe.rfmap.left"]) == "rfmap"
    assert resolve_plot_bounds(payload_cache, spec_by_key["probe.rfmap.left"]) == "bounds"


def test_available_plot_specs_include_passive_events_when_present() -> None:
    payload_cache = FakePayloadCache(passive_events={"stim": "stim-img"})

    specs = available_plot_specs_for_menu(payload_cache, "image")
    spec_by_key = {spec.key: spec for spec in specs}

    assert resolve_plot_payload(
        payload_cache,
        spec_by_key["image.passive_event.stim"],
    ) == "stim-img"


def test_available_plot_specs_logs_missing_optional_dependency_once(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="ephys_alignment_gui.plotting.registry")
    payload_cache = FakeOptionalDependencyMissingPayloadCache()

    assert available_plot_specs_for_menu(payload_cache, "image")
    assert available_plot_specs_for_menu(payload_cache, "image")

    messages = [record.getMessage() for record in caplog.records]
    optional_messages = [
        message for message in messages if "image.passive_event" in message
    ]
    assert optional_messages == [
        "Skipping unavailable dynamic plot menu entries for image.passive_event: "
        "optional dependency 'brainbox' is not installed"
    ]
    assert "Traceback" not in caplog.text


def test_mapping_plot_specs_resolve_existing_mapping_payloads() -> None:
    specs = mapping_plot_specs(
        parent_key="image.raw",
        menu="image",
        renderer="image",
        payloads={"raw_ap": "raw-img"},
    )

    assert [spec.key for spec in specs] == ["image.raw.raw_ap"]
    assert resolve_plot_payload(None, specs[0]) == "raw-img"
