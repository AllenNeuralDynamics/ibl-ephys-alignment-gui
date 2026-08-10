"""Tests for declarative ephys plot registry."""

from __future__ import annotations

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

    def cached(self, method: str, args: tuple = ()) -> Any:
        self.calls.append((method, args))
        if method == "get_fr_img":
            return "fr-img"
        if method == "get_fr_amp_data_line":
            return "line-fr", "line-amp"
        if method == "get_rms_data_img_probe":
            return "image-rms", "probe-rms"
        if method == "get_lfp_correlation_data_img":
            return {"theta": "corr-img"}
        if method == "get_passive_events":
            return self.passive_events
        if method == "get_lfp_spectrum_data":
            return "lfp-img", {"0 - 4 Hz": "probe-lfp"}
        if method == "get_rfmap_data":
            return {"left": "rfmap"}, "bounds"
        return method


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


def test_resolve_plot_payload_uses_cached_method_args_and_index() -> None:
    payload_cache = FakePayloadCache()

    assert resolve_plot_payload(payload_cache, "probe.rms_ap") == "probe-rms"

    assert payload_cache.calls == [("get_rms_data_img_probe", ("AP",))]


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


def test_mapping_plot_specs_resolve_existing_mapping_payloads() -> None:
    specs = mapping_plot_specs(
        parent_key="image.raw",
        menu="image",
        renderer="image",
        payloads={"raw_ap": "raw-img"},
    )

    assert [spec.key for spec in specs] == ["image.raw.raw_ap"]
    assert resolve_plot_payload(None, specs[0]) == "raw-img"
