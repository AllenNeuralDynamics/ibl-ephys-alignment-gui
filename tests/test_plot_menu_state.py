"""Tests for GUI-agnostic plot menu state."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.plot_menu_state import build_plot_menu_state


class FakePlotData:
    def __init__(self, *, spikes: bool = True, rms_ap: bool = True) -> None:
        self.data = {
            "spikes": {"exists": spikes},
            "clusters": {"exists": spikes},
            "rms_AP": {"exists": rms_ap},
            "rms_AP_main": {"exists": False},
            "rms_LF": {"exists": False},
            "rms_LF_main": {"exists": False},
            "psd_lf": {"exists": False},
            "psd_lf_main": {"exists": False},
        }

    def cached(self, method: str, args: tuple = ()) -> Any:
        if method == "get_lfp_correlation_data_img":
            return {}
        if method == "get_passive_events":
            return {}
        if method == "get_lfp_spectrum_data":
            return "lfp-img", {}
        if method == "get_rfmap_data":
            return {}, None
        return method


def test_preserves_previous_selected_key_when_available() -> None:
    state = build_plot_menu_state(
        FakePlotData(),
        previous_selected_keys={"image": "image.rms_ap"},
    )

    assert state.group("image").selected_key == "image.rms_ap"


def test_falls_back_to_default_when_previous_key_is_unavailable() -> None:
    state = build_plot_menu_state(
        FakePlotData(),
        previous_selected_keys={"image": "image.no_longer_available"},
    )

    assert state.group("image").selected_key == "image.fr"


def test_raw_image_payloads_are_available_and_selectable() -> None:
    state = build_plot_menu_state(
        FakePlotData(),
        previous_selected_keys={"image": "image.raw.raw_ap"},
        raw_image_payloads={"raw_ap": "raw-image"},
    )

    image = state.group("image")
    assert image.selected_key == "image.raw.raw_ap"
    assert "image.raw.raw_ap" in {spec.key for spec in image.specs}


def test_empty_plot_group_is_disabled_without_selection() -> None:
    state = build_plot_menu_state(FakePlotData(spikes=False, rms_ap=False))

    line = state.group("line")
    assert not line.enabled
    assert line.selected_key is None
    assert line.selected_spec is None
