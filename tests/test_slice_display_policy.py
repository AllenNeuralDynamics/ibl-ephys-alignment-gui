"""Tests for Qt-free anatomical slice display policy."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.slice_display_policy import (
    SliceDisplayPolicy,
    SliceImageKind,
    SliceSelection,
)


def test_menu_items_and_default_selection_are_separate() -> None:
    policy = SliceDisplayPolicy()
    slice_data = {
        "ccf": np.zeros((2, 2)),
        "label": np.zeros((2, 2, 3), dtype=np.uint8),
        "annotation_ids": np.ones((2, 2), dtype=np.uint16),
        "scale": np.ones(2),
        "offset": np.zeros(2),
        "histology_registration": np.ones((2, 2)),
        "fluorescence": np.ones((2, 2)),
    }

    items = policy.menu_items(
        slice_data=slice_data,
        fp_slice_data={"label": np.zeros((2, 2, 3), dtype=np.uint8)},
        offline=True,
    )

    assert [item.label for item in items] == [
        "CCF",
        "Annotation",
        "Annotation FP",
        "histology_registration",
        "fluorescence",
    ]
    assert policy.default_selection(slice_data) == SliceSelection(
        "slice_data", "histology_registration"
    )
    assert not hasattr(items[0], "is_default")


def test_menu_items_deduplicate_legacy_hist_cb() -> None:
    policy = SliceDisplayPolicy()
    slice_data = {
        "ccf": np.zeros((2, 2)),
        "label": np.zeros((2, 2, 3), dtype=np.uint8),
        "hist_cb": np.ones((2, 2)),
    }

    items = policy.menu_items(
        slice_data=slice_data,
        fp_slice_data=None,
        offline=False,
    )

    hist_cb_items = [
        item
        for item in items
        if item.selection == SliceSelection("slice_data", "hist_cb")
    ]
    assert len(hist_cb_items) == 1
    assert hist_cb_items[0].label == "Histology cerebellar example"


def test_default_selection_falls_back_to_ccf() -> None:
    policy = SliceDisplayPolicy()

    assert policy.default_selection({"ccf": np.zeros((2, 2))}) == SliceSelection(
        "slice_data", "ccf"
    )


def test_choose_selection_restores_previous_when_available() -> None:
    policy = SliceDisplayPolicy()
    previous = SliceSelection("slice_data", "fluorescence")
    default = SliceSelection("slice_data", "ccf")

    decision = policy.choose_selection(
        previous=previous,
        default=default,
        data_by_attr={"slice_data": {"ccf": 1, "fluorescence": 2}},
    )

    assert decision.selection == previous
    assert decision.used_previous


def test_choose_selection_uses_default_when_previous_missing() -> None:
    policy = SliceDisplayPolicy()
    previous = SliceSelection("slice_data", "missing")
    default = SliceSelection("slice_data", "ccf")

    decision = policy.choose_selection(
        previous=previous,
        default=default,
        data_by_attr={"slice_data": {"ccf": 1}},
    )

    assert decision.selection == default
    assert not decision.used_previous


def test_render_decision_classifies_label_rgb_and_scalar() -> None:
    policy = SliceDisplayPolicy()

    assert (
        policy.render_decision({"label": np.zeros((2, 2, 3))}, "label").kind
        is SliceImageKind.LABEL
    )
    assert (
        policy.render_decision({"phase": np.zeros((2, 2, 4))}, "phase").kind
        is SliceImageKind.RGB
    )

    scalar_data = {
        "histology_registration": np.arange(16, dtype=float).reshape(4, 4),
        "annotation_ids": np.pad(np.ones((2, 2), dtype=np.uint16), 1),
    }
    decision = policy.render_decision(scalar_data, "histology_registration")

    assert decision.kind is SliceImageKind.SCALAR
    assert decision.scalar_channel == "histology_registration"
    expected = np.percentile(scalar_data["histology_registration"][1:3, 1:3], [5, 95])
    assert decision.initial_levels == (expected[0], expected[1])


def test_scalar_channel_for_selection_rejects_label_rgb_and_missing() -> None:
    policy = SliceDisplayPolicy()
    data_by_attr = {
        "slice_data": {
            "label": np.zeros((2, 2, 3)),
            "phase": np.zeros((2, 2, 4)),
            "ccf": np.zeros((2, 2)),
        }
    }

    assert (
        policy.scalar_channel_for_selection(
            data_by_attr, SliceSelection("slice_data", "label")
        )
        is None
    )
    assert (
        policy.scalar_channel_for_selection(
            data_by_attr, SliceSelection("slice_data", "phase")
        )
        is None
    )
    assert (
        policy.scalar_channel_for_selection(
            data_by_attr, SliceSelection("slice_data", "missing")
        )
        is None
    )
    assert (
        policy.scalar_channel_for_selection(
            data_by_attr, SliceSelection("slice_data", "ccf")
        )
        == "ccf"
    )
