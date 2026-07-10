"""Tests for Qt-free workflow policy."""

from __future__ import annotations

from pathlib import Path

from ephys_alignment_gui.document import AlignmentDocument
from ephys_alignment_gui.workflow import (
    CHANNEL_INFO_REQUIRED,
    CHOOSE_OUTPUT_FOLDER,
    OUTPUT_REQUIRED,
    PROBE_REQUIRED,
    Blocked,
    Ok,
    WorkflowPolicy,
)


def _check_load_data(
    *,
    probe_selected: bool = True,
    channel_info_loaded: bool = True,
    output_directory_set: bool = True,
):
    doc = AlignmentDocument()
    if probe_selected:
        doc.select_probe("rec1", "probeA")
    doc.set_channel_info_loaded(channel_info_loaded)
    if output_directory_set:
        doc.set_output_directory(Path("/tmp/results/rec1/probeA"))
    return WorkflowPolicy().can_load_data(doc)


def test_load_data_allowed_when_preconditions_are_met():
    assert isinstance(_check_load_data(), Ok)


def test_load_data_requires_selected_probe():
    result = _check_load_data(probe_selected=False)

    assert isinstance(result, Blocked)
    assert result.first.code == PROBE_REQUIRED


def test_load_data_requires_channel_info():
    result = _check_load_data(channel_info_loaded=False)

    assert isinstance(result, Blocked)
    assert result.first.code == CHANNEL_INFO_REQUIRED


def test_load_data_requires_output_directory_with_action():
    result = _check_load_data(output_directory_set=False)

    assert isinstance(result, Blocked)
    assert result.first.code == OUTPUT_REQUIRED
    assert result.first.action == CHOOSE_OUTPUT_FOLDER


def test_load_data_reports_all_missing_requirements_in_order():
    result = _check_load_data(
        probe_selected=False,
        channel_info_loaded=False,
        output_directory_set=False,
    )

    assert isinstance(result, Blocked)
    assert [req.code for req in result.requirements] == [
        PROBE_REQUIRED,
        CHANNEL_INFO_REQUIRED,
        OUTPUT_REQUIRED,
    ]
