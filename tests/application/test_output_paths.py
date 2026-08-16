"""Tests for alignment annotation output path helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ephys_alignment_gui.application.output_paths import (
    alignment_output_package_directory,
    alignment_output_package_name,
    probe_alignment_output_directory,
)


def test_alignment_output_package_name_is_code_ocean_friendly() -> None:
    timestamp = datetime(2026, 8, 16, 14, 32, 5)

    assert (
        alignment_output_package_name("mouse 771432", timestamp)
        == "ibl_annotations_mouse_771432_2026-08-16_14-32-05"
    )


def test_probe_alignment_output_directory_is_nested_under_package() -> None:
    timestamp = datetime(2026, 8, 16, 14, 32, 5)
    package_dir = alignment_output_package_directory(
        Path("/results"),
        "771432",
        timestamp,
    )

    assert package_dir == Path("/results/ibl_annotations_771432_2026-08-16_14-32-05")
    assert probe_alignment_output_directory(
        package_dir,
        "ecephys_771432_2025-03-12_16-11-26",
        "46116",
    ) == Path(
        "/results/ibl_annotations_771432_2026-08-16_14-32-05/"
        "ecephys_771432_2025-03-12_16-11-26/46116"
    )
