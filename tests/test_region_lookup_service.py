"""Tests for Allen region lookup helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ephys_alignment_gui import region_lookup_service
from ephys_alignment_gui.region_lookup_service import RegionLookupService


def test_load_allen_csv_caches_loaded_structure_tree(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls: list[Path] = []
    loaded = object()

    monkeypatch.setattr(
        region_lookup_service.atlas,
        "__file__",
        str(tmp_path / "atlas.py"),
    )

    def load_file_content(path: Path) -> object:
        calls.append(path)
        return loaded

    monkeypatch.setattr(
        region_lookup_service.alfio,
        "load_file_content",
        load_file_content,
    )

    service = RegionLookupService()

    assert service.load_allen_csv() is loaded
    assert service.load_allen_csv() is loaded
    assert calls == [tmp_path / "allen_structure_tree.csv"]
