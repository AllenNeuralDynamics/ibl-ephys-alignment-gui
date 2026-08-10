"""Tests for desktop display-region composition."""

from __future__ import annotations

from typing import Any

import ephys_alignment_gui.desktop.displays as module
from ephys_alignment_gui.desktop.displays import DesktopDisplayPorts, DesktopDisplays


def test_desktop_displays_factory_composes_display_regions(monkeypatch) -> None:
    calls: list[tuple[str, Any, Any]] = []
    displays = {
        "ephys": object(),
        "histology": object(),
        "reference_lines": object(),
        "slice": object(),
    }

    def create_ephys(*, app: Any, ports: Any) -> Any:
        calls.append(("ephys", app, ports))
        return displays["ephys"]

    def create_histology(*, app: Any, ports: Any) -> Any:
        calls.append(("histology", app, ports))
        return displays["histology"]

    def create_reference_lines(*, ports: Any) -> Any:
        calls.append(("reference_lines", None, ports))
        return displays["reference_lines"]

    def create_slice(*, app: Any, ports: Any) -> Any:
        calls.append(("slice", app, ports))
        return displays["slice"]

    monkeypatch.setattr(
        module.DesktopEphysDisplay,
        "create",
        staticmethod(create_ephys),
    )
    monkeypatch.setattr(
        module.DesktopHistologyDisplay,
        "create",
        staticmethod(create_histology),
    )
    monkeypatch.setattr(
        module.DesktopReferenceLineDisplay,
        "create",
        staticmethod(create_reference_lines),
    )
    monkeypatch.setattr(
        module.DesktopSliceDisplay,
        "create",
        staticmethod(create_slice),
    )
    ports = DesktopDisplayPorts(
        ephys="ephys-ports",
        histology="histology-ports",
        reference_lines="reference-line-ports",
        slice="slice-ports",
    )

    result = DesktopDisplays.create(app="app", ports=ports)

    assert result.ephys is displays["ephys"]
    assert result.histology is displays["histology"]
    assert result.reference_lines is displays["reference_lines"]
    assert result.slice is displays["slice"]
    assert calls == [
        ("ephys", "app", "ephys-ports"),
        ("histology", "app", "histology-ports"),
        ("reference_lines", None, "reference-line-ports"),
        ("slice", "app", "slice-ports"),
    ]
