"""Tests for desktop folder dialog wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ephys_alignment_gui.desktop_folder_dialog import DesktopFolderDialog


class FakeDirectoryDialog:
    def __init__(self, selected: str = "/data/mouse") -> None:
        self.selected = selected
        self.calls: list[tuple[Any, str, str]] = []

    def __call__(self, parent: Any, title: str, *, directory: str = "") -> str:
        self.calls.append((parent, title, directory))
        return self.selected


def test_folder_dialog_returns_selected_path() -> None:
    dialog_fn = FakeDirectoryDialog("/data/mouse")
    parent = object()
    dialog = DesktopFolderDialog(
        parent=parent,
        get_existing_directory=dialog_fn,
    )

    selected = dialog.select_existing_directory(
        "Select Mouse Root",
        directory=Path("/data"),
    )

    assert selected == Path("/data/mouse")
    assert dialog_fn.calls == [(parent, "Select Mouse Root", "/data")]


def test_folder_dialog_returns_none_when_cancelled() -> None:
    dialog = DesktopFolderDialog(
        get_existing_directory=FakeDirectoryDialog(""),
    )

    assert dialog.select_existing_directory("Select Mouse Root") is None


def test_folder_dialog_can_return_qt_style_text() -> None:
    dialog = DesktopFolderDialog(
        get_existing_directory=FakeDirectoryDialog("/results"),
    )

    assert dialog.select_existing_directory_text("Select Save Root") == "/results"


def test_folder_dialog_text_returns_blank_when_cancelled() -> None:
    dialog = DesktopFolderDialog(
        get_existing_directory=FakeDirectoryDialog(""),
    )

    assert dialog.select_existing_directory_text("Select Save Root") == ""
