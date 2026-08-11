"""Tests for desktop path widget wrapper."""

from __future__ import annotations

from pathlib import Path

from ephys_alignment_gui.desktop.views.path_view import DesktopPathView


class FakeLineEdit:
    def __init__(self, text: str = "") -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text


def test_path_view_wraps_text_fields_and_mouse_root_widgets() -> None:
    button = object()
    mouse_root = FakeLineEdit("  /data/mouse  ")
    output = FakeLineEdit("  /results  ")
    view = DesktopPathView(
        mouse_root_button=button,
        mouse_root_line=mouse_root,
        output_folder_line=output,
    )

    view.set_mouse_root(Path("/data/new-mouse"))
    view.set_output_directory(Path("/results/probe"))

    assert view.mouse_root_text() == "/data/new-mouse"
    assert view.output_root_text() == "/results/probe"
    assert view.mouse_root_widgets() == [button, mouse_root]


def test_path_view_can_show_save_root_when_no_probe_output_exists() -> None:
    view = DesktopPathView(
        mouse_root_button=object(),
        mouse_root_line=FakeLineEdit(),
        output_folder_line=FakeLineEdit(),
    )

    view.set_output_root(Path("/results"))
    view.set_output_directory(None)

    assert view.output_root_text() == "/results"
