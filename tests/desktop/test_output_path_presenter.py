"""Tests for desktop output-path presentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ephys_alignment_gui.desktop.output_path_presenter import (
    DesktopOutputPathPresenter,
)
from ephys_alignment_gui.path_results import OutputDirectoryDerived, OutputRootSet
from ephys_alignment_gui.workflow import Failed


class FakePathView:
    def __init__(self, calls: list[tuple], *, output_text: str = "/results") -> None:
        self.calls = calls
        self.output_text = output_text

    def set_output_directory(self, output_directory: Path | None) -> None:
        self.calls.append(("output-dir", output_directory))

    def set_output_root(self, output_root: Path) -> None:
        self.calls.append(("output-root", output_root))

    def output_root_text(self) -> str:
        return self.output_text


class FakeCommands:
    def __init__(
        self,
        *,
        derive_result: Any | None = None,
        set_result: Any | None = None,
    ) -> None:
        self.derive_result = derive_result or OutputDirectoryDerived(
            Path("/results/rec/probe")
        )
        self.set_result = set_result or OutputRootSet(
            Path("/results"),
            Path("/results/rec/probe"),
        )
        self.derive_calls = 0
        self.set_calls: list[Path] = []

    def derive_output_directory(self):
        self.derive_calls += 1
        return self.derive_result

    def set_output_root(self, output_root: Path):
        self.set_calls.append(output_root)
        return self.set_result


def _presenter(
    *,
    commands: FakeCommands | None = None,
    calls: list[tuple] | None = None,
    output_text: str = "/results",
) -> tuple[DesktopOutputPathPresenter, FakeCommands, list[tuple]]:
    calls = calls if calls is not None else []
    commands = commands or FakeCommands()
    presenter = DesktopOutputPathPresenter(
        commands=commands,
        path_view=FakePathView(calls, output_text=output_text),
    )
    return presenter, commands, calls


def test_derive_output_directory_renders_probe_output_path() -> None:
    presenter, commands, calls = _presenter()

    assert presenter.derive_output_directory_from_save_root()

    assert commands.derive_calls == 1
    assert calls == [("output-dir", Path("/results/rec/probe"))]


def test_derive_output_directory_returns_false_without_probe_output() -> None:
    presenter, _commands, calls = _presenter(
        commands=FakeCommands(derive_result=OutputDirectoryDerived(None))
    )

    assert not presenter.derive_output_directory_from_save_root()

    assert calls == []


def test_set_save_root_renders_probe_output_when_available() -> None:
    presenter, commands, calls = _presenter()

    assert presenter.set_save_root(Path("/results"))

    assert commands.set_calls == [Path("/results")]
    assert calls == [("output-dir", Path("/results/rec/probe"))]


def test_set_save_root_renders_root_when_no_probe_output_exists() -> None:
    presenter, _commands, calls = _presenter(
        commands=FakeCommands(
            set_result=OutputRootSet(Path("/results"), None),
        )
    )

    assert presenter.set_save_root(Path("/results"))

    assert calls == [("output-root", Path("/results"))]


def test_set_save_root_failure_does_not_render_path() -> None:
    presenter, commands, calls = _presenter(
        commands=FakeCommands(set_result=Failed("bad output"))
    )

    assert not presenter.set_save_root(Path("/bad"))

    assert commands.set_calls == [Path("/bad")]
    assert calls == []


def test_output_folder_edited_sets_save_root_from_text() -> None:
    presenter, commands, calls = _presenter(output_text=" /edited/results ")

    assert presenter.output_folder_edited()

    assert commands.set_calls == [Path("/edited/results")]
    assert calls == [("output-dir", Path("/results/rec/probe"))]


def test_output_folder_edited_ignores_empty_text() -> None:
    presenter, commands, calls = _presenter(output_text=" ")

    assert not presenter.output_folder_edited()

    assert commands.set_calls == []
    assert calls == []
