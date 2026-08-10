"""Tests for desktop load preflight prompts."""

from __future__ import annotations

from typing import Any

from ephys_alignment_gui.desktop.load_preflight_presenter import (
    DesktopLoadPreflightPresenter,
    DesktopOutputFolderPrompt,
    OutputFolderPromptCallbacks,
)
from ephys_alignment_gui.workflow import (
    CHOOSE_OUTPUT_FOLDER,
    Blocked,
    Ok,
    Requirement,
)


class FakeMessageBox:
    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.icon: Any = None
        self.title: str | None = None
        self.text: str | None = None
        self.informative_text: str | None = None
        self.buttons: list[Any] = []
        self.default_button: Any = None
        self.set_button = object()
        self.cancel_button = object()
        self.exec_count = 0

    def setIcon(self, icon: Any) -> None:
        self.icon = icon

    def setWindowTitle(self, title: str) -> None:
        self.title = title

    def setText(self, text: str) -> None:
        self.text = text

    def setInformativeText(self, text: str) -> None:
        self.informative_text = text

    def addButton(self, *args: Any) -> object:
        button = (
            self.set_button
            if args[0] == "Set Output Folder..."
            else self.cancel_button
        )
        self.buttons.append(button)
        return button

    def setDefaultButton(self, button: Any) -> None:
        self.default_button = button

    def exec_(self) -> None:
        self.exec_count += 1

    def clickedButton(self) -> object:
        return self.set_button if self.accept else self.cancel_button


def _output_requirement() -> Requirement:
    return Requirement(
        code="output_required",
        message="Choose output",
        action=CHOOSE_OUTPUT_FOLDER,
    )


def test_output_prompt_for_load_derives_output_without_dialog() -> None:
    calls: list[str] = []
    prompt = DesktopOutputFolderPrompt(
        callbacks=OutputFolderPromptCallbacks(
            derive_output_directory_from_save_root=lambda: (
                calls.append("derive") or True
            ),
            has_output_directory=lambda: False,
            select_output_folder=lambda: calls.append("select") or False,
        ),
        message_box_factory=lambda _parent: (_ for _ in ()).throw(
            AssertionError("dialog should not open")
        ),
    )

    assert prompt.ensure_for_load(_output_requirement())
    assert calls == ["derive"]


def test_output_prompt_accepts_folder_selection() -> None:
    has_output = False
    box = FakeMessageBox(accept=True)

    def select_output() -> bool:
        nonlocal has_output
        has_output = True
        return True

    prompt = DesktopOutputFolderPrompt(
        callbacks=OutputFolderPromptCallbacks(
            derive_output_directory_from_save_root=lambda: False,
            has_output_directory=lambda: has_output,
            select_output_folder=select_output,
        ),
        message_box_factory=lambda _parent: box,
    )

    assert prompt.ensure_for_load(_output_requirement())
    assert box.exec_count == 1
    assert box.title == "Output Folder Required"
    assert box.text == "Choose output"


def test_output_prompt_cancel_blocks_load() -> None:
    box = FakeMessageBox(accept=False)
    prompt = DesktopOutputFolderPrompt(
        callbacks=OutputFolderPromptCallbacks(
            derive_output_directory_from_save_root=lambda: False,
            has_output_directory=lambda: False,
            select_output_folder=lambda: True,
        ),
        message_box_factory=lambda _parent: box,
    )

    assert not prompt.ensure_for_load(_output_requirement())
    assert box.exec_count == 1


def test_load_presenter_retries_policy_after_output_prompt() -> None:
    results = [
        Blocked((_output_requirement(),)),
        Ok(),
    ]
    prompt_calls: list[Requirement] = []
    heavy_loads: list[str] = []
    presenter = DesktopLoadPreflightPresenter(
        can_load_data=lambda: results.pop(0),
        load_heavy_data=lambda: heavy_loads.append("loaded"),
        output_folder_prompt=type(
            "Prompt",
            (),
            {"ensure_for_load": lambda _self, req: prompt_calls.append(req) or True},
        )(),
    )

    assert presenter.load_data_button_pressed()
    assert prompt_calls == [_output_requirement()]
    assert heavy_loads == ["loaded"]


def test_load_presenter_does_not_load_when_prompt_is_cancelled() -> None:
    heavy_loads: list[str] = []
    presenter = DesktopLoadPreflightPresenter(
        can_load_data=lambda: Blocked((_output_requirement(),)),
        load_heavy_data=lambda: heavy_loads.append("loaded"),
        output_folder_prompt=type(
            "Prompt",
            (),
            {"ensure_for_load": lambda _self, _req: False},
        )(),
    )

    assert not presenter.load_data_button_pressed()
    assert heavy_loads == []


def test_load_presenter_logs_non_actionable_requirement(caplog) -> None:
    requirement = Requirement(code="probe_required", message="Select a probe first.")
    presenter = DesktopLoadPreflightPresenter(
        can_load_data=lambda: Blocked((requirement,)),
        load_heavy_data=lambda: None,
        output_folder_prompt=type(
            "Prompt",
            (),
            {"ensure_for_load": lambda _self, _req: True},
        )(),
    )

    assert not presenter.load_data_button_pressed()
    assert "Select a probe first." in caplog.text
