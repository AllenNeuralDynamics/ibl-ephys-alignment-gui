"""Tests for previous-alignment desktop loading workflow."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results import AlignmentChoicesUpdated
from ephys_alignment_gui.application.results.alignment_persistence import (
    NoPreviousAlignments,
)
from ephys_alignment_gui.core.alignment_events import (
    PreviousAlignmentLoadFailed,
    PreviousAlignmentsLoaded,
    PreviousAlignmentsUnavailable,
)
from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.workflow import Failed, Ok
from ephys_alignment_gui.desktop.coordinators.previous_alignment_load_coordinator import (
    DesktopPreviousAlignmentLoadCoordinator,
    PreviousAlignmentLoadCallbacks,
)


class FakeBusyFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return nullcontext()


class FakeCommands:
    def __init__(
        self,
        *,
        events: EventBus | None = None,
        ready: Any = Ok(),
        load_result: Any = NoPreviousAlignments(),
    ) -> None:
        self.events = events
        self.ready = ready
        self.load_result = load_result
        self.load_calls: list[dict[str, Any]] = []

    def can_load_previous_alignments(self) -> Any:
        return self.ready

    def load_previous_alignments(self, **kwargs: Any) -> Any:
        self.load_calls.append(kwargs)
        self._emit_load_result(self.load_result)
        return self.load_result

    def _emit_load_result(self, result: Any) -> None:
        if self.events is None:
            return
        if isinstance(result, Failed):
            self.events.emit(
                PreviousAlignmentLoadFailed(
                    shank_idx=0,
                    message=result.message,
                )
            )
        elif isinstance(result, AlignmentChoicesUpdated):
            self.events.emit(
                PreviousAlignmentsLoaded(
                    shank_idx=0,
                    choices=tuple(result.choices),
                )
            )
        elif isinstance(result, NoPreviousAlignments):
            self.events.emit(PreviousAlignmentsUnavailable(shank_idx=0))


def _coordinator(
    commands: FakeCommands,
    *,
    selected_folder: Path | None = None,
    default_folder: Path | None = None,
    use_docdb: bool = False,
    reload_button: Any = "reload-button",
    select_alignment_result: bool = True,
) -> tuple[DesktopPreviousAlignmentLoadCoordinator, dict[str, Any]]:
    calls: dict[str, Any] = {
        "reload_text": [],
        "rendered_choices": [],
        "selected_alignments": [],
        "select_folder_defaults": [],
    }
    events = EventBus()
    commands.events = events
    busy_factory = FakeBusyFactory()
    coordinator = DesktopPreviousAlignmentLoadCoordinator(
        commands=commands,
        events=events,
        callbacks=PreviousAlignmentLoadCallbacks(
            select_folder=lambda directory: (
                calls["select_folder_defaults"].append(directory) or selected_folder
            ),
            default_folder=lambda: default_folder,
            use_docdb=lambda: use_docdb,
            set_reload_folder_text=calls["reload_text"].append,
            render_alignment_choices=calls["rendered_choices"].append,
            select_alignment=lambda idx: (
                calls["selected_alignments"].append(idx) or select_alignment_result
            ),
            busy_context=busy_factory,
            reload_button=lambda: reload_button,
        ),
    )
    coordinator.connect_previous_alignment_events()
    calls["busy_factory"] = busy_factory
    return coordinator, calls


def test_readiness_failure_does_not_prompt_or_load() -> None:
    commands = FakeCommands(ready=Failed("not ready"))
    events = EventBus()
    commands.events = events
    prompt_calls: list[str] = []
    coordinator = DesktopPreviousAlignmentLoadCoordinator(
        commands=commands,
        events=events,
        callbacks=PreviousAlignmentLoadCallbacks(
            select_folder=lambda _directory: (
                prompt_calls.append("prompt") or Path("/tmp/history")
            ),
            default_folder=lambda: Path("/tmp/output"),
            use_docdb=lambda: False,
            set_reload_folder_text=lambda _text: None,
            render_alignment_choices=lambda _choices: None,
            select_alignment=lambda _idx: True,
            busy_context=FakeBusyFactory(),
            reload_button=lambda: None,
        ),
    )
    coordinator.connect_previous_alignment_events()

    assert not coordinator.load_existing_alignments()
    assert prompt_calls == []
    assert commands.load_calls == []


def test_cancel_without_docdb_returns_false_without_loading() -> None:
    commands = FakeCommands()
    coordinator, calls = _coordinator(commands, selected_folder=None, use_docdb=False)

    assert not coordinator.load_existing_alignments()
    assert commands.load_calls == []
    assert calls["reload_text"] == []
    assert calls["busy_factory"].calls == []


def test_cancel_with_docdb_returns_false_without_loading() -> None:
    commands = FakeCommands(load_result=NoPreviousAlignments())
    coordinator, calls = _coordinator(commands, selected_folder=None, use_docdb=True)

    assert not coordinator.load_existing_alignments()
    assert commands.load_calls == []
    assert calls["reload_text"] == []
    assert calls["rendered_choices"] == []
    assert calls["selected_alignments"] == []
    assert calls["busy_factory"].calls == []


def test_loaded_alignment_choices_fail_when_selection_fails() -> None:
    commands = FakeCommands(load_result=AlignmentChoicesUpdated(["original"]))
    coordinator, calls = _coordinator(
        commands,
        selected_folder=Path("/tmp/alignments"),
        select_alignment_result=False,
    )

    assert not coordinator.load_existing_alignments()
    assert commands.load_calls == [
        {"folder": Path("/tmp/alignments"), "use_docdb": False}
    ]
    assert calls["rendered_choices"] == [["original"]]
    assert calls["selected_alignments"] == [0]


def test_selected_folder_renders_loaded_alignment_choices() -> None:
    commands = FakeCommands(
        load_result=AlignmentChoicesUpdated(["original", "2026-07-09T12:00:00"])
    )
    coordinator, calls = _coordinator(
        commands,
        selected_folder=Path("/tmp/alignments"),
        use_docdb=False,
    )

    assert coordinator.load_existing_alignments()
    assert commands.load_calls == [
        {"folder": Path("/tmp/alignments"), "use_docdb": False}
    ]
    assert calls["reload_text"] == ["/tmp/alignments"]
    assert calls["select_folder_defaults"] == [None]
    assert calls["rendered_choices"] == [["original", "2026-07-09T12:00:00"]]
    assert calls["selected_alignments"] == [0]
    busy_calls = calls["busy_factory"].calls
    assert len(busy_calls) == 1
    assert busy_calls[0][0] == ("Loading alignments...", "Alignments loaded")
    assert busy_calls[0][1] == {"disable_widgets": "reload-button"}


def test_loaded_alignment_event_can_render_without_auto_selecting() -> None:
    commands = FakeCommands(load_result=NoPreviousAlignments())
    coordinator, calls = _coordinator(commands)

    coordinator.on_previous_alignments_loaded(
        PreviousAlignmentsLoaded(
            shank_idx=0,
            choices=("saved", "original"),
            auto_select=False,
        )
    )

    assert calls["rendered_choices"] == [["saved", "original"]]
    assert calls["selected_alignments"] == []


def test_load_alignments_prompt_defaults_to_output_package_directory() -> None:
    commands = FakeCommands(load_result=NoPreviousAlignments())
    output_package_directory = Path(
        "/tmp/results/ibl_annotations_mouse_2026-08-16_14-32-05"
    )
    coordinator, calls = _coordinator(
        commands,
        selected_folder=output_package_directory,
        default_folder=output_package_directory,
        use_docdb=False,
    )

    assert coordinator.load_existing_alignments()

    assert calls["select_folder_defaults"] == [output_package_directory]
    assert commands.load_calls == [
        {"folder": output_package_directory, "use_docdb": False}
    ]


def test_load_failure_returns_false_after_prompt() -> None:
    commands = FakeCommands(load_result=Failed("load failed"))
    coordinator, calls = _coordinator(
        commands,
        selected_folder=Path("/tmp/alignments"),
        use_docdb=False,
    )

    assert not coordinator.load_existing_alignments()
    assert commands.load_calls == [
        {"folder": Path("/tmp/alignments"), "use_docdb": False}
    ]
    assert calls["rendered_choices"] == []
    assert calls["selected_alignments"] == []


def test_no_previous_alignments_does_not_render_choices() -> None:
    commands = FakeCommands(load_result=NoPreviousAlignments())
    coordinator, calls = _coordinator(
        commands,
        selected_folder=Path("/tmp/alignments"),
        use_docdb=False,
    )

    assert coordinator.load_existing_alignments()
    assert commands.load_calls == [
        {"folder": Path("/tmp/alignments"), "use_docdb": False}
    ]
    assert calls["rendered_choices"] == []
    assert calls["selected_alignments"] == []
