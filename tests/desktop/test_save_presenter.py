"""Tests for desktop save/QC presentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ephys_alignment_gui.application.results import VisitedAlignmentOutputsSaved
from ephys_alignment_gui.application.results.alignment_persistence import (
    AlignmentOutputsSaved,
)
from ephys_alignment_gui.application.workflow import (
    CHOOSE_OUTPUT_FOLDER,
    OUTPUT_REQUIRED,
    Blocked,
    Failed,
    Ok,
    Requirement,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.desktop.save_presenter import (
    DesktopSaveCallbacks,
    DesktopSavePresenter,
)
from ephys_alignment_gui.services.alignment_repository import SavedAlignmentOutputs


class FakeBusyContext:
    def __init__(
        self,
        calls: list[tuple],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.calls = calls
        self.calls.append(("busy", args, kwargs))

    def __enter__(self):
        self.calls.append(("busy-enter",))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.calls.append(("busy-exit", exc_type))
        return False


class FakeCommands:
    def __init__(
        self,
        *,
        ready_results: list[Any] | None = None,
        save_result: Any | None = None,
    ) -> None:
        self.ready_results = ready_results or [Ok()]
        self.save_result = save_result or _saved_result()
        self.ready_calls = 0
        self.save_calls: list[dict[str, Any]] = []

    def can_save_alignment_output(self):
        result = self.ready_results[min(self.ready_calls, len(self.ready_results) - 1)]
        self.ready_calls += 1
        return result

    def save_visited_alignment_outputs(self, *, use_docdb: bool):
        self.save_calls.append({"use_docdb": use_docdb})
        return self.save_result


def _requirement() -> Requirement:
    return Requirement(
        OUTPUT_REQUIRED,
        "Choose an output folder before saving.",
        action=CHOOSE_OUTPUT_FOLDER,
    )


def _saved_result(
    *,
    docdb_probe_name: str | None = "probeA_0",
    docdb_error: str | None = None,
) -> VisitedAlignmentOutputsSaved:
    key = AlignmentKey("rec", "stream", 1)
    return VisitedAlignmentOutputsSaved(
        saved_count=1,
        saved_outputs={
            key: AlignmentOutputsSaved(
                saved=SavedAlignmentOutputs(
                    channel_results_path=Path("/results/channels.json"),
                    previous_alignments_path=Path("/results/alignments.json"),
                    ccf_channel_results_path=Path("/results/ccf.json"),
                    docdb_probe_name=docdb_probe_name,
                    docdb_error=docdb_error,
                ),
                previous_alignments={},
            )
        },
        active_choices=["saved", "original"],
    )


def _presenter(
    *,
    commands: FakeCommands | None = None,
    ensure_output: bool = True,
    use_docdb: bool = True,
    histology_available: bool = True,
    ephys_qc: str = "Pass",
    selected_descriptions: list[str] | None = None,
) -> tuple[DesktopSavePresenter, FakeCommands, list[tuple]]:
    calls: list[tuple] = []
    commands = commands or FakeCommands()
    presenter = DesktopSavePresenter(
        commands=commands,
        callbacks=DesktopSaveCallbacks(
            ensure_output_directory=lambda requirement: calls.append(
                ("ensure-output", requirement)
            )
            or ensure_output,
            log_requirement=lambda requirement: calls.append(
                ("requirement", requirement)
            ),
            use_docdb=lambda: use_docdb,
            render_alignment_choices=lambda choices: calls.append(
                ("choices", choices)
            ),
            busy_context=lambda *args, **kwargs: FakeBusyContext(
                calls,
                *args,
                **kwargs,
            ),
            complete_button=lambda: "complete-button",
            histology_available=lambda: histology_available,
            open_qc_dialog=lambda: calls.append(("open-qc",)),
            ephys_qc=lambda: ephys_qc,
            selected_qc_descriptions=lambda: selected_descriptions or [],
            warning=lambda title, message: calls.append(("warning", title, message)),
        ),
    )
    return presenter, commands, calls


def test_save_prompts_for_output_then_saves() -> None:
    blocked = Blocked((_requirement(),))
    presenter, commands, calls = _presenter(
        commands=FakeCommands(ready_results=[blocked, Ok()]),
    )

    assert presenter.save_alignment_outputs()

    assert commands.ready_calls == 2
    assert commands.save_calls == [{"use_docdb": True}]
    assert calls == [
        ("ensure-output", blocked.first),
        (
            "busy",
            ("Saving...", "Saved successfully"),
            {"disable_widgets": "complete-button"},
        ),
        ("busy-enter",),
        ("choices", ["saved", "original"]),
        ("busy-exit", None),
    ]


def test_save_returns_false_when_output_prompt_is_cancelled() -> None:
    blocked = Blocked((_requirement(),))
    presenter, commands, calls = _presenter(
        commands=FakeCommands(ready_results=[blocked]),
        ensure_output=False,
    )

    assert not presenter.save_alignment_outputs()

    assert commands.save_calls == []
    assert calls == [("ensure-output", blocked.first)]


def test_save_logs_failed_command() -> None:
    presenter, commands, calls = _presenter(
        commands=FakeCommands(save_result=Failed("save failed")),
    )

    assert not presenter.save_alignment_outputs()

    assert commands.save_calls == [{"use_docdb": True}]
    assert calls == [
        (
            "busy",
            ("Saving...", "Saved successfully"),
            {"disable_widgets": "complete-button"},
        ),
        ("busy-enter",),
        ("busy-exit", None),
    ]


def test_display_qc_options_opens_dialog_when_histology_available() -> None:
    presenter, _commands, calls = _presenter()

    assert presenter.display_qc_options()

    assert calls == [("open-qc",)]


def test_display_qc_options_noops_without_histology() -> None:
    presenter, _commands, calls = _presenter(histology_available=False)

    assert not presenter.display_qc_options()

    assert calls == []


def test_qc_button_requires_description_for_failing_qc() -> None:
    presenter, commands, calls = _presenter(ephys_qc="Fail")

    assert not presenter.qc_button_clicked()

    assert commands.save_calls == []
    assert calls == [
        ("warning", "Status", "You must select a reason for qc choice"),
        ("open-qc",),
    ]


def test_qc_button_saves_when_description_is_selected() -> None:
    presenter, commands, calls = _presenter(
        ephys_qc="Fail",
        selected_descriptions=["noise"],
    )

    assert presenter.qc_button_clicked()

    assert commands.save_calls == [{"use_docdb": True}]
    assert ("choices", ["saved", "original"]) in calls
