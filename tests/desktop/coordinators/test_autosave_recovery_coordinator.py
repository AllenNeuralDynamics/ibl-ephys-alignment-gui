"""Tests for desktop autosave recovery coordination."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.application.results import ActiveProbeSelectionState
from ephys_alignment_gui.application.results.autosave import (
    AutosaveCheckpointInspected,
    AutosaveCheckpointRecovered,
)
from ephys_alignment_gui.application.results.metadata import (
    ProbeSelected,
    RecordingSelected,
)
from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.desktop.coordinators.autosave_recovery_coordinator import (
    AutosaveRecoveryCallbacks,
    DesktopAutosaveRecoveryCoordinator,
)


class FakeAutosaveCommands:
    def __init__(
        self,
        inspected: AutosaveCheckpointInspected,
        recovered: AutosaveCheckpointRecovered,
    ) -> None:
        self.inspected = inspected
        self.recovered = recovered
        self.inspect_paths: list[Path] = []
        self.recover_paths: list[Path] = []

    def inspect_checkpoint(self, path: Path) -> AutosaveCheckpointInspected:
        self.inspect_paths.append(path)
        return self.inspected

    def recover_checkpoint(self, path: Path) -> AutosaveCheckpointRecovered:
        self.recover_paths.append(path)
        return self.recovered


class FakeMetadataCommands:
    def __init__(self) -> None:
        self.recordings: list[str] = []
        self.probes: list[tuple[str, str]] = []

    def select_recording_metadata(self, recording_id: str) -> RecordingSelected:
        self.recordings.append(recording_id)
        return RecordingSelected(recording_id=recording_id, probes=["probe-name"])

    def select_probe_metadata(
        self,
        recording_id: str,
        probe_name: str,
    ) -> ProbeSelected:
        self.probes.append((recording_id, probe_name))
        return ProbeSelected(
            recording_id=recording_id,
            probe_name=probe_name,
            shanks=["1/1"],
            n_shanks=1,
            output_directory=Path("/tmp/package/rec/probe-name"),
        )


class FakeShankCommands:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def select_shank(self, shank_idx: int, *, source: str) -> object:
        self.calls.append((shank_idx, source))
        return object()


class FakeWorkspaceQueries:
    def __init__(self, package_dir: Path) -> None:
        self.loaded = False
        self.package_dir = package_dir

    def mouse_root_loaded(self) -> bool:
        return self.loaded

    def active_probe_selection_state(self) -> ActiveProbeSelectionState:
        return ActiveProbeSelectionState(
            recording_id="rec",
            probe_name="probe-name",
            shanks=["1/1"],
            n_shanks=1,
            output_directory=self.package_dir / "rec" / "probe-name",
        )

    def active_output_root(self) -> Path:
        return self.package_dir.parent

    def active_output_directory(self) -> Path:
        return self.package_dir / "rec" / "probe-name"

    def active_output_package_directory(self) -> Path:
        return self.package_dir


class FakeSelectionView:
    def __init__(self) -> None:
        self.sessions = ["rec"]
        self.probes: list[str] = []
        self.shanks: list[str] = []
        self.selected_session_indices: list[int] = []
        self.selected_probe_indices: list[int] = []
        self.selected_shank_indices: list[int] = []

    def selection_widgets(self) -> list[str]:
        return ["session", "probe", "shank"]

    def select_session_text(self, session: str) -> int | None:
        if session not in self.sessions:
            return None
        idx = self.sessions.index(session)
        self.selected_session_indices.append(idx)
        return idx

    def populate_probes(self, probes: list[str]) -> None:
        self.probes = list(probes)

    def select_probe_text(self, probe: str) -> int | None:
        if probe not in self.probes:
            return None
        idx = self.probes.index(probe)
        self.selected_probe_indices.append(idx)
        return idx

    def populate_probe_shanks(self, shanks: list[str]) -> None:
        self.shanks = list(shanks)

    def select_shank_index(self, idx: int) -> None:
        self.selected_shank_indices.append(idx)


def test_recover_autosave_loads_mouse_root_recovers_and_activates_selection(
    tmp_path: Path,
) -> None:
    checkpoint_path = _checkpoint_path(tmp_path)
    mouse_root = tmp_path / "mouse-root"
    key = AlignmentKey("rec", "stream", 0)
    app, commands, queries = _app(tmp_path, checkpoint_path, mouse_root, key)
    view = FakeSelectionView()
    coordinator, calls = _coordinator(app, view, selected_folder=checkpoint_path.parent)

    assert coordinator.recover_autosave()

    assert calls["select_folder_defaults"] == [tmp_path / "package"]
    assert calls["set_mouse_roots"] == [mouse_root]
    assert queries.loaded
    assert commands.autosave.inspect_paths == [checkpoint_path, checkpoint_path]
    assert commands.autosave.recover_paths == [checkpoint_path]
    assert calls["confirmed"] == [commands.autosave.inspected]
    assert commands.metadata.recordings == ["rec"]
    assert commands.metadata.probes == [("rec", "probe-name")]
    assert commands.shanks.calls == [(0, "autosave-recovered")]
    assert view.selected_session_indices == [0]
    assert view.selected_probe_indices == [0]
    assert view.selected_shank_indices == [0]
    assert calls["rendered_paths"] == [
        (tmp_path, tmp_path / "package" / "rec" / "probe-name")
    ]
    assert calls["activations"] == [{"preserve_plot_selection": False}]
    assert calls["busy"] == [
        (
            ("Recovering autosave...", "Autosave recovered"),
            {"disable_widgets": ["session", "probe", "shank"]},
        )
    ]


def test_recover_autosave_cancel_does_not_restore(tmp_path: Path) -> None:
    checkpoint_path = _checkpoint_path(tmp_path)
    key = AlignmentKey("rec", "stream", 0)
    app, commands, queries = _app(
        tmp_path,
        checkpoint_path,
        tmp_path / "mouse-root",
        key,
    )
    queries.loaded = True
    coordinator, calls = _coordinator(
        app,
        FakeSelectionView(),
        selected_folder=checkpoint_path.parent,
        confirm=False,
    )

    assert not coordinator.recover_autosave()

    assert commands.autosave.inspect_paths == [checkpoint_path, checkpoint_path]
    assert commands.autosave.recover_paths == []
    assert calls["activations"] == []
    assert calls["busy"] == []


def test_recover_autosave_warns_when_selected_folder_has_no_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "package" / "autosave" / "alignment_document.json"
    key = AlignmentKey("rec", "stream", 0)
    app, commands, _queries = _app(
        tmp_path,
        checkpoint_path,
        tmp_path / "mouse-root",
        key,
    )
    coordinator, calls = _coordinator(
        app,
        FakeSelectionView(),
        selected_folder=tmp_path / "package",
    )

    assert not coordinator.recover_autosave()

    assert commands.autosave.inspect_paths == []
    assert calls["warnings"] == [
        (
            "Recover Autosave",
            "Selected folder does not contain an autosave checkpoint.",
        )
    ]


def _checkpoint_path(tmp_path: Path) -> Path:
    checkpoint_path = tmp_path / "package" / "autosave" / "alignment_document.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("{}", encoding="utf-8")
    return checkpoint_path


def _app(
    tmp_path: Path,
    checkpoint_path: Path,
    mouse_root: Path,
    key: AlignmentKey,
) -> tuple[Any, Any, FakeWorkspaceQueries]:
    inspected = AutosaveCheckpointInspected(
        path=checkpoint_path,
        mouse_id="mouse",
        mouse_root=mouse_root,
        output_package_directory=tmp_path / "package",
        selected_alignment_key=key,
        alignment_state_count=1,
        saveable_alignment_count=1,
        dirty_alignment_count=1,
        recoverable_alignment_count=1,
    )
    recovered = AutosaveCheckpointRecovered(
        path=checkpoint_path,
        backup_path=None,
        selected_alignment_key=key,
        restored_alignment_count=1,
    )
    autosave = FakeAutosaveCommands(inspected, recovered)
    commands = SimpleNamespace(
        autosave=autosave,
        metadata=FakeMetadataCommands(),
        shanks=FakeShankCommands(),
    )
    queries = FakeWorkspaceQueries(tmp_path / "package")
    app = SimpleNamespace(commands=commands, queries=SimpleNamespace(workspace=queries))
    return app, commands, queries


def _coordinator(
    app: Any,
    view: FakeSelectionView,
    *,
    selected_folder: Path | None,
    confirm: bool = True,
) -> tuple[DesktopAutosaveRecoveryCoordinator, dict[str, Any]]:
    calls: dict[str, Any] = {
        "select_folder_defaults": [],
        "set_mouse_roots": [],
        "confirmed": [],
        "rendered_paths": [],
        "activations": [],
        "warnings": [],
        "busy": [],
    }

    def set_mouse_root(path: Path) -> bool:
        calls["set_mouse_roots"].append(path)
        app.queries.workspace.loaded = True
        return True

    def busy_context(*args: Any, **kwargs: Any) -> Any:
        calls["busy"].append((args, kwargs))
        return nullcontext()

    coordinator = DesktopAutosaveRecoveryCoordinator(
        app=app,
        selection_view=view,
        callbacks=AutosaveRecoveryCallbacks(
            select_folder=lambda default: (
                calls["select_folder_defaults"].append(default) or selected_folder
            ),
            default_folder=app.queries.workspace.active_output_package_directory,
            confirm_recovery=lambda inspected: (
                calls["confirmed"].append(inspected) or confirm
            ),
            set_mouse_root=set_mouse_root,
            activate_selected_stream=lambda **kwargs: (
                calls["activations"].append(kwargs) or True
            ),
            render_output_paths=lambda output_root, output_directory: calls[
                "rendered_paths"
            ].append((output_root, output_directory)),
            busy_context=busy_context,
            warning=lambda title, message: calls["warnings"].append(
                (title, message)
            ),
        ),
    )
    return coordinator, calls
