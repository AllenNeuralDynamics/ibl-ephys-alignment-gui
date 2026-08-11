"""Tests for desktop path-selection dialog coordination."""

from __future__ import annotations

from pathlib import Path

from ephys_alignment_gui.desktop.coordinators.path_dialog_coordinator import (
    DesktopPathDialogCallbacks,
    DesktopPathDialogCoordinator,
)


class FakeFolderDialog:
    def __init__(self, selected: Path | None) -> None:
        self.selected = selected
        self.calls: list[tuple[str, Path | None]] = []

    def select_existing_directory(
        self,
        title: str,
        *,
        directory: Path | None = None,
    ) -> Path | None:
        self.calls.append((title, directory))
        return self.selected


def _coordinator(
    *,
    selected: Path | None = Path("/data/mouse"),
    active_mouse_root: Path | None = None,
    input_root: Path | None = None,
    active_output_root: Path | None = None,
) -> tuple[DesktopPathDialogCoordinator, FakeFolderDialog, dict[str, list[Path]]]:
    calls: dict[str, list[Path]] = {"mouse": [], "output": []}
    folder_dialog = FakeFolderDialog(selected)
    coordinator = DesktopPathDialogCoordinator(
        folder_dialog=folder_dialog,
        callbacks=DesktopPathDialogCallbacks(
            active_mouse_root=lambda: active_mouse_root,
            set_mouse_root=lambda path: calls["mouse"].append(path) is None,
            active_output_root=lambda: active_output_root,
            set_save_root=lambda path: calls["output"].append(path) is None,
        ),
        input_root_provider=lambda: input_root,
    )
    return coordinator, folder_dialog, calls


def test_select_mouse_root_uses_active_root_as_start_dir() -> None:
    coordinator, folder_dialog, calls = _coordinator(
        selected=Path("/data/new-mouse"),
        active_mouse_root=Path("/data/current-mouse"),
    )

    assert coordinator.select_mouse_root()

    assert folder_dialog.calls == [
        ("Select Mouse Root", Path("/data/current-mouse")),
    ]
    assert calls["mouse"] == [Path("/data/new-mouse")]
    assert calls["output"] == []


def test_select_mouse_root_uses_existing_input_root_when_no_mouse_loaded(
    tmp_path: Path,
) -> None:
    coordinator, folder_dialog, calls = _coordinator(
        selected=tmp_path / "mouse",
        input_root=tmp_path,
    )

    assert coordinator.select_mouse_root()

    assert folder_dialog.calls == [("Select Mouse Root", tmp_path)]
    assert calls["mouse"] == [tmp_path / "mouse"]


def test_select_mouse_root_ignores_missing_input_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    coordinator, folder_dialog, calls = _coordinator(
        selected=Path("/data/mouse"),
        input_root=missing,
    )

    assert coordinator.select_mouse_root()

    assert folder_dialog.calls == [("Select Mouse Root", None)]
    assert calls["mouse"] == [Path("/data/mouse")]


def test_select_mouse_root_returns_false_when_cancelled(tmp_path: Path) -> None:
    coordinator, folder_dialog, calls = _coordinator(
        selected=None,
        input_root=tmp_path,
    )

    assert not coordinator.select_mouse_root()

    assert folder_dialog.calls == [("Select Mouse Root", tmp_path)]
    assert calls["mouse"] == []


def test_select_output_root_uses_active_output_root_as_start_dir() -> None:
    coordinator, folder_dialog, calls = _coordinator(
        selected=Path("/results/new"),
        active_output_root=Path("/results/current"),
    )

    assert coordinator.select_output_root()

    assert folder_dialog.calls == [
        ("Select Save Root", Path("/results/current")),
    ]
    assert calls["output"] == [Path("/results/new")]
    assert calls["mouse"] == []


def test_select_output_root_returns_false_when_cancelled() -> None:
    coordinator, folder_dialog, calls = _coordinator(
        selected=None,
        active_output_root=Path("/results/current"),
    )

    assert not coordinator.select_output_root()

    assert folder_dialog.calls == [
        ("Select Save Root", Path("/results/current")),
    ]
    assert calls["output"] == []
