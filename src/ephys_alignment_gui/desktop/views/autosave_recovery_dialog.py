"""Autosave recovery confirmation dialog."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PyQt5 import QtWidgets


class DesktopAutosaveRecoveryDialog:
    """Desktop dialog wrapper for confirming autosave recovery."""

    def __init__(self, parent: Any = None) -> None:
        self.parent = parent

    def confirm_recovery(self, inspected: Any) -> bool:
        """Ask whether a checkpoint should replace the live document state."""
        box = QtWidgets.QMessageBox(self.parent)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Recover Autosave")
        box.setText("Recover autosaved alignment work?")
        box.setInformativeText(_summary_text(inspected))
        box.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
        )
        box.setDefaultButton(QtWidgets.QMessageBox.Yes)
        return box.exec_() == QtWidgets.QMessageBox.Yes

    def warning(self, title: str, message: str) -> Any:
        """Show a recovery warning message."""
        return QtWidgets.QMessageBox.warning(self.parent, title, message)


def _summary_text(inspected: Any) -> str:
    modified = _modified_time(getattr(inspected, "path", None))
    selected_key = getattr(inspected, "selected_alignment_key", None)
    skipped_count = len(getattr(inspected, "skipped_keys", ()))
    lines = [
        f"Checkpoint: {getattr(inspected, 'path', '')}",
        f"Modified: {modified}",
        f"Mouse: {getattr(inspected, 'mouse_id', None) or 'unknown'}",
        f"Selected: {_describe_key(selected_key)}",
        f"Recoverable alignments: {getattr(inspected, 'recoverable_alignment_count', 0)}",
        f"Dirty alignments: {getattr(inspected, 'dirty_alignment_count', 0)}",
    ]
    if skipped_count:
        lines.append(f"Skipped invalid alignments: {skipped_count}")
    return "\n".join(lines)


def _modified_time(path: Any) -> str:
    if path is None:
        return "unknown"
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        return modified.isoformat(timespec="seconds")
    except OSError:
        return "unknown"


def _describe_key(key: Any) -> str:
    if key is None:
        return "none"
    return f"{key.recording_id}/{key.ephys_collection}/shank {key.shank_idx + 1}"
