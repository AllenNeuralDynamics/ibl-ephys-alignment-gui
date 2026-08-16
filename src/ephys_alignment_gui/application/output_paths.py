"""Output path helpers for GUI-created alignment annotation packages."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def alignment_output_package_name(mouse_id: str, timestamp: datetime) -> str:
    """Return a Code Ocean-friendly folder name for one annotation session."""
    return f"ibl_annotations_{_safe_path_token(mouse_id)}_{timestamp:%Y-%m-%d_%H-%M-%S}"


def alignment_output_package_directory(
    output_root: Path,
    mouse_id: str,
    timestamp: datetime,
) -> Path:
    """Return the mouse-level output package directory under a save root."""
    return Path(output_root) / alignment_output_package_name(mouse_id, timestamp)


def probe_alignment_output_directory(
    output_package_directory: Path,
    recording_id: str,
    probe_name: str,
) -> Path:
    """Return the per-recording/probe directory inside an output package."""
    return Path(output_package_directory) / recording_id / probe_name


def _safe_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token.strip("._-") or "unknown"
