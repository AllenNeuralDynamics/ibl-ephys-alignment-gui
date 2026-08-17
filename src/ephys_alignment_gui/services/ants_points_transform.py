"""Cancellable ANTs point-transform execution."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

ANTS_POINTS_SUBPROCESS_ENV = "EPHYS_ALIGNMENT_ANTS_POINTS_SUBPROCESS"


class CancelTokenLike(Protocol):
    """Cooperative cancellation token shape used by save jobs."""

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        ...

    @property
    def reason(self) -> str | None:
        """Cancellation reason, if one was supplied."""
        ...


class AntsPointTransformCancelled(RuntimeError):
    """Raised when a cancellable ANTs point transform is terminated."""


def apply_transforms_to_points(
    points_xyz: NDArray,
    *,
    dimension: int,
    transforms: Sequence[str],
    whichtoinvert: Sequence[bool],
    cancel_token: CancelTokenLike | None = None,
    poll_interval_s: float = 0.1,
) -> NDArray:
    """Apply ANTs point transforms, allowing cancellation during native work."""
    if _use_subprocess():
        return _apply_transforms_to_points_subprocess(
            points_xyz,
            dimension=dimension,
            transforms=transforms,
            whichtoinvert=whichtoinvert,
            cancel_token=cancel_token,
            poll_interval_s=poll_interval_s,
        )
    return _apply_transforms_to_points_in_process(
        points_xyz,
        dimension=dimension,
        transforms=transforms,
        whichtoinvert=whichtoinvert,
    )


def _apply_transforms_to_points_subprocess(
    points_xyz: NDArray,
    *,
    dimension: int,
    transforms: Sequence[str],
    whichtoinvert: Sequence[bool],
    cancel_token: CancelTokenLike | None,
    poll_interval_s: float,
) -> NDArray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="ephys_alignment_ants_points_") as tmp:
        tmp_path = Path(tmp)
        points_path = tmp_path / "points.npy"
        output_path = tmp_path / "ccf_xyz.npy"
        error_path = tmp_path / "error.json"
        request_path = tmp_path / "request.json"
        stdout_path = tmp_path / "stdout.txt"
        stderr_path = tmp_path / "stderr.txt"

        np.save(points_path, points_xyz, allow_pickle=False)
        request = {
            "dimension": dimension,
            "points_path": str(points_path),
            "transforms": list(transforms),
            "whichtoinvert": [bool(value) for value in whichtoinvert],
            "output_path": str(output_path),
            "error_path": str(error_path),
        }
        with open(request_path, "w") as f:
            json.dump(request, f)

        with open(stdout_path, "w") as stdout, open(stderr_path, "w") as stderr:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "ephys_alignment_gui.services.ants_points_worker",
                    str(request_path),
                ],
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            while process.poll() is None:
                if cancel_token is not None and cancel_token.cancelled:
                    _terminate_process(process)
                    reason = cancel_token.reason or "cancelled"
                    raise AntsPointTransformCancelled(
                        f"ANTs point transform cancelled: {reason}"
                    )
                time.sleep(poll_interval_s)

        if process.returncode != 0:
            stdout_text = _read_text(stdout_path)
            stderr_text = _read_text(stderr_path)
            message = _worker_error_message(error_path, stderr_text)
            if stdout_text:
                logger.debug("ANTs point worker stdout: %s", stdout_text)
            raise RuntimeError(message)
        return np.load(output_path, allow_pickle=False)


def _apply_transforms_to_points_in_process(
    points_xyz: NDArray,
    *,
    dimension: int,
    transforms: Sequence[str],
    whichtoinvert: Sequence[bool],
) -> NDArray:
    import ants
    import pandas

    points_df = pandas.DataFrame(np.asarray(points_xyz), columns=list("xyz"))
    transformed = ants.apply_transforms_to_points(
        dimension,
        points_df,
        list(transforms),
        whichtoinvert=list(whichtoinvert),
    )
    return transformed.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _worker_error_message(error_path: Path, stderr: str) -> str:
    if error_path.exists():
        try:
            with open(error_path) as f:
                error: dict[str, Any] = json.load(f)
            return (
                "ANTs point transform subprocess failed with "
                f"{error.get('type', 'error')}: {error.get('message', '')}"
            )
        except Exception:
            logger.debug("Failed to read ANTs point worker error", exc_info=True)
    return f"ANTs point transform subprocess failed: {stderr.strip()}"


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        logger.debug("Failed to read ANTs point worker output %s", path, exc_info=True)
        return ""


def _use_subprocess() -> bool:
    value = os.environ.get(ANTS_POINTS_SUBPROCESS_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}
