"""Application-level settings helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

OUTPUT_ROOT_ENV_VAR = "EPHYS_ALIGNMENT_OUTPUT_ROOT"


def output_root_from_environment(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the configured default output root, if one is set."""
    environ = os.environ if environ is None else environ
    raw = environ.get(OUTPUT_ROOT_ENV_VAR, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()
