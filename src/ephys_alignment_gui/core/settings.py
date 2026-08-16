"""Application-level settings helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

OUTPUT_ROOT_ENV_VAR = "EPHYS_ALIGNMENT_OUTPUT_ROOT"
INPUT_ROOT_ENV_VAR = "EPHYS_ALIGNMENT_INPUT_ROOT"
MAX_CACHED_STREAMS_ENV_VAR = "EPHYS_ALIGNMENT_MAX_CACHED_STREAMS"
DEFAULT_MAX_CACHED_STREAMS = 3


def output_root_from_environment(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the configured default output root, if one is set."""
    return _path_from_environment(OUTPUT_ROOT_ENV_VAR, environ)


def input_root_from_environment(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the configured default input-browser root, if one is set."""
    return _path_from_environment(INPUT_ROOT_ENV_VAR, environ)


def max_cached_streams_from_environment(
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """Return the configured stream-runtime cache limit."""
    environ = os.environ if environ is None else environ
    raw = environ.get(MAX_CACHED_STREAMS_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_MAX_CACHED_STREAMS
    if raw.lower() in {"none", "unbounded"}:
        return None
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CACHED_STREAMS
    if value < 1:
        return DEFAULT_MAX_CACHED_STREAMS
    return value


def _path_from_environment(
    name: str,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return an expanded path from an environment variable."""
    environ = os.environ if environ is None else environ
    raw = environ.get(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()
