"""Tests for application settings helpers."""

from __future__ import annotations

from pathlib import Path

from ephys_alignment_gui.core.settings import (
    INPUT_ROOT_ENV_VAR,
    OUTPUT_ROOT_ENV_VAR,
    input_root_from_environment,
    output_root_from_environment,
)


def test_output_root_from_environment_returns_none_when_unset():
    assert output_root_from_environment({}) is None


def test_output_root_from_environment_returns_none_when_blank():
    assert output_root_from_environment({OUTPUT_ROOT_ENV_VAR: "   "}) is None


def test_output_root_from_environment_returns_path():
    assert output_root_from_environment({OUTPUT_ROOT_ENV_VAR: "/tmp/results"}) == Path(
        "/tmp/results"
    )


def test_output_root_from_environment_expands_user():
    assert output_root_from_environment({OUTPUT_ROOT_ENV_VAR: "~/results"}) == Path(
        "~/results"
    ).expanduser()


def test_input_root_from_environment_returns_none_when_unset():
    assert input_root_from_environment({}) is None


def test_input_root_from_environment_returns_none_when_blank():
    assert input_root_from_environment({INPUT_ROOT_ENV_VAR: "   "}) is None


def test_input_root_from_environment_returns_path():
    assert input_root_from_environment({INPUT_ROOT_ENV_VAR: "/tmp/data"}) == Path(
        "/tmp/data"
    )


def test_input_root_from_environment_expands_user():
    assert input_root_from_environment({INPUT_ROOT_ENV_VAR: "~/data"}) == Path(
        "~/data"
    ).expanduser()
