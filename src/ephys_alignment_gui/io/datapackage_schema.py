"""Vendored datapackage schema validation."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

SCHEMA_NAME = "aind-ibl-ephys-alignment-datapackage"
SUPPORTED_SCHEMA_VERSIONS = ("3.1.0",)
SCHEMA_RESOURCE_PARTS = ("schemas", SCHEMA_NAME)


class DatapackageContractError(RuntimeError):
    """Raised when raw datapackage JSON does not match a bundled contract."""


def validate_datapackage_contract(raw: Any) -> None:
    """Validate raw datapackage JSON against a bundled schema contract."""
    if not isinstance(raw, dict):
        raise DatapackageContractError("datapackage.json must contain a JSON object")

    version = raw.get("schema_version")
    if version is None:
        raise DatapackageContractError("datapackage.json has no schema_version")
    if not isinstance(version, str):
        raise DatapackageContractError(
            f"datapackage.json schema_version must be a string, got "
            f"{type(version).__name__}"
        )
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(SUPPORTED_SCHEMA_VERSIONS)
        raise DatapackageContractError(
            f"Unsupported datapackage schema {version}. "
            f"GUI supports bundled schemas: {supported}."
        )

    schema = _load_schema(version)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(raw), key=_validation_error_sort_key)
    if errors:
        raise DatapackageContractError(_format_validation_error(version, errors[0]))


def _load_schema(version: str) -> dict[str, Any]:
    resource = _schema_resource(version)
    loaded = json.loads(resource.read_text())
    if not isinstance(loaded, dict):
        raise DatapackageContractError(
            f"Vendored datapackage schema {version} must contain a JSON object"
        )
    return cast(dict[str, Any], loaded)


def _schema_resource(version: str) -> Any:
    resource = files("ephys_alignment_gui.io")
    for part in (*SCHEMA_RESOURCE_PARTS, version, "datapackage.schema.json"):
        resource = resource.joinpath(part)
    return resource


def _validation_error_sort_key(error: ValidationError) -> tuple[str, ...]:
    return tuple(str(part) for part in error.absolute_path)


def _format_validation_error(version: str, error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    return (
        f"datapackage.json does not match vendored schema {version} "
        f"at {path}: {error.message}"
    )
