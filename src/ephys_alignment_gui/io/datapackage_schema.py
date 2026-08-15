"""Vendored datapackage schema validation.

The gate is on the schema's *major* version, not an exact match. Producers can
then ship additive minor bumps without a lockstep GUI release: an unrecognised
minor is validated against the newest bundled schema sharing its major.

That is only sound while minor bumps stay backward compatible **for what this
GUI reads**. A minor may add fields, and may drop a field no consumer reads
(4.1.0 dropped ``histology.ccf_space.registration`` on exactly those grounds).
A minor that removes or repurposes a field the GUI *does* read would pass this
gate and fail later at access time; that needs a major bump.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from importlib.resources import files
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

logger = logging.getLogger(__name__)

SCHEMA_NAME = "aind-ibl-ephys-alignment-datapackage"
#: Bundled schema artifacts, newest last. One per version the producer released.
BUNDLED_SCHEMA_VERSIONS = ("3.0.0", "3.1.0", "3.2.0", "4.0.0", "4.1.0")
#: Majors this GUI understands well enough to read.
SUPPORTED_SCHEMA_MAJORS = (3, 4)
SCHEMA_RESOURCE_PARTS = ("schemas", SCHEMA_NAME)

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class DatapackageContractError(RuntimeError):
    """Raised when raw datapackage JSON does not match a bundled contract."""


def _parse_version(version: str) -> tuple[int, int, int]:
    """Return ``(major, minor, patch)``, or raise for a malformed version."""
    match = _VERSION_RE.match(version)
    if match is None:
        raise DatapackageContractError(
            f"datapackage.json schema_version {version!r} is not MAJOR.MINOR.PATCH"
        )
    return cast(tuple[int, int, int], tuple(int(part) for part in match.groups()))


def _schema_version_for(version: str) -> str:
    """Pick the bundled schema to validate *version* against.

    Exact match when it is bundled. Otherwise the newest bundled schema with the
    same major that does not exceed *version*, so a package is never checked
    against a contract newer than itself.
    """
    if version in BUNDLED_SCHEMA_VERSIONS:
        return version

    target = _parse_version(version)
    candidates = [
        bundled
        for bundled in BUNDLED_SCHEMA_VERSIONS
        if _parse_version(bundled)[0] == target[0] and _parse_version(bundled) <= target
    ]
    if not candidates:
        bundled = ", ".join(BUNDLED_SCHEMA_VERSIONS)
        raise DatapackageContractError(
            f"No bundled datapackage schema can validate {version}. "
            f"GUI bundles: {bundled}."
        )
    return max(candidates, key=_parse_version)


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

    major = _parse_version(version)[0]
    if major not in SUPPORTED_SCHEMA_MAJORS:
        supported = ", ".join(str(m) for m in SUPPORTED_SCHEMA_MAJORS)
        raise DatapackageContractError(
            f"Unsupported datapackage schema {version}. "
            f"GUI supports schema major versions: {supported}."
        )

    schema_version = _schema_version_for(version)
    schema = _load_schema(schema_version)
    if schema_version != version:
        # Every bundled schema const-pins its own schema_version, so validating a
        # newer minor against an older contract would fail on that alone.
        schema = _relax_schema_version_const(schema, major)
        logger.info(
            "datapackage schema %s is newer than any bundled schema; "
            "validating against %s (same major)",
            version,
            schema_version,
        )

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(raw), key=_validation_error_sort_key)
    if errors:
        raise DatapackageContractError(
            _format_validation_error(schema_version, errors[0])
        )


def _relax_schema_version_const(schema: dict[str, Any], major: int) -> dict[str, Any]:
    """Widen the ``schema_version`` const to any version of the same major."""
    relaxed = copy.deepcopy(schema)
    field = relaxed.get("properties", {}).get("schema_version")
    if isinstance(field, dict):
        field.pop("const", None)
        field["pattern"] = rf"^{major}\.\d+\.\d+$"
    return relaxed


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
