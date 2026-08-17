"""Generate the saved-alignment output package JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_NAME = "aind-ibl-ephys-alignment-output"
SCHEMA_VERSION = "1.0.0"
SCHEMA_ID = (
    "https://schemas.allenneuraldynamics.org/"
    f"{SCHEMA_NAME}/{SCHEMA_VERSION}/datapackage.schema.json"
)
SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ephys_alignment_gui"
    / "io"
    / "schemas"
    / SCHEMA_NAME
    / SCHEMA_VERSION
    / "datapackage.schema.json"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratedBy(_StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ShankFiles(_StrictModel):
    metadata: str = Field(min_length=1)
    channel_locations: str = Field(min_length=1)
    prev_alignments: str = Field(min_length=1)
    ccf_channel_locations: str = Field(min_length=1)


class Shank(_StrictModel):
    shank_idx: int = Field(ge=0)
    shank_id: int = Field(ge=1)
    files: ShankFiles


class Probe(_StrictModel):
    ephys_collection: str = Field(min_length=1)
    logical_probe: str = Field(min_length=1)
    probe_id: str | None
    n_shanks: int = Field(ge=1)
    shanks: dict[str, Shank]


class Recording(_StrictModel):
    recording_id: str = Field(min_length=1)
    probes: dict[str, Probe]


class AlignmentOutputPackage(_StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        title="AIND IBL Ephys Alignment GUI Output Package",
        json_schema_extra={
            "$id": SCHEMA_ID,
            "description": (
                "Top-level manifest for alignment outputs saved by the "
                "IBL ephys alignment GUI."
            ),
        },
    )

    schema_name: Literal["aind-ibl-ephys-alignment-output"]
    schema_version: Literal["1.0.0"]
    generated_by: GeneratedBy
    mouse_id: str | None
    recordings: dict[str, Recording]


def main() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **AlignmentOutputPackage.model_json_schema(),
    }
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n")


if __name__ == "__main__":
    main()
