"""Allen region lookup helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import one.alf.io as alfio
from iblatlas import atlas


class RegionLookupService:
    """Load Allen structure metadata and describe region IDs."""

    def __init__(self) -> None:
        self.allen: Any | None = None

    def load_allen_csv(self) -> Any:
        """Load the Allen structure tree bundled with iblatlas."""
        if self.allen is not None:
            return self.allen
        allen_path = Path(Path(atlas.__file__).parent, "allen_structure_tree.csv")
        self.allen = alfio.load_file_content(allen_path)
        return self.allen

    def get_region_description(self, region_idx: int) -> tuple[str, str]:
        """Return user-facing description and lookup label for a region ID."""
        allen = self.allen if self.allen is not None else self.load_allen_csv()
        struct_idx = np.where(allen["id"] == region_idx)[0][0]
        description = ""
        region_lookup = allen["acronym"][struct_idx] + ": " + allen["name"][struct_idx]

        if region_lookup == "void: void":
            region_lookup = "root: root"

        if not description:
            description = region_lookup + "\nNo information available for this region"
        else:
            description = region_lookup + "\n" + description

        return description, region_lookup
