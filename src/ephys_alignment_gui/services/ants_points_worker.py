"""Subprocess entry point for ANTs point transforms."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import ants
import numpy as np
import pandas


def main(argv: list[str] | None = None) -> int:
    """Apply ANTs transforms to points described by a JSON request."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: ants_points_worker <request.json>", file=sys.stderr)
        return 2

    request_path = Path(argv[0])
    with open(request_path) as f:
        request = json.load(f)
    output_path = Path(request["output_path"])
    error_path = Path(request["error_path"])

    try:
        points = np.load(request["points_path"], allow_pickle=False)
        points_df = pandas.DataFrame(points, columns=list("xyz"))
        transformed = ants.apply_transforms_to_points(
            int(request["dimension"]),
            points_df,
            list(request["transforms"]),
            whichtoinvert=list(request["whichtoinvert"]),
        )
        ccf_xyz = transformed.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64)
        np.save(output_path, ccf_xyz, allow_pickle=False)
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with open(error_path, "w") as f:
            json.dump(error, f, indent=2)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
