"""Validate all SQMC–EKF comparison artifacts after download.

Checks that the timestamped run directory contains both methods' summaries,
prediction records, metrics, CSVs, plots, and report, and that every numeric
value is finite. Fails loudly with diagnostics on any missing or non-finite
artifact.

The run layout is:
    outputs/DDMMYYYY_HHMM/
        results/   (JSON/CSV/MD: configs, metadata, metrics, predictions, report)
        images/    (PNG plots)

Usage:
    python -m rbsqmc.comparison.sqmc_ekf.scripts.validate_sqmc_ekf_outputs [OUTPUTS_DIR]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np


REQUIRED_METHODS = ("ekf", "sqmc")

# Files that must exist under results/ for each method.
REQUIRED_RESULTS = (
    "comparison_config.json",
    "run_config.json",
    "dataset_metadata.json",
    "run_metadata.json",
    "summary.json",
    "performance_metrics.csv",
    "logz_history.csv",
    "REPORT.md",
    "DRAFT.md",
)

# Per-method JSON artifacts under results/.
REQUIRED_METHOD_JSON = (
    "predictions.json",
    "metrics.json",
)

# Per-method images under images/.
REQUIRED_METHOD_IMAGES = (
    "top5_strengths.png",
    "timeseries_states.png",
    "pre_worldcup_rankings.png",
    "post_worldcup_rankings.png",
)


def _latest_run_dir(outputs_dir: str) -> Path:
    dirs = [
        p for p in Path(outputs_dir).iterdir()
        if p.is_dir() and p.name[:8].isdigit()
    ]
    if not dirs:
        raise FileNotFoundError(f"No timestamped run directory under {outputs_dir}")
    return max(dirs, key=lambda p: p.name)


def _json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _ensure_finite(label: str, tree) -> None:
    import numbers

    def walk(v):
        if isinstance(v, (dict, list, tuple)):
            for x in (v.values() if isinstance(v, dict) else v):
                walk(x)
        elif isinstance(v, numbers.Number):
            if not np.isfinite(float(v)):
                raise ValueError(f"{label}: non-finite value {v}")

    walk(tree)


def validate_run(outputs_dir: str) -> None:
    run_dir = _latest_run_dir(outputs_dir)
    results_dir = run_dir / "results"
    images_dir = run_dir / "images"
    print(f"Validating run directory: {run_dir}")

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    # Top-level results files.
    for filename in REQUIRED_RESULTS:
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {filename} in {results_dir}")
        if path.suffix == ".json":
            _ensure_finite(f"results/{filename}", _json(path))

    # Per-method JSON + images.
    for method in REQUIRED_METHODS:
        for filename in REQUIRED_METHOD_JSON:
            path = results_dir / f"{method}_{filename}"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path.name}")
            _ensure_finite(f"results/{path.name}", _json(path))
        for filename in REQUIRED_METHOD_IMAGES:
            path = images_dir / f"{method}_{filename}"
            if not path.exists():
                raise FileNotFoundError(f"Missing plot {path}")

    # Per-method summary.json under results/<method>/.
    for method in REQUIRED_METHODS:
        path = results_dir / method / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        _ensure_finite(f"results/{method}/summary.json", _json(path))

    # Cross-method epoch alignment from logz_history.csv.
    import csv as _csv

    with open(results_dir / "logz_history.csv") as f:
        rows = list(_csv.DictReader(f))
    n_e = sum(1 for r in rows if r["method"] == "ekf")
    n_s = sum(1 for r in rows if r["method"] == "sqmc")
    if n_e != n_s:
        raise ValueError(f"logz_history epoch mismatch: EKF={n_e}, SQMC={n_s}")

    print(f"OK: run directory {run_dir} validated")


def main() -> None:
    outputs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "outputs"
    )
    validate_run(outputs_dir)


if __name__ == "__main__":
    main()
