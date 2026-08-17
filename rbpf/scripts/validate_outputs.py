"""Validate downloaded RBPF smoothing artifacts without importing JAX.

Checks that all required output files exist, are non-empty, and contain
finite numeric values (for JSON files).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED = (
    "optimized_params.json",
    "run_config.json",
    "em_log_marginal_likelihood_curve.png",
    "filter/top_strengths.png",
    "filter/timeseries_states.png",
    "filter/correlation_matrix.png",
    "filter/initial_correlation_matrix.png",
    "filter/filter_states.npz",
    "filter/log_normalizing_constant.png",
    "smoothing/smoothed_trajectories.png",
    "smoothing/smoothed_uncertainty.png",
)


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate(directory: Path) -> None:
    missing = []
    for name in REQUIRED:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    if missing:
        raise ValueError(f"missing or empty required artifacts: {missing}")

    for name in ("optimized_params.json", "run_config.json"):
        value = json.loads((directory / name).read_text(encoding="utf-8"))
        if not _finite(value):
            raise ValueError(f"non-finite numeric value in {name}")

    params = json.loads((directory / "optimized_params.json").read_text(encoding="utf-8"))
    for key in ("kappa", "alpha", "beta"):
        if key not in params:
            raise ValueError(f"missing key in optimized_params.json: {key}")
        if not math.isfinite(float(params[key])):
            raise ValueError(f"non-finite value for {key}: {params[key]}")

    print(f"Validation passed: {len(REQUIRED)} artifacts OK in {directory}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate RBPF smoothing outputs")
    parser.add_argument("directory", help="Output directory to validate")
    args = parser.parse_args(argv)
    validate(Path(args.directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())