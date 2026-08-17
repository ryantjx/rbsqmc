"""Validate downloaded RBPF v3 training artifacts without importing JAX."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED = (
    "progress.log",
    "em_initial_params.json",
    "em_final_params.json",
    "training_arrays.npz",
    "training_summary.json",
    "performance_summary.json",
    "evaluation_summary.json",
    "baseline_comparison.json",
    "optimal_filter/filter_states.npz",
    "optimal_filter/optimal_filter_summary.json",
    "optimal_filter/top_strengths.png",
    "optimal_filter/timeseries_states.png",
    "optimal_filter/correlation_matrix.png",
    "optimal_filter/log_normalizing_constant.png",
    "objective_terms_by_epoch.png",
    "log_marginal_history.png",
    "transition_decomposition.png",
    "smoothed_team_trajectories_with_intervals.png",
    "heldout_log_score_by_date.png",
    "result_calibration.png",
)


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate(directory: Path) -> None:
    for name in REQUIRED:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty required artifact: {name}")
    for name in (
        "em_initial_params.json",
        "em_final_params.json",
        "training_summary.json",
        "performance_summary.json",
        "evaluation_summary.json",
        "baseline_comparison.json",
        "optimal_filter/optimal_filter_summary.json",
    ):
        value = json.loads((directory / name).read_text(encoding="utf-8"))
        if not _finite(value):
            raise ValueError(f"non-finite numeric value in {name}")
    evaluation = json.loads(
        (directory / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    if evaluation.get("hard_failures") or not evaluation.get("passed", False):
        raise ValueError("evaluation_summary.json reports a hard failure")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    validate(args.directory)
    print(f"Validated RBPF v3 artifacts in {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
