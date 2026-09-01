"""Temporary backfill: run the observe phase for the 20260827_2025 comparison.

Loads each method's best_params.json and writes the post-prediction observe
artifacts (observe_team_states*.png/json + final_filter_after_prediction/) into
that method's prediction/ directory, mirroring train_model.py's run_observe step.

Usage:
    python -m rbsqmc.scripts._backfill_observe_20260827_2025
"""

import os

from rbsqmc.src.model.observe import run_observe
from rbsqmc.src.utils.helpers import load_params

COMPARE_DIR = os.path.join("rbsqmc", "outputs", "compare", "20260827_2025")

# Match the config used by the 20260827_2025 comparison run.
CFG = {
    "training_start_date": "1980-01-01",
    "test_start_date": "2024-01-01",
    "prediction_start_date": "2026-06-11",
    "n_particles": 256,
    "max_goals": 8,
    "seed": 0,
    "include_friendly": True,
    "teams": "worldcup2026",
}


def main() -> None:
    for method in ("smc", "sqmc"):
        method_dir = os.path.join(COMPARE_DIR, method)
        params_path = os.path.join(method_dir, "best_params.json")
        params = load_params(params_path)
        output_dir = os.path.join(method_dir, "prediction")
        print(f"\n{'=' * 60}\nBackfilling observe for {method.upper()}\n{'=' * 60}")
        run_observe(
            cfg=CFG,
            params=params,
            output_dir=output_dir,
            method=method,
        )


if __name__ == "__main__":
    main()
