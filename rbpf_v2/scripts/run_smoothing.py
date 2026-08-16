"""Run the corrected v2 smoother with the configuration used by rbpf/smoothing.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbpf_v2_matplotlib")

import jax
import numpy as np

from rbpf.src.data import WORLDCUP_2026_TEAMS, get_results as get_rbpf_results
from rbpf.src.helpers import default_init_params as rbpf_default_init_params
from rbpf_v2.src.data import slice_results, synthetic_results
from rbpf_v2.src.evaluation import evaluate_run
from rbpf_v2.src.model_trained import load_params, save_params
from rbpf_v2.src.smoothing import MCEMConfig, run_mcem
from rbpf_v2.src.utils import EMParams, tree_to_python


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run RBPF v2 using the parameter choices from rbpf/src/smoothing.py"
    )
    p.add_argument("--start-date", default="2000-01-01")
    p.add_argument("--end-date", default="2026-01-01")
    p.add_argument("--output-dir", default="rbpf_v2/outputs/smoothing")
    p.add_argument(
        "--initial-params",
        help="Optional EMParams JSON; defaults to rbpf/src/smoothing.py initialization",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-particles", type=int, default=50)
    p.add_argument("--n-smoother-paths", type=int, default=50)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--n-gradient-steps", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--max-goals", type=int, default=8)
    p.add_argument(
        "--holdout-days", type=int, default=1,
        help="Number of final match days reserved for chronological evaluation",
    )
    p.add_argument("--synthetic", action="store_true", help=argparse.SUPPRESS)
    return p


def _load_inputs(args):
    if args.synthetic:
        frame, data, team_id_to_name = synthetic_results()
        params = (
            load_params(args.initial_params)
            if args.initial_params
            else EMParams(*rbpf_default_init_params(len(team_id_to_name)))
        )
        _validate_initial_params(params, len(team_id_to_name))
        return frame, data, team_id_to_name, params

    frame, data, team_id_to_name = get_rbpf_results(
        start_date=args.start_date,
        end_date=args.end_date,
        max_goals=args.max_goals,
        include_friendly=False,
        teams_only=WORLDCUP_2026_TEAMS,
    )
    params = (
        load_params(args.initial_params)
        if args.initial_params
        else EMParams(*rbpf_default_init_params(
            num_teams=len(team_id_to_name), team_id_to_name=team_id_to_name
        ))
    )
    _validate_initial_params(params, len(team_id_to_name))
    return frame, data, team_id_to_name, params


def _validate_initial_params(params: EMParams, num_teams: int) -> None:
    if params.mean_0.shape != (num_teams, 2):
        raise ValueError(
            f"initial mean_0 must have shape {(num_teams, 2)}, got {params.mean_0.shape}"
        )
    if params.gamma_0.shape != (num_teams, num_teams):
        raise ValueError(
            "initial gamma_0 team dimension does not match the selected data"
        )
    if params.B.shape != (2, 2):
        raise ValueError("initial B must have shape (2, 2)")
    gamma_eigenvalues = np.linalg.eigvalsh(np.asarray(params.gamma_0))
    b_eigenvalues = np.linalg.eigvalsh(np.asarray(params.B))
    if gamma_eigenvalues.min() <= 0 or b_eigenvalues.min() <= 0:
        raise ValueError("initial gamma_0 and B must be positive definite")
    if float(params.kappa) <= 0:
        raise ValueError("initial kappa must be positive")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    import time
    def log(msg): print(f"[run_smoothing {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    log("main: loading inputs (get_rbpf_results / prepare_results)...")
    frame, data, team_id_to_name, initial_params = _load_inputs(args)
    log(f"main: inputs loaded, {len(team_id_to_name)} teams, "
        f"data.timestamp.size={int(data.timestamp.size)}")
    n_days = int(data.timestamp.size)
    if args.holdout_days < 0 or args.holdout_days >= n_days:
        raise ValueError("holdout-days must be non-negative and smaller than observed days")
    holdout = slice_results(data, -args.holdout_days, None) if args.holdout_days else None
    train_data = slice_results(data, None, -args.holdout_days) if args.holdout_days else data
    log(f"main: train_data.timestamp.size={int(train_data.timestamp.size)}")

    config = MCEMConfig(
        n_filter_particles=args.n_particles,
        n_smoother_particles=args.n_smoother_paths,
        n_epochs=args.n_epochs,
        n_gradient_steps=args.n_gradient_steps,
        learning_rate=args.learning_rate,
        max_goals=args.max_goals,
        acceptance_tolerance=1e-6,
    )
    log("main: calling run_mcem...")
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    save_params(initial_params, target / "em_initial_params.json")
    result = run_mcem(
        jax.random.key(args.seed), train_data, initial_params, config
    )

    save_params(result["final_params"], target / "em_final_params.json")
    np.savez_compressed(
        target / "training_arrays.npz",
        paths=result["final_smoothed_paths"],
        probabilities=result["backward_probabilities"],
        indices=result["backward_component_indices"],
    )
    summary = {
        "data": {
            "start_date": args.start_date, "end_date": args.end_date,
            "n_train_days": int(train_data.timestamp.size),
            "n_holdout_days": args.holdout_days,
            "n_teams": len(team_id_to_name),
        },
        "seed": args.seed, "config": config._asdict(),
        "params_history": result["params_history"],
        "mstep_history": result["mstep_history"],
        "log_marginal_history": result["log_marginal_history"],
        "diagnostics_history": result["diagnostics_history"],
        "final_log_marginal_likelihood": result["final_log_marginal_likelihood"],
    }
    (target / "training_summary.json").write_text(
        json.dumps(tree_to_python(summary), indent=2)
    )
    report = evaluate_run(
        result, train_data, holdout, seed=args.seed, output_dir=target
    )
    print(
        f"RBPF v2 smoothing complete: {len(team_id_to_name)} teams, "
        f"{train_data.timestamp.size} training days, passed={report['passed']}"
    )
    print(f"Optimized parameters: {target / 'em_final_params.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
