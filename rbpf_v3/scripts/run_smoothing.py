"""Standalone Cuthbert-backed RBPF v3 training and evaluation runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbpf_v3_matplotlib")

import jax
import numpy as np

from rbpf_v3.src.data import WORLDCUP_2026_TEAMS, get_results
from rbpf_v3.src.evaluation import evaluate_run, tree_to_python
from rbpf_v3.src.helpers import default_init_params, load_params, save_params
from rbpf_v3.src.progress import progress
from rbpf_v3.src.smoothing import MCEMConfig, run_mcem
from rbpf_v3.src.utils import EMParams


BACKEND = "cuthbert"
DEFAULT_OUTPUT = "rbpf_v3/outputs/smoothing_cuthbert"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Run the Cuthbert RBPF v3 smoother")
    command.add_argument("--start-date", default="2000-01-01")
    command.add_argument("--end-date", default="2026-01-01")
    command.add_argument("--initial-params")
    command.add_argument("--n-particles", type=int, default=50)
    command.add_argument("--n-smoother-paths", type=int, default=50)
    command.add_argument("--n-epochs", type=int, default=5)
    command.add_argument("--n-gradient-steps", type=int, default=20)
    command.add_argument("--learning-rate", type=float, default=1e-3)
    command.add_argument("--max-goals", type=int, default=8)
    command.add_argument("--holdout-days", type=int, default=1)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--path-batch-size", type=int, default=32)
    command.add_argument("--log-every-gradient-steps", type=int, default=5)
    command.add_argument("--return-backward-probabilities", action="store_true")
    command.add_argument("--profile", action="store_true")
    command.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    command.add_argument("--synthetic", action="store_true", help=argparse.SUPPRESS)
    return command


def _slice_results(results, start=None, stop=None):
    return jax.tree.map(lambda value: value[slice(start, stop)], results)


def _synthetic_results():
    import pandas as pd

    from rbpf_v3.src.helpers import generate_results_jax

    rows = []
    for timestamp, date, scores in zip(
        (1, 3, 8, 10),
        pd.to_datetime(("2024-01-01", "2024-01-03", "2024-01-08", "2024-01-10")),
        ((1, 0), (0, 0), (2, 1), (1, 2)),
    ):
        rows.append(
            {
                "date": date,
                "timestamp": timestamp,
                "home_team": "A",
                "away_team": "B",
                "home_id": 0,
                "away_id": 1,
                "home_score": scores[0],
                "away_score": scores[1],
                "tournament": "Synthetic",
            }
        )
    frame, data = generate_results_jax(pd.DataFrame(rows))
    return frame, data, {0: "A", 1: "B"}


def _validate_initial_params(params: EMParams, num_teams: int) -> None:
    if params.mean_0.shape != (num_teams, 2):
        raise ValueError(f"mean_0 must have shape {(num_teams, 2)}")
    if params.gamma_0.shape != (num_teams, num_teams):
        raise ValueError("gamma_0 team dimension does not match the data")
    if params.B.shape != (2, 2):
        raise ValueError("B must have shape (2,2)")
    if np.linalg.eigvalsh(np.asarray(params.gamma_0)).min() <= 0:
        raise ValueError("gamma_0 must be positive definite")
    if np.linalg.eigvalsh(np.asarray(params.B)).min() <= 0:
        raise ValueError("B must be positive definite")
    if float(params.kappa) <= 0:
        raise ValueError("kappa must be positive")


def _load_inputs(args):
    if args.synthetic:
        frame, data, names = _synthetic_results()
    else:
        frame, data, names = get_results(
            start_date=args.start_date,
            end_date=args.end_date,
            max_goals=args.max_goals,
            include_friendly=False,
            teams_only=WORLDCUP_2026_TEAMS,
        )
    params = (
        load_params(args.initial_params)
        if args.initial_params
        else default_init_params(len(names), team_id_to_name=names)
    )
    _validate_initial_params(params, len(names))
    return frame, data, names, params


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(tree_to_python(value), indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    total_start = time.perf_counter()
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    os.environ["RBSQMC_PROGRESS_LOG"] = str(target / "progress.log")
    try:
        progress("loading data and initial parameters...")
        _, data, team_names, initial_params = _load_inputs(args)
        n_days = int(data.timestamp.size)
        if args.holdout_days < 0 or args.holdout_days >= n_days:
            raise ValueError("holdout-days must be non-negative and smaller than observed days")
        holdout = _slice_results(data, -args.holdout_days, None) if args.holdout_days else None
        train_data = _slice_results(data, None, -args.holdout_days) if args.holdout_days else data
        progress(f"backend={BACKEND} device={jax.devices()[0]}")
        progress(
            f"dimensions D={train_data.timestamp.size}, N={args.n_particles}, "
            f"S={args.n_smoother_paths}, M={len(team_names)}"
        )
        save_params(initial_params, str(target / "em_initial_params.json"))
        config = MCEMConfig(
            n_filter_particles=args.n_particles,
            n_smoother_particles=args.n_smoother_paths,
            n_epochs=args.n_epochs,
            n_gradient_steps=args.n_gradient_steps,
            learning_rate=args.learning_rate,
            max_goals=args.max_goals,
            path_batch_size=args.path_batch_size,
            log_every_gradient_steps=args.log_every_gradient_steps,
            return_backward_probabilities=args.return_backward_probabilities,
            profile=args.profile,
        )
        training_start = time.perf_counter()
        result = run_mcem(jax.random.key(args.seed), train_data, initial_params, config)
        jax.block_until_ready(result["final_smoothed_states"].x)
        training_seconds = time.perf_counter() - training_start
        save_params(result["final_params"], str(target / "em_final_params.json"))
        smoothed = result["final_smoothed_states"]
        arrays = {
            "paths": np.asarray(smoothed.x),
            "indices": np.asarray(smoothed.component_indices),
            "ess": np.asarray(smoothed.diagnostics.ess_by_time),
            "entropy": np.asarray(smoothed.diagnostics.entropy_by_time),
        }
        if smoothed.diagnostics.probabilities is not None:
            arrays["probabilities"] = np.asarray(smoothed.diagnostics.probabilities)
        np.savez_compressed(target / "training_arrays.npz", **arrays)
        summary = {
            "backend": BACKEND,
            "seed": args.seed,
            "data": {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "n_train_days": int(train_data.timestamp.size),
                "n_holdout_days": args.holdout_days,
                "n_teams": len(team_names),
            },
            "config": result["config"],
            "params_history": result["params_history"],
            "mstep_history": result["mstep_history"],
            "log_marginal_history": result["log_marginal_history"],
            "diagnostics_history": result["diagnostics_history"],
            "final_log_marginal_likelihood": result["final_log_marginal_likelihood"],
        }
        _write_json(target / "training_summary.json", summary)
        progress("running final evaluation and artifact generation...")
        evaluation_start = time.perf_counter()
        report = evaluate_run(
            result, train_data, holdout, seed=args.seed, output_dir=target
        )
        evaluation_seconds = time.perf_counter() - evaluation_start
        performance = {
            "backend": BACKEND,
            "seed": args.seed,
            "device": str(jax.devices()[0]),
            "D": int(train_data.timestamp.size),
            "N": args.n_particles,
            "S": args.n_smoother_paths,
            "M": len(team_names),
            "timing_history": result["timing_history"],
            "final_e_step": result["final_timing"],
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - total_start,
        }
        _write_json(target / "performance_summary.json", performance)
        if args.profile:
            jax.profiler.save_device_memory_profile(str(target / "device_memory.prof"))
        progress(f"artifacts saved to {target}")
        progress(f"evaluation passed={report['passed']}")
        return 0 if report["passed"] else 1
    except Exception as error:
        progress(f"{type(error).__name__}: {error}", stream="ERR")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
