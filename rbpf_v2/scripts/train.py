from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from rbpf_v2.src.data import get_results, slice_results, synthetic_results
from rbpf_v2.src.helpers import default_init_params
from rbpf_v2.src.evaluation import evaluate_run
from rbpf_v2.src.model_trained import save_params
from rbpf_v2.src.smoothing import MCEMConfig, run_mcem
from rbpf_v2.src.utils import tree_to_python


def parser():
    p = argparse.ArgumentParser(description="Train the RB-aware football smoother")
    p.add_argument("--data")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-particles", type=int, default=16)
    p.add_argument("--n-smoother-paths", type=int, default=16)
    p.add_argument("--n-epochs", type=int, default=1)
    p.add_argument("--n-gradient-steps", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--max-goals", type=int, default=8)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _, data, teams = get_results(args.data, max_goals=args.max_goals) if args.data else synthetic_results()
    holdout = slice_results(data, -1, None) if data.timestamp.size > 1 else None
    train_data = slice_results(data, None, -1) if holdout is not None else data
    config = MCEMConfig(args.n_particles, args.n_smoother_paths, args.n_epochs,
                        args.n_gradient_steps, args.learning_rate, args.max_goals, 1e-6)
    result = run_mcem(jax.random.key(args.seed), train_data,
                      default_init_params(len(teams)), config)
    save_params(result["final_params"], target / "em_final_params.json")
    serializable = {
        "seed": args.seed, "config": config._asdict(),
        "params_history": result["params_history"],
        "mstep_history": result["mstep_history"],
        "log_marginal_history": result["log_marginal_history"],
        "diagnostics_history": result["diagnostics_history"],
        "final_log_marginal_likelihood": result["final_log_marginal_likelihood"],
    }
    (target / "training_summary.json").write_text(json.dumps(tree_to_python(serializable), indent=2))
    # A compact array checkpoint makes the separate evaluation command reproducible.
    import numpy as np
    np.savez_compressed(target / "training_arrays.npz",
                        paths=result["final_smoothed_paths"],
                        probabilities=result["backward_probabilities"],
                        indices=result["backward_component_indices"])
    evaluate_run(result, train_data, holdout, seed=args.seed, output_dir=target)
    print(f"RBPF v2 training complete: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
