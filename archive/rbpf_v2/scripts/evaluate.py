from __future__ import annotations

import argparse
from pathlib import Path

import jax

from rbpf_v2.src.data import get_results, synthetic_results
from rbpf_v2.src.evaluation import evaluate_run
from rbpf_v2.src.model import run_filter
from rbpf_v2.src.model_trained import load_params
from rbpf_v2.src.smoothing import rb_backward_simulation


def parser():
    p = argparse.ArgumentParser(description="Evaluate a saved RBPF v2 model")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-particles", type=int, default=16)
    p.add_argument("--n-smoother-paths", type=int, default=16)
    p.add_argument("--max-goals", type=int, default=8)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    model_dir = Path(args.model_dir)
    target = Path(args.output_dir) if args.output_dir else model_dir
    _, data, _ = get_results(args.data, max_goals=args.max_goals) if args.data else synthetic_results()
    params = load_params(model_dir / "em_final_params.json")
    key, filter_key, smoother_key = jax.random.split(jax.random.key(args.seed), 3)
    filtered, augmented = run_filter(filter_key, data, params, args.n_particles,
                                     args.max_goals)
    smoothed = rb_backward_simulation(
        smoother_key, filtered, augmented, params, args.n_smoother_paths
    )
    result = {
        "final_params": params, "final_filter_states": filtered,
        "final_augmented_data": augmented,
        "final_smoothed_paths": smoothed.particles.x,
        "backward_probabilities": smoothed.backward_probabilities,
        "backward_component_indices": smoothed.component_indices,
        "final_log_marginal_likelihood": filtered.log_normalizing_constant[-1],
        "config": {"n_particles": args.n_particles,
                   "n_smoother_paths": args.n_smoother_paths},
    }
    report = evaluate_run(result, data, seed=args.seed, output_dir=target)
    print(f"RBPF v2 evaluation complete: passed={report['passed']} ({target})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
