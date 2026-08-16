from __future__ import annotations

import argparse
from pathlib import Path

import jax

from rbpf_v2.src.data import synthetic_results
from rbpf_v2.src.evaluation import evaluate_run
from rbpf_v2.src.helpers import default_init_params
from rbpf_v2.src.model_trained import save_params
from rbpf_v2.src.smoothing import MCEMConfig, run_mcem


def parser():
    p = argparse.ArgumentParser(description="Tiny CPU RBPF v2 end-to-end gate")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-particles", type=int, default=16)
    p.add_argument("--n-smoother-paths", type=int, default=16)
    p.add_argument("--n-epochs", type=int, default=1)
    p.add_argument("--n-gradient-steps", type=int, default=2)
    p.add_argument("--max-goals", type=int, default=8)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    target = Path(args.output_dir)
    if target.exists() and any(target.iterdir()):
        print("Refusing to overwrite a non-empty output directory")
        return 2
    target.mkdir(parents=True, exist_ok=True)
    _, data, teams = synthetic_results()
    config = MCEMConfig(args.n_particles, args.n_smoother_paths, args.n_epochs,
                        args.n_gradient_steps, 1e-3, args.max_goals, 1e-6)
    result = run_mcem(jax.random.key(args.seed), data,
                      default_init_params(len(teams)), config)
    save_params(result["final_params"], target / "em_final_params.json")
    report = evaluate_run(result, data, seed=args.seed, output_dir=target)
    print("Final filter: complete")
    print("RB-aware smoother: complete")
    print(f"Evaluation: passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
