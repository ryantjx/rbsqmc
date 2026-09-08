"""Reserved SQMC benchmark entry point.

The previous SQMC CPU/GPU comparison has intentionally been removed. This
module keeps the command-line and output contract available while a new
benchmark design is being prepared.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = "sqmc/sqmc/scripts/outputs/sqmc_gpu"
DEFAULT_PARTICLE_COUNTS = [64, 256, 1024, 4096, 16384, 65536]
DEFAULT_N_STEPS = 100
DEFAULT_N_REPS = 7
DEFAULT_N_WARMUPS = 2
DEFAULT_BASE_DIMENSION = 10
DEFAULT_SEED = 42


def capture_hardware() -> dict:
    """Capture metadata without running a benchmark."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _write_placeholder_figure(path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        "No benchmark comparison has been defined yet.",
        ha="center",
        va="center",
        fontsize=14,
    )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="SQMC benchmark scaffold (comparison disabled)."
    )
    argument_parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    argument_parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    argument_parser.add_argument("--warmups", type=int, default=DEFAULT_N_WARMUPS)
    argument_parser.add_argument(
        "--particle-counts", type=int, nargs="+", default=DEFAULT_PARTICLE_COUNTS
    )
    argument_parser.add_argument(
        "--base-dimension", type=int, default=DEFAULT_BASE_DIMENSION
    )
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    argument_parser.add_argument("--platforms", nargs="+", default=["gpu"])
    argument_parser.add_argument(
        "--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR)
    )
    argument_parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    argument_parser.add_argument("--_results-json", type=Path, help=argparse.SUPPRESS)
    return argument_parser


def _config(args) -> dict:
    return {
        "n_steps": args.n_steps,
        "n_reps": args.n_reps,
        "warmups": args.warmups,
        "particle_counts": args.particle_counts,
        "base_dimension": args.base_dimension,
        "seed": args.seed,
        "platforms": args.platforms,
        "status": "comparison_not_defined",
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args._child:
        payload = {"config": _config(args), "hardware": capture_hardware()}
        encoded = json.dumps(payload)
        if args._results_json:
            args._results_json.parent.mkdir(parents=True, exist_ok=True)
            args._results_json.write_text(encoded, encoding="utf-8")
        else:
            print(encoded)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _config(args)
    payload = {"config": config, "hardware": capture_hardware()}
    (args.output_dir / "sqmc_gpu_vs_cpu.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_placeholder_figure(
        args.output_dir / "sqmc_gpu_vs_cpu.png", "SQMC benchmark scaffold"
    )
    print(f"Saved benchmark scaffold outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
