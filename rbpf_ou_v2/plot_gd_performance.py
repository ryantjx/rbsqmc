"""Generate GD performance graphics from saved training outputs.

Reads ``gd_log_marginal_history.json`` and ``gd_loss_history.json`` from a
training output directory and produces a GD performance plot. This runs
entirely on CPU (no GPU / no Colab needed) — it only reads the saved JSONs and
plots them.

Usage:
    python plot_gd_performance.py [--output-dir DIR] [--out PATH]

Defaults:
    --output-dir  rbpf_ou_v2/outputs_gpu_l4
    --out         <output-dir>/gd_performance.png
"""

import argparse
import json
import os
import sys

# Ensure the repo root is importable even when this file is run directly as a
# script (in which case sys.path[0] is the script's own directory, not the root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rbpf_ou_v2.src.graphic import plot_gd_performance


def main():
    parser = argparse.ArgumentParser(description="Plot GD training performance.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rbpf_ou_v2/outputs_gpu_l4",
        help="Directory containing gd_log_marginal_history.json and gd_loss_history.json.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: <output-dir>/gd_performance.png).",
    )
    args = parser.parse_args()

    log_marg_path = os.path.join(args.output_dir, "gd_log_marginal_history.json")
    loss_path = os.path.join(args.output_dir, "gd_loss_history.json")

    if not os.path.exists(log_marg_path):
        print(f"ERROR: {log_marg_path} not found. Run the GD training first.")
        sys.exit(1)

    with open(log_marg_path) as f:
        log_marginal_history = json.load(f)
    loss_history = None
    if os.path.exists(loss_path):
        with open(loss_path) as f:
            loss_history = json.load(f)

    out = args.out or os.path.join(args.output_dir, "gd_performance.png")
    plot_gd_performance(
        log_marginal_history=log_marginal_history,
        loss_history=loss_history,
        output_path=out,
    )
    print(f"Saved GD performance plot to {out}")


if __name__ == "__main__":
    main()
