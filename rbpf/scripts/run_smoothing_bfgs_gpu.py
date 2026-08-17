"""Bootstrap a Colab GPU and run the BFGS smoothing EM pipeline.

This script runs inside the Colab VM (uploaded by ``colab run``). It:
  1. Clones/updates the repository (or uses an uploaded copy with --no-clone).
  2. Installs runtime dependencies.
  3. Loads the config JSON (smoothing_bfgs_gpu_config.json).
  4. Imports and calls ``run_EM`` from ``rbpf.src.smoothing_bfgs`` directly.
  5. Saves outputs and generates plots.

Usage (local dry-run):
    python rbpf/scripts/run_smoothing_bfgs_gpu.py --config rbpf/scripts/config/smoothing_bfgs_gpu_config.json --dry-run

Usage (Colab, via the orchestrator):
    bash rbpf/scripts/run_smoothing_bfgs_colab.sh
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_DIR = Path("/content/rbsqmc")
DEFAULT_CONFIG = {
    "start_date": "2000-01-01",
    "end_date": "2025-12-31",
    "n_particles": 10000,
    "n_smoother_paths": 10000,
    "n_epochs": 3,
    "max_goals": 8,
    "seed": 0,
    "initial_params": "",
    "output_dir": "rbpf/outputs/smoothing_bfgs",
    "m_step": "bfgs",
    "maxiter": 1,
    "maxcor": 10,
    "gpu_type": "A100",
    "colab_timeout": 7200,
    "repo_url": "https://github.com/ryantjx/rbsqmc.git",
}


def log(message: str, *, stream: str = "OUT") -> None:
    outer = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inner = datetime.now().strftime("%H:%M:%S")
    print(f"[{outer}] {stream}: [{inner}] {message}", flush=True)


def run(command, *, cwd=None, forward_raw=False) -> None:
    log("Running: " + " ".join(map(str, command)))
    process = subprocess.Popen(
        command, cwd=cwd, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in process.stdout:
        if forward_raw:
            print(line, end="", flush=True)
        else:
            log(line.rstrip("\n"))
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}:\n"
            + " ".join(map(str, command))
        )


def is_colab() -> bool:
    return Path("/content").exists()


# ---------------------------------------------------------------------------
# Bootstrap: clone repo + install deps
# ---------------------------------------------------------------------------
def bootstrap(no_clone: bool = False) -> tuple[Path, str]:
    """Clone/update the repository, install uv, and create an isolated venv.

    Returns (repo_root, uv_bin_path).
    """
    if no_clone:
        if not (REPO_DIR / "rbpf").is_dir():
            raise FileNotFoundError(
                f"--no-clone: expected repo at {REPO_DIR} but rbpf/ is missing"
            )
        log(f"Using uploaded repo at {REPO_DIR} (no clone)")
    elif (REPO_DIR / ".git").is_dir():
        log(f"Using existing checkout at {REPO_DIR}")
        run(["git", "-c", "http.version=HTTP/1.1", "pull", "--ff-only"], cwd=REPO_DIR)
    else:
        last_error = None
        for attempt in range(1, 4):
            if REPO_DIR.exists():
                shutil.rmtree(REPO_DIR)
            try:
                run([
                    "git", "-c", "http.version=HTTP/1.1", "clone",
                    "--depth", "1", "--single-branch", "--branch", "main",
                    DEFAULT_CONFIG["repo_url"], str(REPO_DIR),
                ])
                last_error = None
                break
            except RuntimeError as error:
                last_error = error
                log(f"clone attempt {attempt}/3 failed")
        if last_error is not None:
            raise last_error
    # Install uv and create an isolated venv to avoid conflicts with
    # Colab's pre-installed packages (numpy/scipy version mismatches).
    run(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    # uv may install to /usr/local/bin/uv or ~/.local/bin/uv depending on the
    # environment. Use shutil.which to find it, falling back to both paths.
    uv_bin = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv_bin).exists():
        uv_bin = "/usr/local/bin/uv"
    venv_dir = REPO_DIR / ".venv"
    run([uv_bin, "venv", str(venv_dir), "--python", "3.12"])
    run([
        uv_bin, "pip", "install", "-p", str(venv_dir),
        "jax[cuda12]==0.11.0", "cuthbert==0.0.14", "optax==0.2.8",
        "numpy==2.4.6", "scipy==1.18.0",
        "pandas==3.0.5", "pyarrow==25.0.1", "matplotlib==3.11.1",
        "tqdm",
    ])
    return REPO_DIR, uv_bin


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(repo_root: Path, override: str | None = None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if override:
        path = Path(override)
        if not path.is_absolute() and not path.parent.parts:
            path = repo_root / "rbpf/scripts/config" / path
    else:
        path = repo_root / "rbpf/scripts/config/smoothing_bfgs_gpu_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing smoothing configuration: {path}")
    config.update(json.loads(path.read_text(encoding="utf-8")))

    positive_integer = (
        "n_particles", "n_smoother_paths", "n_epochs", "max_goals", "colab_timeout",
    )
    for key in positive_integer:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    return config


# ---------------------------------------------------------------------------
# Training: import and call run_EM from rbpf.src.smoothing_bfgs
# ---------------------------------------------------------------------------
def train(config: dict, repo_root: Path) -> None:
    """Import smoothing_bfgs functions and run the EM pipeline using the config."""
    # Use the uv-created venv's Python instead of the system Python.
    venv_python = str(repo_root / ".venv" / "bin" / "python")
    if Path(venv_python).exists():
        os.environ["RBSQMC_PYTHON"] = venv_python
    else:
        os.environ["RBSQMC_PYTHON"] = sys.executable

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ.setdefault("RBSQMC_PLATFORM", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbpf_matplotlib")

    from rbpf.src.data import get_results, WORLDCUP_2026_TEAMS
    from rbpf.src.helpers import default_init_params, save_params
    from rbpf.src.model import run_filter
    from rbpf.src.smoothing_bfgs import run_EM
    from rbpf.src.smoothing import E_step
    from rbpf.src.graphic import plot_all, plot_log_marginal_likelihood_curve, plot_all_smoothing

    import jax

    key = jax.random.PRNGKey(config["seed"])

    # 1. Load data
    df, model_inputs, team_id_to_name = get_results(
        start_date=config["start_date"],
        end_date=config["end_date"],
        max_goals=config["max_goals"],
        include_friendly=False,
        teams_only=WORLDCUP_2026_TEAMS,
    )
    print(f"Loaded football results data from {config['start_date']} to {config['end_date']}. "
          f"Number of unique dates: {len(df['date'].unique())}. "
          f"Number of unique teams: {len(team_id_to_name)}.")

    # 2. Run EM (BFGS M-step)
    print("[main] Running EM (BFGS M-step)...")
    latest_params, params_history, log_marginal_likelihood_history, mstep_history = run_EM(
        key=key,
        model_inputs=model_inputs,
        params=default_init_params(len(team_id_to_name)),
        n_particles=config["n_particles"],
        n_smoothed_trajectories=config["n_smoother_paths"],
        num_epochs=config["n_epochs"],
        max_goals=config["max_goals"],
        maxiter=config.get("maxiter", 1),
        maxcor=config.get("maxcor", 10),
    )
    print("[main] EM finished.")

    # 3. Save params + run config
    save_path = str(repo_root / config["output_dir"])
    os.makedirs(save_path, exist_ok=True)
    save_params(latest_params, save_path + "/optimized_params.json")
    print("[main] Saved optimized params.")

    run_config = {
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "n_particles": config["n_particles"],
        "n_smoother_paths": config["n_smoother_paths"],
        "n_epochs": config["n_epochs"],
        "max_goals": config["max_goals"],
        "seed": config["seed"],
        "m_step": "bfgs",
        "maxiter": config.get("maxiter", 1),
        "maxcor": config.get("maxcor", 10),
        "output_dir": save_path,
    }
    with open(save_path + "/run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print("[main] Saved run config.")

    # 4. Final filter
    print("[main] Running final filter with optimized params...")
    filtered_states, model_inputs_rbpf = run_filter(
        key=key,
        model_inputs=model_inputs,
        params=latest_params,
        n_particles=config["n_particles"],
        max_goals=config["max_goals"],
    )
    print(f"[main] Final filter done. Final log marginal = {filtered_states.log_normalizing_constant[-1]:.4f}")

    # 5. Plots
    plot_all(
        filtered_states=filtered_states,
        augmented_results=model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=save_path + "/filter",
        params=latest_params,
    )
    print("[main] Saved filter plots.")

    log_marginal_likelihood_history.append(filtered_states.log_normalizing_constant[-1])
    plot_log_marginal_likelihood_curve(
        log_marginal_likelihoods=log_marginal_likelihood_history,
        save_path=save_path + "/em_log_marginal_likelihood_curve.png",
    )
    print("[main] Saved EM log marginal likelihood curve.")

    # 6. Final smoothing pass + plots
    print("[main] Running final smoothing pass...")
    final_smoothed, _ = E_step(
        key=key,
        model_inputs=model_inputs,
        params=latest_params,
        n_particles=config["n_particles"],
        n_smoothed_trajectories=config["n_smoother_paths"],
        max_goals=config["max_goals"],
    )
    plot_all_smoothing(
        smoothed_trajectories=final_smoothed.x,
        team_id_to_name=team_id_to_name,
        df=df,
        top_n=10,
        save_path=save_path + "/smoothing",
    )
    print("[main] Saved smoothing plots.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Colab GPU bootstrap for RBPF BFGS smoothing EM")
    p.add_argument("--config", help="Optional config JSON override")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-clone", action="store_true",
                   help="Skip git clone; use the repo already uploaded to /content/rbsqmc")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    if args.dry_run:
        config = load_config(Path.cwd(), args.config)
        print(f"bfgs: n_particles={config['n_particles']}, n_epochs={config['n_epochs']}")
        return 0

    repo_root, uv_bin = bootstrap(no_clone=args.no_clone) if is_colab() else (Path(__file__).resolve().parents[2], "uv")
    config = load_config(repo_root, args.config)

    # Verify GPU using uv run (uses the isolated .venv)
    run([
        uv_bin, "run", "--project", str(repo_root),
        "python", "-c",
        "import jax; print('JAX devices:', jax.devices()); "
        "assert jax.default_backend() == 'gpu', 'Colab GPU is not active'",
    ], cwd=repo_root)

    log(f"m_step=bfgs, output_dir={config['output_dir']}")
    log(f"Writing remote outputs to {repo_root / config['output_dir']}")

    # Run the training via uv run (uses the isolated .venv, avoiding Colab's
    # pre-installed package conflicts).
    # Write the config path to an env var so train() can read it.
    config_path = Path(args.config) if args.config else repo_root / "rbpf/scripts/config/smoothing_bfgs_gpu_config.json"
    if not config_path.is_absolute():
        config_path = repo_root / "rbpf/scripts/config" / config_path
    os.environ["RBSQMC_CONFIG"] = str(config_path)
    os.environ.setdefault("RBSQMC_PLATFORM", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbpf_matplotlib")
    # Override Colab's MPLBACKEND which points to matplotlib_inline (not
    # installed in the isolated venv). Use 'agg' for headless plotting.
    os.environ["MPLBACKEND"] = "Agg"

    run([
        uv_bin, "run", "--project", str(repo_root),
        "python", "-c",
        "import sys; sys.path.insert(0, '.'); "
        "from rbpf.src.smoothing_bfgs import main; main()",
    ], cwd=repo_root, forward_raw=True)

    log("GPU smoothing completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"{type(error).__name__}: {error}", stream="ERR")
        raise