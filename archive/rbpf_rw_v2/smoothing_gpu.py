"""Run the RBPF direct-gradient-descent (v2) on a GPU/TPU accelerator, locally or on Colab.

This single script combines:
  * the Colab bootstrap (clone repo, install deps, verify accelerator), and
  * the GD core (particle filter + direct gradient descent on -log Z).

It reuses the GD machinery from ``rbpf_rw_v2/src/train.py`` (parameter set:
estimates ``gamma_0``, ``gamma_Q``, ``B``, ``alpha``, ``beta``; ``mean_0``
fixed). Runtime configuration (N, steps, dates, teams, hardware) is read from
``smoothing_gpu_config.json``.

On Colab the repo is cloned/updated and the ``rbpf_rw_v2`` package is made
importable *after* the bootstrap, which is why ``rbpf_rw_v2`` imports happen
lazily inside the GD function rather than at module top level.

Usage:
    # Local (repo already present):
    python smoothing_gpu.py [GPU_N] [START_DATE]

    # Colab (auto-detected via /content, or force with --colab):
    colab run --gpu T4 --keep smoothing_gpu.py [GPU_N] [START_DATE]
    colab run --tpu v5e1 --keep smoothing_gpu.py [GPU_N] [START_DATE]

Writes (into ``rbpf_rw_v2/outputs_gpu``):
  - gd_params_init.json
  - gd_params_final.json
  - gd_log_marginal_history.json
  - gd_loss_history.json
"""

import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Shared configuration (single source of truth)
# ---------------------------------------------------------------------------
# ``colab run`` executes this file as notebook cells, where ``__file__`` is
# undefined. Fall back to the current working directory in that case.
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
# Repo root is the parent of the rbpf_rw_v2 package directory (e.g. .../rbsqmc).
_REPO_ROOT = os.path.dirname(_HERE)
# Ensure the repo root is importable even when this file is run directly as a
# script (in which case sys.path[0] is the script's own directory, not the root).
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

REPO_DIR = "/content/rbsqmc"
TEST_DIR = os.path.join(REPO_DIR, "rbpf_rw_v2")

# Defaults so the module imports even when the config file is absent (which is
# the case on a fresh Colab VM, where `colab run` uploads only this script).
_DEFAULT_CONFIG = {
    "N": 200,
    "n_steps": 200,
    "learning_rate": 0.01,
    "start_date": "1950-01-01",
    "end_date": "2025-12-31",
    "teams": "WORLDCUP_2026_TEAMS",
    "max_goals": 8,
    "output_dir": "rbpf_rw_v2/outputs_gpu",
    "hardware": "gpu",
    "gpu_type": "T4",
    "tpu_type": "v5e1",
    "colab_timeout": 3600,
    "repo_url": "https://github.com/ryantjx/rbsqmc.git",
}
CONFIG_PATH = os.environ.get(
    "RBSQMC_CONFIG", os.path.join(_HERE, "smoothing_gpu_config.json")
)


def _candidate_config_paths() -> list[str]:
    """Locations to look for the config file, in priority order.

    On Colab, ``colab run`` uploads this script to ``/content/`` and may leave a
    *stale* ``/content/smoothing_gpu_config.json`` from a previous run. The
    committed config in the cloned repo (``TEST_DIR``) is authoritative, so it
    is checked FIRST. ``CONFIG_PATH`` (the uploaded script's directory) is only
    a fallback for local runs where the repo config is not present.
    """
    paths = []
    if os.path.abspath(TEST_DIR) != os.path.abspath(_HERE):
        paths.append(os.path.join(TEST_DIR, "smoothing_gpu_config.json"))
    paths.append(CONFIG_PATH)
    return paths


def _load_config() -> dict:
    """Load the config JSON, falling back to defaults if it is unavailable."""
    for path in _candidate_config_paths():
        if os.path.exists(path):
            with open(path, "r") as _f:
                return json.load(_f)
    return dict(_DEFAULT_CONFIG)


CONFIG = _load_config()
REPO_URL = str(CONFIG["repo_url"])


def _repo_root() -> str:
    """Repository root for the current environment.

    Locally ``_REPO_ROOT`` is the parent of this script's directory. On a Colab
    VM the repo is cloned to ``REPO_DIR`` (``/content/rbsqmc``) and the working
    directory is moved there during ``bootstrap``, so that is the true root.
    """
    if os.path.exists(os.path.join(REPO_DIR, "rbpf")):
        return REPO_DIR
    return _REPO_ROOT


OUTPUT_DIR = os.path.join(_repo_root(), CONFIG["output_dir"])

DEPS = [
    "jax", "jaxlib", "numpy", "scipy", "polars", "pandas",
    "matplotlib", "seaborn", "altair", "numba", "tqdm", "cuthbert", "optax",
]


# ---------------------------------------------------------------------------
# Logging / subprocess helpers
# ---------------------------------------------------------------------------
def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kwargs):
    log(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): "
            f"{' '.join(cmd) if isinstance(cmd, list) else cmd}"
        )
    return result


def run_streaming(cmd):
    log(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): "
            f"{' '.join(cmd) if isinstance(cmd, list) else cmd}"
        )


def is_colab() -> bool:
    """True when running on a Colab VM (or /content exists)."""
    return os.path.exists("/content") or "--colab" in sys.argv


def bootstrap():
    """Clone/update the repo, install deps, and cd into the test package dir.

    Only meaningful on a fresh Colab VM. On a local checkout (repo already
    present) this is effectively a no-op except for dependency install.
    """
    global CONFIG, OUTPUT_DIR  # reassigned after the repo is cloned

    log("=" * 60)
    log("SMOOTHING RUNNER — BOOTSTRAP")
    log("=" * 60)

    if os.path.exists(REPO_DIR):
        log(f"Repo already exists at {REPO_DIR}, resetting to origin/main...")
        # Force-reset to the latest remote commit so the committed config
        # (dates, teams, GPU) always takes effect, even if a previous run left
        # local changes or a stale checkout on the VM.
        run(["git", "-C", REPO_DIR, "fetch", "origin"], check=False)
        run(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"], check=False)
    else:
        log(f"Cloning {REPO_URL} → {REPO_DIR}")
        run(["git", "clone", REPO_URL, REPO_DIR])

    log("Installing Python dependencies...")
    run([sys.executable, "-m", "pip", "install", "-q", *DEPS])

    log("Verifying JAX accelerator availability...")
    run([sys.executable, "-c",
         "import jax; print('Devices:', jax.devices()); "
         "print('Platform:', jax.default_backend())"])

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    log("Set XLA_PYTHON_CLIENT_PREALLOCATE=false, ALLOCATOR=platform")

    if os.path.exists(TEST_DIR):
        os.chdir(TEST_DIR)
        # Make the cloned repo importable (``import rbpf`` needs the repo root,
        # which is REPO_DIR on Colab, not the notebook's /content).
        for p in (REPO_DIR, TEST_DIR):
            if p not in sys.path:
                sys.path.insert(0, p)
    else:
        os.chdir(_HERE)
    log(f"Working directory: {os.getcwd()}")

    # Remove any stale uploaded config at /content so the committed repo config
    # (which is authoritative) is always used.
    stale = os.path.join(_HERE, "smoothing_gpu_config.json")
    if os.path.exists(stale) and os.path.abspath(stale) != os.path.abspath(
        os.path.join(TEST_DIR, "smoothing_gpu_config.json")
    ):
        try:
            os.remove(stale)
            log(f"Removed stale uploaded config: {stale}")
        except OSError as e:
            log(f"WARNING: could not remove stale config {stale}: {e}")

    # Reload the config now that the repo (and its config JSON) is on disk.
    CONFIG = _load_config()
    OUTPUT_DIR = os.path.join(_repo_root(), CONFIG["output_dir"])
    log(f"Configuration reloaded from {CONFIG_PATH}")

    # Let rbpf_rw_v2/src modules (which default to cpu) use the requested
    # accelerator. GPU -> cuda, TPU -> tpu. Only set if the caller didn't
    # already choose one. This MUST happen after the config reload above so the
    # committed repo config (hardware: tpu/gpu) is authoritative, not the stale
    # pre-clone config.
    if "RBSQMC_PLATFORM" not in os.environ:
        if CONFIG["hardware"] == "tpu":
            os.environ["RBSQMC_PLATFORM"] = "tpu"
        else:
            os.environ["RBSQMC_PLATFORM"] = "cuda"
    log(f"RBSQMC_PLATFORM={os.environ.get('RBSQMC_PLATFORM')}")


# ---------------------------------------------------------------------------
# GD core (imports rbpf_rw_v2 lazily so it works on a freshly-cloned Colab VM)
# ---------------------------------------------------------------------------
def run_gd(n_particles: int, start_date: str):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from archive.rbpf_rw_v2.src.data import get_results, ACTIVE_TEAMS, WORLDCUP_2026_TEAMS
    from archive.rbpf_rw_v2.src.helpers import default_init_params, params_to_dict
    from archive.rbpf_rw_v2.src.train import MAX_GOALS, run_gd as run_gd_trainer

    _TEAM_SETS = {
        "ACTIVE_TEAMS": ACTIVE_TEAMS,
        "WORLDCUP_2026_TEAMS": WORLDCUP_2026_TEAMS,
    }
    teams = _TEAM_SETS[CONFIG["teams"]]
    end_date = str(CONFIG["end_date"])

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    print("[GD] Loading data...")
    data, model_inputs, team_id_to_name = get_results(
        start_date=start_date,
        end_date=end_date,
        max_goals=MAX_GOALS,
        teams_only=teams,
    )
    num_teams = len(team_id_to_name)
    print(f"[GD] Loaded {len(data)} matches, {num_teams} teams, "
          f"date=[{start_date}, {end_date}]")
    key = jax.random.PRNGKey(42)
    print("[GD] Initializing parameters...")
    params = default_init_params(num_teams, team_id_to_name=team_id_to_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "gd_params_init.json"), "w") as f:
        json.dump(params_to_dict(params), f, indent=2)
    print(f"[GD] Saved init params to {os.path.join(OUTPUT_DIR, 'gd_params_init.json')}")

    print(f"[GD] Starting direct GD (N={n_particles}, n_steps={CONFIG['n_steps']}, "
          f"date=[{start_date}, {end_date}])")

    final_params, log_marginal_history, diagnostics = run_gd_trainer(
        model_inputs=model_inputs,
        init_params=params,
        num_teams=num_teams,
        n_particles=n_particles,
        n_steps=int(CONFIG["n_steps"]),
        learning_rate=float(CONFIG["learning_rate"]),
        key=key,
    )
    print("[GD] GD run completed.")

    with open(os.path.join(OUTPUT_DIR, "gd_params_final.json"), "w") as f:
        json.dump(params_to_dict(final_params), f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "gd_log_marginal_history.json"), "w") as f:
        json.dump(np.asarray(log_marginal_history).tolist(), f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "gd_loss_history.json"), "w") as f:
        json.dump(np.asarray(diagnostics["loss_history"]).tolist(), f, indent=2)

    try:
        from archive.rbpf_rw_v2.src.graphic import plot_em_convergence
        plot_em_convergence(
            log_marginal_history,
            save_path=os.path.join(OUTPUT_DIR, "gd_convergence.png"),
        )
        print("Saved GD convergence plot to",
              os.path.join(OUTPUT_DIR, "gd_convergence.png"))
    except Exception as e:  # non-fatal: plotting should not kill the run
        print("WARNING: could not save GD convergence plot:", e)

    print("GD completed. Final parameters:")
    print("  gamma_Q:", final_params.gamma_Q.shape)
    print("  alpha:", final_params.alpha)
    print("  beta:", final_params.beta)
    print("  B:", final_params.B)
    print("  gamma_0:", final_params.gamma_0.shape)
    print("  mean_0:", final_params.mean_0.shape)
    print(f"Log marginal history: {np.asarray(log_marginal_history).tolist()}")

    print_summary()


def print_summary():
    """Print a summary of the GD output files (params + history)."""
    log("Fetching results summary...")
    final_path = os.path.join(OUTPUT_DIR, "gd_params_final.json")
    if os.path.exists(final_path):
        with open(final_path, "r") as f:
            params = json.load(f)
        log("Final GD parameters:")
        log(f"  gamma_Q = {params.get('gamma_Q', 'N/A')} [ESTIMATED]")
        log(f"  alpha  = {params.get('alpha', 'N/A')} [ESTIMATED]")
        log(f"  beta   = {params.get('beta', 'N/A')} [ESTIMATED]")
        log(f"  B      = {params.get('B', 'N/A')} [ESTIMATED]")
        log(f"  gamma_0 shape = {len(params.get('gamma_0', []))}x"
            f"{len(params.get('gamma_0', [[]])[0]) if params.get('gamma_0') else 'N/A'} [ESTIMATED]")
        log(f"  mean_0 shape = {len(params.get('mean_0', []))}x"
            f"{len(params.get('mean_0', [[]])[0]) if params.get('mean_0') else 'N/A'} [FIXED]")
    else:
        log(f"Warning: {final_path} not found")

    hist_path = os.path.join(OUTPUT_DIR, "gd_log_marginal_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            history = json.load(f)
        log(f"Log marginal history: {history}")
    else:
        log(f"Warning: {hist_path} not found")

    log("Output files:")
    if os.path.isdir(OUTPUT_DIR):
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, fname)
            size = os.path.getsize(fpath)
            log(f"  {fname} ({size:,} bytes)")
    else:
        log(f"  Output directory {OUTPUT_DIR} does not exist")


def main():
    # Strip the --colab flag before positional-arg parsing.
    args = [a for a in sys.argv[1:] if a != "--colab"]

    log("=" * 60)
    log("SMOOTHING RUNNER — STARTING")
    log("  Reuses rbpf_rw_v2/src/train.py direct GD (-log Z).")
    log("=" * 60)

    if is_colab():
        bootstrap()

    # Read N and start_date AFTER bootstrap so the committed repo config
    # (dates, teams, N) takes effect, not the stale pre-bootstrap CONFIG.
    n_particles = int(args[0]) if args and args[0].isdigit() else int(CONFIG["N"])
    start_date = (
        args[1] if len(args) > 1 else str(CONFIG["start_date"])
    )

    run_gd(n_particles=n_particles, start_date=start_date)

    log("=" * 60)
    log("DONE")
    log("=" * 60)


if __name__ == "__main__":
    main()
