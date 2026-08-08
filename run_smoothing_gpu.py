"""Bootstrap script for running rbpf/smoothing.py on a Google Colab GPU.

Usage (from your Mac):

    colab run --gpu T4 --keep run_smoothing_gpu.py

Then download the outputs:

    colab download -s <session> /content/rbsqmc/rbpf/outputs ./outputs

Then tear down:

    colab stop -s <session>

This script:
  1. Clones the rbsqmc repo
  2. Installs all Python dependencies (including cuthbert from PyPI)
  3. Patches jax_platforms from "cpu" to "gpu" in smoothing.py, model.py, data.py
  4. Runs smoothing.py with unbuffered output so progress streams in real time
  5. Prints a summary of the final parameters
"""

import subprocess
import sys
import os
import json
import time

REPO_URL = "https://github.com/ryantjx/rbsqmc.git"
REPO_DIR = "/content/rbsqmc"
RBPF_DIR = os.path.join(REPO_DIR, "rbpf")
OUTPUT_DIR = os.path.join(RBPF_DIR, "outputs")

DEPS = [
    "jax",
    "jaxlib",
    "numpy",
    "scipy",
    "polars",
    "pandas",
    "matplotlib",
    "seaborn",
    "altair",
    "numba",
    "tqdm",
    "cuthbert",
]

# Files that hardcode CPU — patched to CUDA
PATCH_TARGETS = ["smoothing.py", "model.py", "data.py"]


def log(msg: str):
    """Print with timestamp and flush for real-time streaming."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd, **kwargs):
    """Run a command, streaming output in real time."""
    log(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): "
            f"{' '.join(cmd) if isinstance(cmd, list) else cmd}"
        )
    return result


def run_streaming(cmd):
    """Run a command with stdout/stderr piped to our stdout in real time."""
    log(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): "
            f"{' '.join(cmd) if isinstance(cmd, list) else cmd}"
        )


def patch_cpu_to_cuda(filepath: str):
    """Replace jax_platforms 'cpu' with 'cuda' in a file."""
    with open(filepath, "r") as f:
        src = f.read()

    original = src
    # Handle both single and double quote variants
    for old, new in [
        ('jax.config.update("jax_platforms", "cpu")',
         'jax.config.update("jax_platforms", "cuda")'),
        ("jax.config.update('jax_platforms', 'cpu')",
         "jax.config.update('jax_platforms', 'cuda')"),
    ]:
        src = src.replace(old, new)

    if src != original:
        with open(filepath, "w") as f:
            f.write(src)
        log(f"  Patched {filepath}: cpu → cuda")
    else:
        log(f"  No CPU lock found in {filepath} (already CUDA or not present)")


def main():
    log("=" * 60)
    log("SMOOTHING GPU RUNNER — STARTING")
    log("=" * 60)

    # --- Step 1: Clone the repo ---
    if os.path.exists(REPO_DIR):
        log(f"Repo already exists at {REPO_DIR}, pulling latest...")
        run(["git", "-C", REPO_DIR, "pull", "--rebase"], check=False)
    else:
        log(f"Cloning {REPO_URL} → {REPO_DIR}")
        run(["git", "clone", REPO_URL, REPO_DIR])

    # --- Step 2: Install dependencies ---
    log("Installing Python dependencies...")
    run([sys.executable, "-m", "pip", "install", "-q", *DEPS])

    # --- Step 3: Patch jax_platforms cpu → cuda ---
    # Colab T4 uses CUDA backend (not 'gpu' or 'rocm')
    log("Patching jax_platforms from CPU to CUDA...")
    for fname in PATCH_TARGETS:
        fpath = os.path.join(RBPF_DIR, fname)
        if os.path.exists(fpath):
            patch_cpu_to_cuda(fpath)
        else:
            log(f"  Warning: {fpath} not found, skipping")

    # --- Step 4: Verify JAX sees the GPU ---
    log("Verifying JAX GPU availability...")
    check_cmd = [
        sys.executable, "-c",
        "import jax; print('Devices:', jax.devices()); "
        "print('Platform:', jax.default_backend())"
    ]
    run(check_cmd)

    # --- Step 5: Run smoothing.py with unbuffered output ---
    os.chdir(RBPF_DIR)
    log(f"Working directory: {os.getcwd()}")

    # Create outputs directory (smoothing.py's main() calls save_params before run_em
    # creates the directory with os.makedirs)
    os.makedirs("outputs", exist_ok=True)
    log("Created outputs/ directory")

    log("Running smoothing.py (unbuffered, streaming output)...")
    log("=" * 60)

    # -u = unbuffered, so tqdm and print stream in real time
    run_streaming([sys.executable, "-u", "smoothing.py"])

    log("=" * 60)
    log("smoothing.py COMPLETED")
    log("=" * 60)

    # --- Step 6: Print summary of results ---
    log("Fetching results summary...")

    final_path = os.path.join(OUTPUT_DIR, "em_params_final.json")
    if os.path.exists(final_path):
        with open(final_path, "r") as f:
            params = json.load(f)
        log("Final EM parameters:")
        log(f"  κ (kappa)     = {params.get('init_kappa', 'N/A')}")
        log(f"  α (alpha)     = {params.get('init_alpha', 'N/A')}")
        log(f"  β (beta)      = {params.get('init_beta', 'N/A')}")
        log(f"  friendly_scale = {params.get('init_friendly_scale', 'N/A')}")
        log(f"  B shape       = {len(params.get('init_B', []))}x{len(params.get('init_B', [[]])[0]) if params.get('init_B') else 'N/A'}")
        log(f"  Γ_0 shape     = {len(params.get('init_gamma', []))}x{len(params.get('init_gamma', [[]])[0]) if params.get('init_gamma') else 'N/A'}")
        log(f"  μ_0 shape     = {len(params.get('init_mean', []))}x{len(params.get('init_mean', [[]])[0]) if params.get('init_mean') else 'N/A'}")
    else:
        log(f"Warning: {final_path} not found")

    # List all output files
    log("Output files:")
    if os.path.isdir(OUTPUT_DIR):
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, fname)
            size = os.path.getsize(fpath)
            log(f"  {fname} ({size:,} bytes)")
    else:
        log(f"  Output directory {OUTPUT_DIR} does not exist")

    log("=" * 60)
    log("DONE — use 'colab download' to retrieve outputs, then 'colab stop'")
    log("=" * 60)


if __name__ == "__main__":
    main()