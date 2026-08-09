"""Bootstrap script for running rbpf/smoothing_v2.py on a Google Colab GPU.

V2: Fixed κ=1.0, B=I₂. Estimates Γ₀ (with IW prior), α, β, μ₀.

Usage:
    colab run --gpu T4 --keep run_smoothing_gpu_v2.py [GPU_N]
"""

import subprocess
import sys
import os
import json
import time
import re

REPO_URL = "https://github.com/ryantjx/rbsqmc.git"
REPO_DIR = "/content/rbsqmc"
RBPF_DIR = os.path.join(REPO_DIR, "rbpf")
OUTPUT_DIR = os.path.join(RBPF_DIR, "outputs_gpu_v2")

DEPS = [
    "jax", "jaxlib", "numpy", "scipy", "polars", "pandas",
    "matplotlib", "seaborn", "altair", "numba", "tqdm", "cuthbert",
]

PATCH_TARGETS = ["smoothing_v2.py", "model.py", "data.py"]


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def patch_cpu_to_cuda(filepath: str):
    with open(filepath, "r") as f:
        src = f.read()
    original = src
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
        log(f"  No CPU lock found in {filepath}")


def patch_file(filepath: str, replacements: list[tuple[str, str]]):
    with open(filepath, "r") as f:
        src = f.read()
    original = src
    for old, new in replacements:
        src = src.replace(old, new)
    if src != original:
        with open(filepath, "w") as f:
            f.write(src)
        log(f"  Patched {filepath}")
    else:
        log(f"  No changes needed in {filepath}")


def main():
    log("=" * 60)
    log("SMOOTHING V2 GPU RUNNER — STARTING")
    log("  Fixed κ=1.0, B=I₂. Estimates Γ₀, α, β, μ₀.")
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
    log("Patching jax_platforms from CPU to CUDA...")
    for fname in PATCH_TARGETS:
        fpath = os.path.join(RBPF_DIR, fname)
        if os.path.exists(fpath):
            patch_cpu_to_cuda(fpath)
        else:
            log(f"  Warning: {fpath} not found, skipping")

    # --- Step 4: Patch N if specified ---
    smoothing_path = os.path.join(RBPF_DIR, "smoothing_v2.py")
    if len(sys.argv) > 1:
        GPU_N = sys.argv[1]
        log(f"Patching N to {GPU_N} (overridden via sys.argv)...")
        with open(smoothing_path, "r") as f:
            src = f.read()
        src = re.sub(r'^N = \d+\n', f'N = {GPU_N}\n', src, count=1, flags=re.MULTILINE)
        with open(smoothing_path, "w") as f:
            f.write(src)
        log(f"  Patched N → {GPU_N}")
    else:
        log("Using N from smoothing_v2.py (no override)")

    # --- Step 5: Verify JAX sees the GPU ---
    log("Verifying JAX GPU availability...")
    run([sys.executable, "-c",
         "import jax; print('Devices:', jax.devices()); "
         "print('Platform:', jax.default_backend())"])

    # --- Step 6: Run smoothing_v2.py ---
    os.chdir(RBPF_DIR)
    log(f"Working directory: {os.getcwd()}")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    log("Set XLA_PYTHON_CLIENT_PREALLOCATE=false, ALLOCATOR=platform")

    os.makedirs("outputs_gpu_v2", exist_ok=True)
    log("Created outputs_gpu_v2/ directory")

    log("Running smoothing_v2.py (unbuffered, streaming output)...")
    log("=" * 60)
    run_streaming([sys.executable, "-u", "smoothing_v2.py"])

    log("=" * 60)
    log("smoothing_v2.py COMPLETED")
    log("=" * 60)

    # --- Step 7: Print summary ---
    log("Fetching results summary...")
    final_path = os.path.join(OUTPUT_DIR, "em_params_final.json")
    if os.path.exists(final_path):
        with open(final_path, "r") as f:
            params = json.load(f)
        log("Final EM parameters (V2):")
        log(f"  κ (kappa)     = {params.get('init_kappa', 'N/A')} [FIXED]")
        log(f"  α (alpha)     = {params.get('init_alpha', 'N/A')}")
        log(f"  β (beta)      = {params.get('init_beta', 'N/A')}")
        log(f"  friendly_scale = {params.get('init_friendly_scale', 'N/A')}")
        log(f"  B = {params.get('init_B', 'N/A')} [FIXED]")
        log(f"  Γ_0 shape     = {len(params.get('init_gamma', []))}x{len(params.get('init_gamma', [[]])[0]) if params.get('init_gamma') else 'N/A'}")
        log(f"  μ_0 shape     = {len(params.get('init_mean', []))}x{len(params.get('init_mean', [[]])[0]) if params.get('init_mean') else 'N/A'}")
    else:
        log(f"Warning: {final_path} not found")

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