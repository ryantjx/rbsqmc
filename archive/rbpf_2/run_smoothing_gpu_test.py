"""Bootstrap script for running rbpf_2/smoothing.py on a Google Colab GPU — TEST MODE.

Uses lightweight test parameters for a quick smoke test:
  - N = 10 particles (instead of 1000)
  - start_date = "2020-01-01" (instead of "2000-01-01")
  - n_epochs = 2 (instead of 10)

Usage (from your Mac):

    colab run --gpu T4 --keep run_smoothing_gpu_test.py

Then download the outputs:

    colab download -s <session> /content/rbsqmc/rbpf_2/outputs ./outputs_test

Then tear down:

    colab stop -s <session>
"""

import subprocess
import sys
import os
import json
import time

REPO_URL = "https://github.com/ryantjx/rbsqmc.git"
REPO_DIR = "/content/rbsqmc"
RBPF_DIR = os.path.join(REPO_DIR, "rbpf_2")
OUTPUT_DIR = os.path.join(RBPF_DIR, "outputs_gpu")

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

# Test parameters
TEST_N = 10
TEST_START_DATE = "2020-01-01"
TEST_N_EPOCHS = 2


def log(msg: str):
    """Print with timestamp and flush for real-time streaming."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd, **kwargs):
    """Run a command, streaming stdout/stderr in real time."""
    log(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    # Don't capture — let output stream directly to the Colab cell output
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


def patch_file(filepath: str, replacements: list[tuple[str, str]]):
    """Apply a list of (old, new) string replacements to a file."""
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
    log("SMOOTHING GPU TEST RUNNER — STARTING")
    log(f"  N = {TEST_N}, start_date = {TEST_START_DATE}, n_epochs = {TEST_N_EPOCHS}")
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

    # --- Step 3: Patch jax_platforms cpu → cuda in all three files ---
    # Colab T4 uses CUDA backend (not 'gpu' or 'rocm')
    log("Patching jax_platforms from CPU to CUDA...")
    for fname in PATCH_TARGETS:
        fpath = os.path.join(RBPF_DIR, fname)
        if os.path.exists(fpath):
            patch_file(fpath, [
                ('jax.config.update("jax_platforms", "cpu")',
                 'jax.config.update("jax_platforms", "cuda")'),
                ("jax.config.update('jax_platforms', 'cpu')",
                 "jax.config.update('jax_platforms', 'cuda')"),
            ])
        else:
            log(f"  Warning: {fpath} not found, skipping")

    # --- Step 4: Patch smoothing.py with test parameters ---
    log("Patching smoothing.py with test parameters...")
    smoothing_path = os.path.join(RBPF_DIR, "smoothing.py")
    with open(smoothing_path, "r") as f:
        src = f.read()

    # Patch N (handle any value: committed N=100, local N=1000, etc.)
    import re
    src = re.sub(r'^N = \d+\n', f'N = {TEST_N}\n', src, count=1, flags=re.MULTILINE)

    # Patch start_date in both download_results and read_results calls
    src = src.replace('start_date="2000-01-01"', f'start_date="{TEST_START_DATE}"')
    src = src.replace("start_date='2000-01-01'", f"start_date='{TEST_START_DATE}'")

    # Patch n_epochs (handle any value)
    src = re.sub(r'n_epochs=\d+, output_dir', f'n_epochs={TEST_N_EPOCHS}, output_dir', src)

    # Patch output_dir to outputs_gpu (label GPU outputs separately)
    src = src.replace('output_dir="./outputs"', 'output_dir="./outputs_gpu"')
    src = src.replace('"./outputs/em_params_init.json"', '"./outputs_gpu/em_params_init.json"')

    with open(smoothing_path, "w") as f:
        f.write(src)
    log(f"  Patched {smoothing_path}")

    # Verify the patches
    with open(smoothing_path, "r") as f:
        verify = f.read()
    log(f"  N = {TEST_N}: {'OK' if f'N = {TEST_N}' in verify else 'FAILED'}")
    log(f"  start_date = {TEST_START_DATE}: {'OK' if TEST_START_DATE in verify else 'FAILED'}")
    log(f"  n_epochs = {TEST_N_EPOCHS}: {'OK' if f'n_epochs={TEST_N_EPOCHS}' in verify else 'FAILED'}")
    log(f"  output_dir = outputs_gpu: {'OK' if 'outputs_gpu' in verify else 'FAILED'}")

    # --- Step 5: Verify JAX sees the GPU ---
    log("Verifying JAX GPU availability...")
    check_cmd = [
        sys.executable, "-c",
        "import jax; print('Devices:', jax.devices()); "
        "print('Platform:', jax.default_backend())"
    ]
    run(check_cmd)

    # --- Step 6: Run smoothing.py with unbuffered output ---
    os.chdir(RBPF_DIR)
    log(f"Working directory: {os.getcwd()}")

    # Create outputs_gpu directory (smoothing.py's main() calls save_params before run_em
    # creates the directory with os.makedirs)
    os.makedirs("outputs_gpu", exist_ok=True)
    log("Created outputs_gpu/ directory")

    log("Running smoothing.py (unbuffered, streaming output)...")
    log("=" * 60)

    # -u = unbuffered, so tqdm and print stream in real time
    run_streaming([sys.executable, "-u", "smoothing.py"])

    log("=" * 60)
    log("smoothing.py COMPLETED")
    log("=" * 60)

    # --- Step 7: Print summary of results ---
    log("Fetching results summary...")

    final_path = os.path.join(OUTPUT_DIR, "em_params_final.json")
    if os.path.exists(final_path):
        with open(final_path, "r") as f:
            params = json.load(f)
        log("Final EM parameters:")
        log(f"  κ (kappa)      = {params.get('init_kappa', 'N/A')}")
        log(f"  α (alpha)      = {params.get('init_alpha', 'N/A')}")
        log(f"  β (beta)       = {params.get('init_beta', 'N/A')}")
        log(f"  friendly_scale = {params.get('init_friendly_scale', 'N/A')}")
        log(f"  B shape        = {len(params.get('init_B', []))}x{len(params.get('init_B', [[]])[0]) if params.get('init_B') else 'N/A'}")
        log(f"  Γ_0 shape      = {len(params.get('init_gamma', []))}x{len(params.get('init_gamma', [[]])[0]) if params.get('init_gamma') else 'N/A'}")
        log(f"  μ_0 shape      = {len(params.get('init_mean', []))}x{len(params.get('init_mean', [[]])[0]) if params.get('init_mean') else 'N/A'}")
    else:
        log(f"Warning: {final_path} not found")

    # Print per-epoch parameters
    log("Per-epoch parameters:")
    for epoch in range(TEST_N_EPOCHS):
        epoch_path = os.path.join(OUTPUT_DIR, f"em_params_epoch_{epoch}.json")
        if os.path.exists(epoch_path):
            with open(epoch_path, "r") as f:
                ep = json.load(f)
            log(f"  Epoch {epoch}: κ={ep.get('init_kappa', '?')}, "
                f"α={ep.get('init_alpha', '?')}, β={ep.get('init_beta', '?')}")
        else:
            log(f"  Epoch {epoch}: file not found")

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
    log("TEST DONE — use 'colab download' to retrieve outputs, then 'colab stop'")
    log("=" * 60)


if __name__ == "__main__":
    main()