# Running `rbpf/model.py` on Google Colab via `google-colab-cli`

This repo is **not a single-file script**. `model.py` imports sibling modules
(`bivariate_poisson`, `data`, `graphic`) and a **local** package `cuthbert`
that is not on PyPI (it lives at `github.com/ryantjx/cuthbert`).

Because `colab run` / `colab exec -f` only transmit the *content* of one file
to the remote kernel, sibling imports will fail. You must get a shell on the
VM, clone the repos, install dependencies, and run from the `rbpf/` directory.

## 0. Install the CLI (one-time, on your Mac)

```bash
uv tool install google-colab-cli
# or
pip install google-colab-cli
```

Authenticate on first use — the CLI defaults to ADC; run any `colab` command
and follow the OAuth prompt, or use `--auth oauth2`.

## 1. Provision a session

```bash
colab new -s rbsqmc
# For a GPU (optional — JAX runs on CPU here by default):
# colab new -s rbsqmc --gpu T4
```

## 2. Open a shell on the VM

```bash
colab ssh -s rbsqmc      # SSH shell (recommended)
# or
colab console -s rbsqmc # raw tmux TTY
```

All the following commands run **inside that remote shell**.

## 3. Clone the repos on the VM

```bash
cd /content
git clone https://github.com/ryantjx/rbsqmc.git
git clone https://github.com/ryantjx/cuthbert.git
```

## 4. Install dependencies

```bash
# Python deps from requirements.txt
pip install jax jaxlib numpy scipy polars pandas matplotlib seaborn altair numba pytest
```

### `cuthbert`

`cuthbert` **is on PyPI** (latest `0.0.14`). Just install it:

```bash
pip install cuthbert
```

You can skip cloning `cuthbert` in step 3.

> **Note on the local repo:** your `github.com/ryantjx/cuthbert` checkout is a
> uv *workspace* pinned at `0.0.10`, but that workspace only takes effect when
> you run `uv` **from inside the `cuthbert/` repo**. Your `rbsqmc` venv has no
> link to it — it already installs `cuthbert` straight from PyPI (currently
> `0.0.13`). To upgrade locally:
> ```bash
> cd /Users/ryant/Github/ryantjx/rbsqmc && uv pip install --upgrade cuthbert
> ```
>
> Only clone your fork on the VM if you need unpublished local edits that aren't
> in the PyPI release. In that case:
> ```bash
> pip install -e /content/cuthbert/pkg/cuthbertlib
> pip install -e /content/cuthbert/pkg/cuthbert
> ```

## 5. Run the model

`model.py` uses relative imports (`from bivariate_poisson import loglik`,
`from data import ...`, `from graphic import ...`) and writes outputs to
`./outputs/base/`, so you **must** run it from the `rbpf/` directory:

```bash
cd /content/rbsqmc/rbpf
python model.py
```

Outputs (plots + parquet) are written to `/content/rbsqmc/rbpf/outputs/base/`.

## 6. Retrieve outputs to your Mac

From a **local** terminal:

```bash
colab download -s rbsqmc /content/rbsqmc/rbpf/outputs/base ./outputs/base
```

Or grab individual files:

```bash
colab download -s rbsqmc /content/rbsqmc/rbpf/outputs/base/states.parquet ./states.parquet
```

## 7. Tear down the VM

```bash
colab stop -s rbsqmc
```

---

## One-shot alternative (ephemeral, no manual session)

If you don't want to manage the session lifecycle, use `colab run` — but you
still need the sibling files + `cuthbert` on the VM. The cleanest way is a
small bootstrap script that clones, installs, and runs, then pass *that* to
`colab run`:

```bash
cat > /tmp/run_rbsqmc.py <<'EOF'
import subprocess, sys, os
subprocess.run(["git", "clone", "https://github.com/ryantjx/rbsqmc.git", "/content/rbsqmc"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jax","jaxlib","numpy","scipy","polars","pandas",
                "matplotlib","seaborn","altair","numba","cuthbert"], check=True)
os.chdir("/content/rbsqmc/rbpf")
subprocess.run([sys.executable, "model.py"], check=True)
EOF

colab run --keep /tmp/run_rbsqmc.py
```

> This installs `cuthbert` from PyPI. If you need unpublished local edits
> instead, add the `git clone cuthbert` + `pip install -e` lines from step 4.

Use `--keep` so the VM stays up and you can `colab download` the outputs
afterwards, then `colab stop` when done. Without `--keep` the VM is torn down
immediately on completion and you lose the outputs.

---

## Running `rbpf/smoothing.py` on GPU

`smoothing.py` (and its dependencies `model.py`, `data.py`) all hardcode
`jax.config.update("jax_platforms", "cpu")`. To run on GPU, these must be
patched to `"cuda"` on the VM (Colab T4 uses the CUDA backend, not 'gpu').
The repo includes a bootstrap script (`run_smoothing_gpu.py`) that handles
cloning, installing, patching, and
running with real-time progress output.

### One-shot (recommended)

```bash
colab run --gpu T4 --keep run_smoothing_gpu.py
```

This will:
1. Clone `rbsqmc` to `/content/rbsqmc`
2. Install all deps (including `cuthbert` from PyPI and `tqdm`)
3. Patch `jax_platforms` from `"cpu"` to `"cuda"` in `smoothing.py`, `model.py`, `data.py`
4. Verify JAX sees the GPU
5. Run `smoothing.py` with unbuffered output (tqdm + print stream in real time)
6. Print a summary of the final EM parameters and list all output files

### Retrieve outputs

From a **local** terminal:

```bash
colab download -s rbsqmc /content/rbsqmc/rbpf/outputs ./outputs
```

### Tear down

```bash
colab stop -s rbsqmc
```

### Manual alternative

If you prefer to manage the session yourself:

```bash
colab new -s rbsqmc --gpu T4
colab ssh -s rbsqmc
```

Inside the remote shell:

```bash
cd /content
git clone https://github.com/ryantjx/rbsqmc.git
pip install jax jaxlib numpy scipy polars pandas matplotlib seaborn altair numba tqdm cuthbert

cd /content/rbsqmc/rbpf
# Patch all three files: cpu → cuda
sed -i 's/jax.config.update("jax_platforms", "cpu")/jax.config.update("jax_platforms", "cuda")/' smoothing.py model.py data.py

python -u smoothing.py
```

Then `colab download` and `colab stop` as above.

---

## Notes

- **Network**: `data.py` downloads `results.csv` from
  `raw.githubusercontent.com/martj42/international_results/master/results.csv`,
  so the VM needs internet (Colab VMs do by default).
- **Working directory**: always `cd` into `rbpf/` before running — the
  relative imports and `./outputs/base/` paths depend on it.
- **cuthbert comes from PyPI by default**. Your local `github.com/ryantjx/cuthbert`
  checkout is a uv workspace, but that only applies inside the `cuthbert/` repo
  itself — `rbsqmc` has no link to it and installs `cuthbert` from PyPI. Only clone
  your fork on the VM if you need unpublished local edits.
- **JAX on CPU**: `model.py` and `smoothing.py` force CPU via
  `jax.config.update("jax_platforms", "cpu")`. For GPU runs, patch this to
  `"cuda"` (the `run_smoothing_gpu.py` script does this automatically).
- **smoothing.py imports `tqdm`**: ensure it's installed (`pip install tqdm`).