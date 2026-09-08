"""CLI entrypoint that wires the complete SQMC–EKF comparison pipeline.

Runs data loading, training, prediction, evaluation, plots, and the report
into a single timestamped run directory under ``outputs/DDMMYYYY_HHMM/``.

Usage:
    python -m rbsqmc.comparison.sqmc_ekf.run \
        --config rbsqmc/comparison/sqmc_ekf/config/config.json \
        --data rbsqmc/data/results.csv \
        [--smoke] [--output-dir ...]
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import jax

from rbsqmc.src.data.data_ekf import load_dataset as data_mod_load_dataset
from rbsqmc.comparison.sqmc_ekf.scripts import evaluate as eval_mod
from rbsqmc.comparison.sqmc_ekf.scripts import plots as plots_mod
from rbsqmc.src.model.ekf import predict as predict_mod
from rbsqmc.comparison.sqmc_ekf.scripts import report as report_mod
from rbsqmc.src.model.ekf import train


DEFAULT_DATA = "rbsqmc/data/results.csv"


def _output_dir(base):
    stamp = datetime.now().strftime("%d%m%Y_%H%M")
    out = os.path.join(base, stamp)
    os.makedirs(out, exist_ok=True)
    return out


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, default=lambda o: o.tolist())


def _metrics_flat(metrics):
    """Serialize a Metrics object (all + worldcup) to a dict."""
    return {"all": metrics.all, "worldcup": metrics.worldcup}


def _config_digest(config):
    """SHA-256 of the canonicalized config, matching sqmc/comparison's protocol."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit():
    """Return the current HEAD revision, or None if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def compute_setup(config):
    """Collect host/device/software setup info, mirroring the reference run_config."""
    devices = {}
    for backend in ("cpu", "gpu"):
        try:
            devices[backend] = [
                {"kind": d.device_kind, "id": d.id, "platform": d.platform}
                for d in jax.devices(backend)
            ]
        except Exception:
            devices[backend] = []
    try:
        import jaxlib
        jaxlib_version = jaxlib.__version__
    except Exception:
        jaxlib_version = None

    gpu = None
    nvidia_smi = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        nvidia_smi = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception:
        pass

    cpu_models = []
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        cpu_models = sorted({
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        })
    except Exception:
        cpu_models = [platform.processor()] if platform.processor() else []

    host_memory = None
    try:
        host_memory = Path("/proc/meminfo").read_text()
    except Exception:
        pass

    return {
        "source_commit": _git_commit(),
        "run_id": datetime.now().strftime("%d%m%Y_%H%M"),
        "config_sha256": _config_digest(config),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_models": cpu_models,
        "host_memory": host_memory,
        "gpu_name_memory_MiB_driver": gpu,
        "nvidia_smi": nvidia_smi,
        "devices": devices,
        "jax": jax.__version__,
        "jaxlib": jaxlib_version,
        "versions": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    }


def run(cfg, data_path, smoke=False, output_dir=None):
    """Execute the full comparison and return a results dict.

    When ``output_dir`` is given it is used as the exact run directory (no
    timestamp subdir is appended), so the Colab worker can point it at the VM
    run root and the ``results/``/``images/`` subfolders land directly there.
    """
    dataset = data_mod_load_dataset(data_path, cfg, smoke=smoke)
    if output_dir:
        run_dir = output_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        base_out = os.path.join(os.path.dirname(__file__), "outputs")
        run_dir = _output_dir(base_out)
    # Separate machine-readable results (JSON/CSV/MD) from images (PNG).
    results_dir = os.path.join(run_dir, "results")
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Persist the exact config + dataset metadata for reproducibility.
    _save_json(os.path.join(results_dir, "comparison_config.json"), cfg)
    _save_json(os.path.join(results_dir, "run_config.json"), compute_setup(cfg))
    _save_json(os.path.join(results_dir, "dataset_metadata.json"), dataset.metadata)

    methods = train.Methods(dataset, cfg)
    root = jax.random.PRNGKey(cfg["seed"])

    results = {"cfg": cfg}
    results["ekf"] = _run_method("ekf", methods, dataset, cfg, root, results_dir, images_dir)
    results["sqmc"] = _run_method("sqmc", methods, dataset, cfg, root, results_dir, images_dir)

    # Comparison plots + report.
    plots_mod.plot_convergence(
        results["ekf"]["history"], results["sqmc"]["history"], images_dir
    )
    plots_mod.plot_prediction(
        results["ekf"]["records"], cfg["max_goals"],
        os.path.join(images_dir, "ekf_predictions")
    )
    plots_mod.plot_prediction(
        results["sqmc"]["records"], cfg["max_goals"],
        os.path.join(images_dir, "sqmc_predictions")
    )
    report_mod.write_report(results_dir, dataset, results)

    # Store per-method record/metric artifacts.
    for name in ("ekf", "sqmc"):
        _save_json(os.path.join(results_dir, f"{name}_predictions.json"),
                   results[name]["records"])
        _save_json(os.path.join(results_dir, f"{name}_metrics.json"),
                   _metrics_flat(results[name]["metrics"]))
    _save_json(os.path.join(results_dir, "summary.json"), {
        "ekf": results["ekf"]["summary"], "sqmc": results["sqmc"]["summary"]})

    # Performance comparison tables.
    write_performance_metrics_csv(results_dir, results)
    write_logz_history_csv(results_dir, results)

    # Reproducibility metadata: when the run finished and what it produced.
    _save_json(os.path.join(results_dir, "run_metadata.json"), {
        "completed_at_utc": datetime.utcnow().isoformat() + "Z",
        "run_id": os.path.basename(run_dir),
        "config_sha256": _config_digest(cfg),
        "source_commit": _git_commit(),
        "prediction_count": dataset.metadata["prediction_count"],
        "worldcup_count": dataset.metadata["worldcup_count"],
        "methods": list(results),
        "results_dir": "results",
        "images_dir": "images",
        "artifacts": [
            "results/REPORT.md", "results/DRAFT.md", "results/summary.json",
            "results/comparison_config.json", "results/run_config.json",
            "results/dataset_metadata.json",
            "results/performance_metrics.csv", "results/logz_history.csv",
            "images/logz_overlay_train_test.png",
            "results/ekf_predictions.json", "results/sqmc_predictions.json",
            "results/ekf_metrics.json", "results/sqmc_metrics.json",
            "images/ekf_top5_strengths.png", "images/sqmc_top5_strengths.png",
            "images/ekf_pre_worldcup_rankings.png", "images/sqmc_pre_worldcup_rankings.png",
            "images/ekf_post_worldcup_rankings.png", "images/sqmc_post_worldcup_rankings.png",
        ],
    })

    print(f"Comparison complete: {run_dir}", flush=True)
    return results, run_dir


def write_performance_metrics_csv(run_dir, results):
    """Write one row per method with compute time and prediction metrics."""
    rows = []
    for method in ("ekf", "sqmc"):
        r = results[method]
        m = r["metrics"].worldcup or r["metrics"].all
        s = r["summary"]
        rows.append({
            "method": method,
            "compile_sec": s["compilation_sec"],
            "train_execution_sec": s["execution_sec"],
            "prediction_sec": r["prediction_sec"],
            "pipeline_sec": s["compilation_sec"] + s["execution_sec"] + r["prediction_sec"],
            "final_train_logz": s["final_train_logz"],
            "final_test_logz": s["final_test_logz"],
            "best_test_logz": s["best_test_logz"],
            "mean_brier_score": m.get("mean_brier_score"),
            "brier_skill_score_vs_uniform": m.get("brier_skill_score_vs_uniform"),
            "outcome_accuracy": m.get("outcome_accuracy"),
            "exact_score_accuracy": m.get("exact_score_accuracy"),
            "mean_log_likelihood": m.get("mean_log_likelihood"),
            "n_scored": m.get("n_scored"),
        })
    path = os.path.join(run_dir, "performance_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_logz_history_csv(run_dir, results):
    """Write per-epoch train/test logZ for both methods (long format)."""
    rows = []
    for method in ("ekf", "sqmc"):
        for h in results[method]["history"]:
            rows.append({"method": method, "epoch": h["epoch"],
                         "train_logz": h["train_logz"], "test_logz": h["test_logz"]})
    path = os.path.join(run_dir, "logz_history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "epoch", "train_logz", "test_logz"])
        writer.writeheader()
        writer.writerows(rows)


def _run_method(method, methods, dataset, cfg, root, results_dir, images_dir):
    """Train a method, then predict and evaluate it on the held-out split."""
    raw, history, summary = methods.train(method, os.path.join(results_dir, method))
    params = methods.params(method, raw)

    # Time the prediction + evaluation phase separately from training.
    pred_start = time.perf_counter()
    if method == "sqmc":
        key = jax.random.fold_in(root, 2_000_000)
        pred = predict_mod.predict_sqmc(dataset, params, cfg, key)
        states, augmented = _sqmc_states(dataset, params, cfg, root)
        plots_mod.plot_correlation(
            params["model"], augmented, dataset.teams, images_dir
        )
    else:
        pred = predict_mod.predict_ekf(dataset, params, cfg)
        states, augmented = _ekf_states(dataset, params, len(dataset.teams))

    records = eval_mod.build_records(dataset, pred.grids, pred.logp)
    metrics = eval_mod.compute_metrics(records)
    prediction_sec = time.perf_counter() - pred_start

    plots_mod.plot_ranking_trajectory(
        states, dataset.teams, dataset.sqmc.timestamp, images_dir, method,
        pre_index=dataset.train_count + dataset.test_count,
        post_index=-1,
    )

    return {"raw": raw, "params": params, "history": history, "summary": summary,
            "pred": pred, "records": records, "metrics": metrics,
            "states": states, "prediction_sec": prediction_sec}


def _sqmc_states(dataset, params, cfg, root):
    """Weighted SQMC posterior moments plus the augmented gamma trajectory."""
    from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc as run_filter

    key = jax.random.fold_in(root, 3_000_000)
    result, augmented = run_filter(
        key, dataset.sqmc, params["model"], cfg["n_particles"], cfg["max_goals"]
    )
    return plots_mod.wrap_sqmc(result, len(dataset.teams)), augmented


def _ekf_states(dataset, params, num_teams):
    """Native EKF moments wrapped as a single-particle FilterStates."""
    from rbsqmc.src.model.ekf import model as ekf

    history = ekf.run_filter(dataset.inputs, params, num_teams)
    mean, cov = ekf.synchronized_moments(
        dataset.inputs, history, params, num_teams
    )
    return plots_mod.MeanFilterStates(mean, cov), None


def main():
    parser = argparse.ArgumentParser(description="Run SQMC vs EKF comparison")
    parser.add_argument("--config", required=True, type=str,
                        help="Path to the comparison config JSON.")
    parser.add_argument("--data", default=DEFAULT_DATA, type=str,
                        help="Path to the results CSV/parquet.")
    parser.add_argument("--smoke", action="store_true",
                        help="Use a smoke subset for a fast run.")
    parser.add_argument("--output-dir", default=None, type=str,
                        help="Optional explicit output directory.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.smoke:
        cfg["smoke"] = True
        cfg["n_epochs"] = min(cfg.get("n_epochs", 100), 3)
        cfg["n_reps"] = min(cfg.get("n_reps", 25), 2)
        cfg["n_particles"] = min(cfg.get("n_particles", 512), 64)

    run(cfg, args.data, smoke=args.smoke, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
