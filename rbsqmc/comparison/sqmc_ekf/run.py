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
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import numpy as np

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
        "execution_backend": jax.default_backend(),
        "jax": jax.__version__,
        "jaxlib": jaxlib_version,
        "versions": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    }


def run(cfg, data_path, smoke=False, output_dir=None, methods=("ekf", "sqmc")):
    """Execute the comparison for the requested methods and return a results dict.

    When ``output_dir`` is given it is used as the exact run directory (no
    timestamp subdir is appended), so the Colab worker can point it at the VM
    run root and the ``results/``/``images/`` subfolders land directly there.

    ``methods`` selects which methods to train/predict/evaluate. A single-method
    run persists that method's artifacts (predictions, metrics, history, images)
    but skips the combined report/plots/CSVs; use :func:`combine` to merge an
    EKF run and an SQMC run into a complete comparison.
    """
    # A stored ``smoke`` key in the config is authoritative so that a launcher
    # writing the effective config to disk reproduces the same subset as the
    # explicit --smoke CLI flag.
    smoke = smoke or bool(cfg.get("smoke"))
    requested = os.environ.get("RBSQMC_PLATFORM")
    if requested in {"cpu", "cuda"}:
        expected = "gpu" if requested == "cuda" else "cpu"
        if jax.default_backend() != expected:
            raise RuntimeError(f"Requested {expected} execution, got {jax.default_backend()}")
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

    methods_obj = train.Methods(dataset, cfg)
    root = jax.random.PRNGKey(cfg["seed"])

    results = {"cfg": cfg}
    for name in methods:
        results[name] = _run_method(name, methods_obj, dataset, cfg, root, results_dir, images_dir)
        # Persist per-method artifacts so a partial run can be combined later.
        _save_json(os.path.join(results_dir, f"{name}_predictions.json"), results[name]["records"])
        _save_json(os.path.join(results_dir, f"{name}_metrics.json"), _metrics_flat(results[name]["metrics"]))
        _save_json(os.path.join(results_dir, f"{name}_history.json"), results[name]["history"])
        results[name]["summary"]["prediction_sec"] = results[name]["prediction_sec"]
        _save_json(os.path.join(results_dir, name, "summary.json"), results[name]["summary"])
        plots_mod.plot_prediction(results[name]["records"], cfg["max_goals"],
                                  os.path.join(images_dir, f"{name}_predictions"))

    if set(methods) == {"ekf", "sqmc"}:
        _write_combined(results_dir, images_dir, dataset, results, cfg, run_dir)
    else:
        from rbsqmc.comparison.sqmc_ekf.scripts.sqmc_ekf_protocol import validate_run
        validate_run(run_dir, cfg, methods=methods)
        print(f"Partial comparison ({', '.join(methods)}) complete: {run_dir}", flush=True)
    return results, run_dir


def _write_combined(results_dir, images_dir, dataset, results, cfg, run_dir):
    """Write the combined report, overlay plot, CSVs, metadata and validate."""
    plots_mod.plot_convergence(
        results["ekf"]["history"], results["sqmc"]["history"], images_dir
    )
    report_mod.write_report(results_dir, dataset, results)
    _save_json(os.path.join(results_dir, "summary.json"), {
        "ekf": results["ekf"]["summary"], "sqmc": results["sqmc"]["summary"]})
    write_performance_metrics_csv(results_dir, results)
    write_logz_history_csv(results_dir, results)
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
    from rbsqmc.comparison.sqmc_ekf.scripts.validate_sqmc_ekf_outputs import validate_artifacts
    validate_artifacts(run_dir, cfg)
    print(f"Comparison complete: {run_dir}", flush=True)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _unflatten_metrics(flat):
    """Rebuild a Metrics object from its flattened ``{"all": ..., "worldcup": ...}`` dict."""
    return eval_mod.Metrics(all=flat["all"], worldcup=flat["worldcup"])


# Execution-specific fields that legitimately differ between a local EKF run
# and a GPU SQMC run; every other field must match for a fair comparison.
_EXECUTION_FIELDS = frozenset({
    "gpu", "gpu_type", "colab_timeout", "setup_timeout", "transfer_timeout",
    "session", "session_name", "run_id", "resolved_utc", "source_bundle_sha256",
})


def _check_partial(name, src, cfg, dataset):
    """Verify a partial run's config and dataset match the combine request."""
    src_results = Path(src) / "results"
    partial_cfg = _read_json(src_results / "comparison_config.json")
    differing = ((set(partial_cfg) ^ set(cfg)) - _EXECUTION_FIELDS) | {
        k for k in set(partial_cfg) & set(cfg)
        if partial_cfg[k] != cfg[k] and k not in _EXECUTION_FIELDS
    }
    if differing:
        raise ValueError(f"{name} run config differs in scientific fields: {sorted(differing)}")
    metadata = _read_json(src_results / "dataset_metadata.json")
    for field in ("source_sha256", "train_count", "test_count", "prediction_count", "worldcup_count"):
        if metadata.get(field) != dataset.metadata.get(field):
            raise ValueError(f"{name} run used a different dataset ({field} mismatch)")
    return metadata


def combine(ekf_dir, sqmc_dir, output_dir, data_path, cfg, smoke=False):
    """Merge a local EKF run and a GPU SQMC run into a complete comparison.

    Reads each partial run's persisted per-method artifacts, copies both sets
    of images, and writes the combined report/plots/CSVs/metadata, then
    validates the merged run. Both partial runs must share the same scientific
    configuration and the same frozen dataset.
    """
    dataset = data_mod_load_dataset(data_path, cfg, smoke=smoke)
    results = {"cfg": cfg}
    metadata = None
    for name, src in (("ekf", ekf_dir), ("sqmc", sqmc_dir)):
        partial_metadata = _check_partial(name, src, cfg, dataset)
        if metadata is not None and partial_metadata != metadata:
            raise ValueError("The two partial runs used different datasets")
        metadata = partial_metadata
        src = Path(src)
        src_results = src / "results"
        summary = _read_json(src_results / name / "summary.json")
        results[name] = {
            "history": _read_json(src_results / f"{name}_history.json"),
            "summary": summary,
            "records": _read_json(src_results / f"{name}_predictions.json"),
            "metrics": _unflatten_metrics(_read_json(src_results / f"{name}_metrics.json")),
            "prediction_sec": summary["prediction_sec"],
        }
    run_dir = Path(output_dir)
    results_dir = run_dir / "results"
    images_dir = run_dir / "images"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    # Copy both partial runs' images (the overlay plot is regenerated below).
    for src in (Path(ekf_dir), Path(sqmc_dir)):
        for path in (src / "images").rglob("*"):
            if path.is_file():
                dest = images_dir / path.relative_to(src / "images")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
    # The validator reads each method's summary from results/<method>/; each
    # partial run only contains its own method's summary.
    for name, src in (("ekf", ekf_dir), ("sqmc", sqmc_dir)):
        dest = results_dir / name / "summary.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(src) / "results" / name / "summary.json", dest)
    _save_json(results_dir / "comparison_config.json", cfg)
    _save_json(results_dir / "run_config.json", compute_setup(cfg))
    # Retain the actual training machines; the combined run itself runs on CPU.
    for name, src in (("ekf", ekf_dir), ("sqmc", sqmc_dir)):
        shutil.copyfile(Path(src) / "results" / "run_config.json",
                        results_dir / name / "run_config.json")
    _save_json(results_dir / "dataset_metadata.json", dataset.metadata)
    # Re-persist the per-method artifacts at the combined run's top level.
    for name in ("ekf", "sqmc"):
        _save_json(results_dir / f"{name}_predictions.json", results[name]["records"])
        _save_json(results_dir / f"{name}_metrics.json", _metrics_flat(results[name]["metrics"]))
    _write_combined(results_dir, images_dir, dataset, results, cfg, run_dir)
    return results, str(run_dir)


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
    # synchronized_moments returns post-match states only. Restore the prior
    # so both methods' index i means "before match i", including the WC split.
    mean = np.concatenate([np.asarray(history["mean"][:1]), np.asarray(mean)])
    cov = np.concatenate([np.asarray(history["cov"][:1]), np.asarray(cov)])
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
    parser.add_argument("--methods", default="both", choices=["both", "ekf", "sqmc"],
                        help="Which methods to run (default: both).")
    parser.add_argument("--combine", nargs=2, metavar=("EKF_DIR", "SQMC_DIR"),
                        help="Merge a local EKF run and a GPU SQMC run into a complete comparison.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.smoke:
        cfg["smoke"] = True
        cfg["n_epochs"] = min(cfg.get("n_epochs", 100), 3)
        cfg["n_reps"] = min(cfg.get("n_reps", 25), 2)
        cfg["n_particles"] = min(cfg.get("n_particles", 512), 64)

    if args.combine:
        if args.methods != "both" or args.smoke:
            parser.error("--combine cannot be combined with --methods or --smoke")
        ekf_dir, sqmc_dir = args.combine
        if not args.output_dir:
            parser.error("--combine requires --output-dir")
        # A stored ``smoke`` flag in the config is authoritative: partial runs
        # trained on the smoke subset whenever their config recorded it, so the
        # combine step must load the same subset regardless of this CLI flag.
        combine(ekf_dir, sqmc_dir, args.output_dir, args.data, cfg,
                smoke=bool(cfg.get("smoke")))
        return

    methods = ("ekf", "sqmc") if args.methods == "both" else (args.methods,)
    run(cfg, args.data, smoke=args.smoke, output_dir=args.output_dir, methods=methods)


if __name__ == "__main__":
    main()
