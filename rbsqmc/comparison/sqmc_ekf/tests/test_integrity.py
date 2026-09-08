"""Regression checks for result integrity, download lifecycle and state indexing."""

import csv
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rbsqmc.comparison.sqmc_ekf import run as comparison
from rbsqmc.comparison.sqmc_ekf.scripts import evaluate, sqmc_ekf_protocol as protocol
from rbsqmc.comparison.sqmc_ekf.scripts.validate_sqmc_ekf_outputs import (
    REQUIRED_METHOD_IMAGES, _latest_run_dir,
)
from rbsqmc.src.model.ekf import model as ekf


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete_run(root):
    """Use the real output writers and evaluator to construct a small valid run."""
    cfg = dict(n_epochs=2, n_reps=2, max_goals=1, prediction_start_date="2026-06-11",
               source_commit="a" * 40, run_id="08092026_1200", session_name="test", transfer_timeout=10)
    results_dir, images = root / "results", root / "images"
    results_dir.mkdir(parents=True)
    images.mkdir()
    write_json(results_dir / "comparison_config.json", cfg)
    provenance = dict(config_sha256=protocol.config_digest(cfg), source_commit=cfg["source_commit"])
    write_json(results_dir / "run_config.json", provenance)
    write_json(results_dir / "run_metadata.json", dict(provenance, prediction_count=2, worldcup_count=2))
    write_json(results_dir / "dataset_metadata.json", dict(train_count=2, test_count=1,
                                                          prediction_count=2, worldcup_count=2))
    dataset = SimpleNamespace(train_count=0, test_count=0, frame=pd.DataFrame([
        dict(date=pd.Timestamp("2026-06-11"), home_team="A", away_team="B", home_score=0,
             away_score=0, tournament="FIFA World Cup"),
        dict(date=pd.Timestamp("2026-06-12"), home_team="B", away_team="A", home_score=1,
             away_score=0, tournament="FIFA World Cup"),
    ]))
    grids = np.array([[[.4, .1], [.3, .2]], [[.1, .2], [.5, .2]]])
    records = evaluate.build_records(dataset, grids, np.log(np.array([.4, .5]) + 1e-12))
    metrics = evaluate.compute_metrics(records)
    histories = [dict(epoch=1, train_logz=-10., test_logz=-4.),
                 dict(epoch=2, train_logz=-9., test_logz=-5.)]
    summaries, results = {}, {}
    for method in ("ekf", "sqmc"):
        summary = dict(n_epochs_completed=2, checkpoint_policy="final_epoch",
                       learning_rate_schedule="cosine", gradient_replicas=2 if method == "sqmc" else 1,
                       final_train_logz=-9., final_test_logz=-5., best_test_logz=-4., best_test_epoch=1,
                       compilation_sec=.1, execution_sec=.2)
        summaries[method] = summary
        write_json(results_dir / method / "summary.json", summary)
        write_json(results_dir / f"{method}_predictions.json", records)
        write_json(results_dir / f"{method}_metrics.json", dict(all=metrics.all, worldcup=metrics.worldcup))
        results[method] = dict(summary=summary, history=histories, metrics=metrics, prediction_sec=.3)
    write_json(results_dir / "summary.json", summaries)
    comparison.write_performance_metrics_csv(results_dir, results)
    comparison.write_logz_history_csv(results_dir, results)
    for name in ("REPORT.md", "DRAFT.md"):
        (results_dir / name).write_text("Two epochs completed; comparison fixture.\n")
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    for name in ["logz_overlay_train_test.png"] + [f"{m}_{s}" for m in ("ekf", "sqmc") for s in REQUIRED_METHOD_IMAGES]:
        (images / name).write_bytes(buffer.getvalue())
    return cfg


def edit_json(root, name, change):
    path = root / "results" / name
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


def test_real_writers_pass_production_validation(tmp_path):
    cfg = complete_run(tmp_path)
    protocol.validate_run(tmp_path, cfg)


@pytest.mark.parametrize("defect", ["empty", "nan_json", "nan_csv", "truncated_epochs", "duplicate_epoch",
                                      "bad_grid", "wrong_metrics", "different_fixture", "wrong_logp",
                                      "corrupt_png", "wrong_config", "wrong_provenance"])
def test_production_rejects_invalid_artifacts(tmp_path, defect):
    cfg = complete_run(tmp_path)
    if defect == "empty":
        (tmp_path / "results/REPORT.md").write_text("")
    elif defect == "nan_json":
        edit_json(tmp_path, "ekf_predictions.json", lambda records: records[0].update(log_likelihood=float("nan")))
    elif defect in ("nan_csv", "truncated_epochs", "duplicate_epoch"):
        path = tmp_path / "results/logz_history.csv"
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        if defect == "nan_csv":
            rows[0]["train_logz"] = "nan"
        elif defect == "truncated_epochs":
            rows = [r for r in rows if r["epoch"] == "1"]
        else:
            rows[1]["epoch"] = "1"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif defect == "bad_grid":
        edit_json(tmp_path, "ekf_predictions.json", lambda records: records[0]["score_probabilities"][0].update(probability=-.1))
    elif defect == "wrong_metrics":
        edit_json(tmp_path, "ekf_metrics.json", lambda metrics: metrics["worldcup"].update(mean_brier_score=.987))
    elif defect == "different_fixture":
        edit_json(tmp_path, "sqmc_predictions.json", lambda records: records[0].update(home="C"))
    elif defect == "wrong_logp":
        edit_json(tmp_path, "ekf_predictions.json", lambda records: records[0].update(log_likelihood=-100.))
    elif defect == "corrupt_png":
        (tmp_path / "images/ekf_pre_worldcup_rankings.png").write_bytes(b"not a PNG")
    elif defect == "wrong_config":
        cfg = dict(cfg, n_epochs=100)
    elif defect == "wrong_provenance":
        edit_json(tmp_path, "run_config.json", lambda value: value.update(config_sha256="b" * 64))
    with pytest.raises(ValueError):
        protocol.validate_run(tmp_path, cfg)


def test_empty_placeholder_bundle_is_rejected(tmp_path):
    for name in ("results/summary.json", "results/performance_metrics.csv", "results/logz_history.csv",
                 "results/REPORT.md", "results/DRAFT.md"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    with pytest.raises(ValueError, match="Missing or empty"):
        protocol.validate_run(tmp_path, {"n_epochs": 100})


def test_latest_timestamp_is_chronological_across_months(tmp_path):
    for name in ("31082026_1200", "01092026_1200"):
        (tmp_path / name).mkdir()
    assert _latest_run_dir(tmp_path).name == "01092026_1200"


def test_root_refresh_preserves_completed_results_and_rejects_repeat_results(tmp_path, monkeypatch):
    scripts = Path(comparison.__file__).parent / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import run_sqmc_ekf_local as local
    remote, output = tmp_path / "remote", tmp_path / "downloaded"
    cfg = complete_run(remote)
    output.mkdir()
    write_json(remote / "comparison_config.json", cfg)
    write_json(remote / "run_config.json", dict(source_commit=cfg["source_commit"],
                                                config_sha256=protocol.config_digest(cfg)))
    write_json(remote / "remote_status.json", dict(run=dict(execution="pending")))
    (remote / "remote_logs.txt").write_text("setup done\n")
    protocol.make_archive(remote, "root", cfg)
    protocol.make_archive(remote, "run", cfg)

    def transfer(argv, **kwargs):
        assert argv[1] == "download"
        shutil.copyfile(remote / Path(argv[-2]).name, argv[-1])
        return ""

    launcher = local.Launcher(cfg, output, run=transfer)
    launcher.download("root")
    assert launcher.status["run"]["download"] == "pending"
    launcher.download("run")
    assert launcher.status["run"]["download"] == "complete"
    write_json(remote / "remote_status.json", dict(run=dict(execution="complete")))
    protocol.make_archive(remote, "root", cfg)
    launcher.download("root")
    assert launcher.status["run"]["download"] == "complete"
    assert json.loads((output / "remote_status.json").read_text())["run"]["execution"] == "complete"
    with pytest.raises(RuntimeError, match="complete download"):
        launcher.download("run")


def test_download_rejects_semantically_invalid_but_checksummed_results(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(comparison.__file__).parent / "scripts"))
    import run_sqmc_ekf_local as local
    remote, output = tmp_path / "remote", tmp_path / "downloaded"
    cfg = complete_run(remote)
    output.mkdir()
    (remote / "results/logz_history.csv").write_text("method,epoch,train_logz,test_logz\n")
    protocol.make_archive(remote, "run", cfg)
    def transfer(argv, **kwargs):
        shutil.copyfile(remote / Path(argv[-2]).name, argv[-1])
    launcher = local.Launcher(cfg, output, run=transfer)
    with pytest.raises(ValueError, match="Empty CSV"):
        launcher.download("run")
    assert launcher.status["run"]["download"] != "complete"
    assert not (output / "results").exists()


def test_pre_worldcup_ekf_ranking_is_independent_of_first_worldcup_result():
    # The two WC rows share a timestamp; the first must not affect the pre-WC
    # state, but it must affect the next state. Use the actual factorial filter.
    inputs = ekf.MatchInputs(jnp.array([0, 1, 0, 1]), jnp.array([1, 0, 1, 0]),
                             jnp.array([[1., 0.], [0., 0.], [3., 0.], [1., 1.]]),
                             jnp.zeros(4, dtype=bool), jnp.array([1., 2., 3., 3.]),
                             jnp.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]))
    params = ekf.constrain(ekf.initial_raw(jnp.eye(2)))
    data = SimpleNamespace(inputs=inputs)
    states, _ = comparison._ekf_states(data, params, 2)
    changed = SimpleNamespace(inputs=inputs._replace(score=inputs.score.at[2].set(jnp.array([0., 3.]))))
    changed_states, _ = comparison._ekf_states(changed, params, 2)
    assert states.particles.x.shape == (5, 1, 2, 2)
    assert states.ekf_cov.shape == (5, 2, 2, 2)
    np.testing.assert_allclose(states.particles.x[0, 0], jnp.zeros((2, 2)))
    np.testing.assert_allclose(states.ekf_cov[0], jnp.broadcast_to(params["init_cov"], (2, 2, 2)))
    np.testing.assert_allclose(states.particles.x[2], changed_states.particles.x[2])
    assert not np.allclose(states.particles.x[3], changed_states.particles.x[3])
