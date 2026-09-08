"""Validate comparison contents before completion and after archive download.

Standard-library only so the local Colab launcher needs no scientific packages.
Accept an exact run directory, or select the chronologically latest timestamped
run under an outputs directory when invoked as a CLI.
"""

import csv
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib


METHODS = ("ekf", "sqmc")
REQUIRED_RESULTS = (
    "comparison_config.json", "run_config.json", "dataset_metadata.json",
    "run_metadata.json", "summary.json", "performance_metrics.csv",
    "logz_history.csv", "REPORT.md", "DRAFT.md",
)
REQUIRED_METHOD_IMAGES = (
    "top5_strengths.png", "timeseries_states.png",
    "pre_worldcup_rankings.png", "post_worldcup_rankings.png",
)
FIXTURE_FIELDS = ("date", "home", "away", "actual_home_score", "actual_away_score",
                  "tournament", "worldcup_eligible")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonempty(path):
    _require(path.is_file() and path.stat().st_size > 0, f"Missing or empty artifact: {path}")


def _ensure_finite(label, tree):
    if isinstance(tree, dict):
        for value in tree.values():
            _ensure_finite(label, value)
    elif isinstance(tree, list):
        for value in tree:
            _ensure_finite(label, value)
    elif isinstance(tree, (int, float)):
        _require(math.isfinite(tree), f"{label}: non-finite value {tree}")


def _json(path):
    _nonempty(path)
    value = json.loads(path.read_text())
    _ensure_finite(str(path), value)
    return value


def _csv(path):
    _nonempty(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(bool(rows), f"Empty CSV: {path}")
    return rows


def _close(actual, expected, label):
    if expected is None:
        _require(actual is None or actual == "", f"{label}: expected an unscored value")
    elif isinstance(expected, (int, float)):
        _require(not isinstance(actual, bool), f"{label}: expected a number")
        value = float(actual)
        _require(math.isfinite(value) and math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-8),
                 f"{label}: {actual} != {expected}")
    else:
        _require(actual == expected, f"{label}: {actual!r} != {expected!r}")


def _png(path):
    """Check PNG structure, chunk checksums and the compressed image stream."""
    _nonempty(path)
    data = path.read_bytes()
    _require(data[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG: {path}")
    offset, chunks, compressed = 8, [], bytearray()
    while offset < len(data):
        _require(offset + 12 <= len(data), f"Truncated PNG: {path}")
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size
        _require(end <= len(data), f"Truncated PNG chunk: {path}")
        kind, content = data[offset + 4:offset + 8], data[offset + 8:end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        _require(zlib.crc32(kind + content) == crc, f"PNG checksum mismatch: {path}")
        if not chunks:
            _require(kind == b"IHDR" and size == 13, f"Invalid PNG header: {path}")
            width, height = struct.unpack_from(">II", content)
            _require(width > 0 and height > 0, f"Empty PNG dimensions: {path}")
        if kind == b"IDAT":
            compressed.extend(content)
        chunks.append(kind)
        offset = end
        if kind == b"IEND":
            _require(size == 0 and end == len(data), f"Invalid PNG ending: {path}")
            break
    _require(chunks[-1:] == [b"IEND"] and compressed, f"Missing PNG image data: {path}")
    decoder = zlib.decompressobj()
    pixels = decoder.decompress(compressed)
    _require(bool(pixels) and decoder.eof and not decoder.unused_data, f"Invalid PNG stream: {path}")


def _predictions(records, cfg, count):
    _require(isinstance(records, list) and len(records) == count, "Prediction count mismatch")
    size = cfg["max_goals"] + 1
    previous_date = date.fromisoformat(cfg["prediction_start_date"])
    fixtures = []
    for index, record in enumerate(records):
        label = f"prediction {index}"
        when = date.fromisoformat(record["date"])
        _require(when >= previous_date, f"{label}: wrong split or chronological order")
        previous_date = when
        _require(record["home"] and record["away"] and record["home"] != record["away"],
                 f"{label}: invalid teams")
        eligible = when.year == 2026 and record["tournament"] == "FIFA World Cup"
        _require(record["worldcup_eligible"] is eligible, f"{label}: wrong World Cup eligibility")
        cells = record["score_probabilities"]
        _require(len(cells) == size**2, f"{label}: incomplete score grid")
        grid = {}
        for cell in cells:
            h, a, probability = cell["home"], cell["away"], cell["probability"]
            _require(type(h) is int and type(a) is int and 0 <= h < size and 0 <= a < size,
                     f"{label}: invalid grid coordinates")
            _require((h, a) not in grid, f"{label}: duplicate grid coordinates")
            _require(type(probability) in (int, float) and 0 <= probability <= 1,
                     f"{label}: probability outside [0, 1]")
            grid[h, a] = probability
        _close(sum(grid.values()), 1., f"{label} grid mass")
        # Verify outcome probabilities and score selection against the saved grid,
        # rather than trusting mutually inconsistent derived fields.
        probs = [sum(p for (h, a), p in grid.items() if h > a),
                 sum(p for (h, a), p in grid.items() if h == a),
                 sum(p for (h, a), p in grid.items() if h < a)]
        for field, value in zip(("prob_home_win", "prob_draw", "prob_away_win"), probs):
            _close(record[field], value, f"{label} {field}")
        best = max(range(size**2), key=lambda i: grid[divmod(i, size)])
        _require((record["predicted_home_score"], record["predicted_away_score"]) == divmod(best, size),
                 f"{label}: predicted score disagrees with grid")
        h, a = record["actual_home_score"], record["actual_away_score"]
        _require(type(h) is int and type(a) is int and -1 <= h < size and -1 <= a < size,
                 f"{label}: invalid actual scores")
        # The existing predictors use log(p + 1e-12); validate that convention
        # without changing the scoring model as part of an artifact check.
        expected_logp = math.log(grid[h, a] + 1e-12) if h >= 0 and a >= 0 else 0.
        _close(record["log_likelihood"], expected_logp, f"{label} log probability")
        fixtures.append(tuple(record[field] for field in FIXTURE_FIELDS))
    return fixtures


def _metrics(records):
    known = [r for r in records if min(r["actual_home_score"], r["actual_away_score"]) >= 0]
    brier, exact, outcome, logp = [], [], [], []
    for record in known:
        h, a = record["actual_home_score"], record["actual_away_score"]
        actual = 0 if h > a else 1 if h == a else 2
        probs = [record["prob_home_win"], record["prob_draw"], record["prob_away_win"]]
        brier.append(sum((p - int(i == actual))**2 for i, p in enumerate(probs)))
        outcome.append(max(range(3), key=probs.__getitem__) == actual)
        exact.append((record["predicted_home_score"], record["predicted_away_score"]) == (h, a))
        logp.append(record["log_likelihood"])
    n = len(known)
    mean_brier = sum(brier) / n if n else None
    return dict(n_predictions=len(records), n_scored=n, brier_score_definition="sum_over_home_draw_away",
                mean_brier_score=mean_brier, uniform_reference_brier_score=2 / 3,
                brier_skill_score_vs_uniform=1 - mean_brier / (2 / 3) if n else None,
                total_log_likelihood=sum(logp) if n else None,
                mean_log_likelihood=sum(logp) / n if n else None,
                exact_score_accuracy=sum(exact) / n if n else None,
                outcome_accuracy=sum(outcome) / n if n else None)


def validate_partial(run_dir, config, method):
    """Check a single machine's artifacts before collecting or reusing them.

    Cross-method fixture agreement and the combined reports are checked by
    validate_artifacts after combining; each partial must already be complete.
    """
    root = Path(run_dir)
    results = root / "results"
    cfg = _json(results / "comparison_config.json")
    _require(cfg == config, "Effective comparison configuration mismatch")
    provenance = _json(results / "run_config.json")
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _require(provenance["config_sha256"] == digest, "Partial config digest mismatch")
    if "source_commit" in cfg:
        _require(provenance["source_commit"] == cfg["source_commit"], "Partial source commit mismatch")
    summary = _json(results / method / "summary.json")
    history = _json(results / f"{method}_history.json")
    _require(summary["n_epochs_completed"] == cfg["n_epochs"], f"{method}: incomplete training")
    _require([row["epoch"] for row in history] == list(range(1, cfg["n_epochs"] + 1)),
             f"{method}: missing, duplicated or unordered epochs")
    for field in ("train_logz", "test_logz"):
        _close(summary[f"final_{field}"], history[-1][field], f"{method} final {field}")
    metadata = _json(results / "dataset_metadata.json")
    records = _json(results / f"{method}_predictions.json")
    _predictions(records, cfg, metadata["prediction_count"])
    metrics = _json(results / f"{method}_metrics.json")
    wc = [r for r in records if r["worldcup_eligible"]]
    _require(len(wc) == metadata["worldcup_count"], f"{method}: World Cup count mismatch")
    scored_wc = [r for r in wc if min(r["actual_home_score"], r["actual_away_score"]) >= 0]
    for group, expected in dict(all=_metrics(records), worldcup=_metrics(scored_wc) if scored_wc else None).items():
        if expected is None:
            _require(metrics[group] is None, f"{method}: metrics for unscored fixtures")
        else:
            for field, value in expected.items():
                _close(metrics[group][field], value, f"{method} {group} {field}")
    for suffix in REQUIRED_METHOD_IMAGES:
        _png(root / "images" / f"{method}_{suffix}")


def validate_artifacts(run_dir, config=None):
    """Validate exactly this run; optional config binds it to the launch request."""
    root = Path(run_dir)
    results, images = root / "results", root / "images"
    for filename in REQUIRED_RESULTS:
        _nonempty(results / filename)
    for path in results.rglob("*.json"):
        _json(path)
    cfg = _json(results / "comparison_config.json")
    if config is not None:
        _require(cfg == config, "Effective comparison configuration mismatch")
    epochs, goals = cfg["n_epochs"], cfg["max_goals"]
    _require(type(epochs) is int and epochs > 0, "Invalid configured epoch count")
    _require(type(goals) is int and goals >= 0, "Invalid configured score bound")
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for filename in ("run_config.json", "run_metadata.json"):
        provenance = _json(results / filename)
        _require(provenance["config_sha256"] == digest, f"{filename}: config digest mismatch")
        if "source_commit" in cfg:
            _require(provenance["source_commit"] == cfg["source_commit"], f"{filename}: source commit mismatch")
    metadata = _json(results / "dataset_metadata.json")
    run_metadata = _json(results / "run_metadata.json")
    count = metadata["prediction_count"]
    for field in ("train_count", "test_count", "prediction_count"):
        _require(type(metadata[field]) is int and metadata[field] > 0, f"Invalid {field}")
    _require(type(metadata["worldcup_count"]) is int and 0 <= metadata["worldcup_count"] <= count,
             "Invalid World Cup count")
    for field in ("prediction_count", "worldcup_count"):
        _require(run_metadata[field] == metadata[field], f"Run metadata {field} mismatch")
    summaries = _json(results / "summary.json")
    history = _csv(results / "logz_history.csv")
    performance = _csv(results / "performance_metrics.csv")
    _require(len(history) == 2 * epochs and {r["method"] for r in history} == set(METHODS),
             "History does not contain the configured epochs for both methods")
    _require(len(performance) == 2 and {r["method"] for r in performance} == set(METHODS),
             "Performance CSV must contain both methods exactly once")
    aligned_fixtures = None
    for method in METHODS:
        summary = _json(results / method / "summary.json")
        _require(summary == summaries[method], f"{method}: summary copies differ")
        _require(summary["n_epochs_completed"] == epochs, f"{method}: incomplete training")
        _require(summary["checkpoint_policy"] == "final_epoch", f"{method}: wrong checkpoint policy")
        _require(summary["learning_rate_schedule"] == "cosine", f"{method}: wrong optimizer schedule")
        _require(summary["gradient_replicas"] == (cfg["n_reps"] if method == "sqmc" else 1),
                 f"{method}: wrong replica count")
        rows = [r for r in history if r["method"] == method]
        _require([int(r["epoch"]) for r in rows] == list(range(1, epochs + 1)),
                 f"{method}: missing, duplicated or unordered epochs")
        for row in rows:
            _ensure_finite(f"{method} history", [float(row["train_logz"]), float(row["test_logz"])])
        best = max(rows, key=lambda row: float(row["test_logz"]))
        for field, value in (("final_train_logz", float(rows[-1]["train_logz"])),
                             ("final_test_logz", float(rows[-1]["test_logz"])),
                             ("best_test_logz", float(best["test_logz"])), ("best_test_epoch", int(best["epoch"]))):
            _close(summary[field], value, f"{method} {field}")
        records = _json(results / f"{method}_predictions.json")
        fixtures = _predictions(records, cfg, count)
        _require(aligned_fixtures is None or fixtures == aligned_fixtures, "Methods evaluated different fixtures or results")
        aligned_fixtures = fixtures
        wc = [r for r in records if r["worldcup_eligible"]]
        _require(len(wc) == metadata["worldcup_count"], f"{method}: World Cup count mismatch")
        metrics = _json(results / f"{method}_metrics.json")
        scored_wc = [r for r in wc if min(r["actual_home_score"], r["actual_away_score"]) >= 0]
        expected_metrics = dict(all=_metrics(records), worldcup=_metrics(scored_wc) if scored_wc else None)
        for group, expected in expected_metrics.items():
            if expected is None:
                _require(metrics[group] is None, f"{method}: metrics for unscored World Cup fixtures")
            else:
                for field, value in expected.items():
                    _close(metrics[group][field], value, f"{method} {group} {field}")
        row = next(r for r in performance if r["method"] == method)
        for field in row.keys() - {"method"}:
            if row[field] != "":
                _ensure_finite(f"{method} performance {field}", float(row[field]))
        headline = expected_metrics["worldcup"] or expected_metrics["all"]
        for field in ("mean_brier_score", "brier_skill_score_vs_uniform", "outcome_accuracy",
                      "exact_score_accuracy", "mean_log_likelihood", "n_scored"):
            _close(row[field], headline[field], f"{method} performance {field}")
        for field in ("final_train_logz", "final_test_logz", "best_test_logz"):
            _close(row[field], summary[field], f"{method} performance {field}")
        _close(row["compile_sec"], summary["compilation_sec"], f"{method} compilation time")
        _close(row["train_execution_sec"], summary["execution_sec"], f"{method} training time")
        times = [float(row[k]) for k in ("compile_sec", "train_execution_sec", "prediction_sec")]
        _require(all(t >= 0 for t in times), f"{method}: negative timing")
        _close(row["pipeline_sec"], sum(times), f"{method} pipeline time")
        for suffix in REQUIRED_METHOD_IMAGES:
            _nonempty(images / f"{method}_{suffix}")
    _nonempty(images / "logz_overlay_train_test.png")
    for path in images.rglob("*.png"):
        _png(path)
    print(f"OK: comparison artifacts validated: {root}")


def _latest_run_dir(outputs_dir):
    candidates = []
    for path in Path(outputs_dir).iterdir():
        if path.is_dir():
            try:
                candidates.append((datetime.strptime(path.name, "%d%m%Y_%H%M"), path))
            except ValueError:
                continue
    _require(bool(candidates), f"No timestamped runs under {outputs_dir}")
    return max(candidates)[1]


def validate_run(path, config=None):
    root = Path(path)
    if config is None and not (root / "results").is_dir():
        root = _latest_run_dir(root)
    validate_artifacts(root, config)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "outputs"
    validate_run(path)


if __name__ == "__main__":
    main()
