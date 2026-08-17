import json

import jax

from rbpf_v2.src.evaluation import REQUIRED_PLOTS, evaluate_run
from rbpf_v2.src.smoothing import MCEMConfig, run_mcem


def test_required_artifacts(tmp_path, small_model):
    _, data, params = small_model
    result = run_mcem(jax.random.key(8), data, params, MCEMConfig(8, 8, 1, 1, 1e-3, 8, 1e-6))
    report = evaluate_run(result, data, seed=8, output_dir=tmp_path)
    assert report["passed"]
    json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert (tmp_path / "baseline_comparison.json").stat().st_size
    for name in REQUIRED_PLOTS:
        assert (tmp_path / name).stat().st_size
