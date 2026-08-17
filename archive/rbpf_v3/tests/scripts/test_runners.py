import json

import jax.numpy as jnp
import numpy as np
import pytest

from rbpf_v3.scripts import run_smoothing, run_smoothing_noncuthbert
from rbpf_v3.scripts.validate_outputs import REQUIRED, validate


@pytest.mark.parametrize("module", [run_smoothing, run_smoothing_noncuthbert])
def test_standalone_runner_writes_matching_artifacts(module, tmp_path, monkeypatch):
    supplied_params = module.default_init_params(2)._replace(
        mean_0=jnp.full((2, 2), 0.25)
    )
    monkeypatch.setattr(module, "load_params", lambda _: supplied_params)
    assert module.main(
        [
            "--synthetic",
            "--initial-params",
            "nonzero-mean.json",
            "--output-dir",
            str(tmp_path),
            "--n-particles",
            "6",
            "--n-smoother-paths",
            "5",
            "--n-epochs",
            "1",
            "--n-gradient-steps",
            "1",
        ]
    ) == 0
    validate(tmp_path)
    performance = json.loads((tmp_path / "performance_summary.json").read_text())
    assert performance["backend"] == module.BACKEND
    assert all((tmp_path / name).stat().st_size > 0 for name in REQUIRED)
    for filename in ("em_initial_params.json", "em_final_params.json"):
        saved = json.loads((tmp_path / filename).read_text())
        np.testing.assert_array_equal(saved["mean_0"], np.zeros((2, 2)))
