import json

import pytest

from rbpf_v3.scripts import run_smoothing, run_smoothing_noncuthbert
from rbpf_v3.scripts.validate_outputs import REQUIRED, validate


@pytest.mark.parametrize("module", [run_smoothing, run_smoothing_noncuthbert])
def test_standalone_runner_writes_matching_artifacts(module, tmp_path):
    assert module.main(
        [
            "--synthetic",
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
