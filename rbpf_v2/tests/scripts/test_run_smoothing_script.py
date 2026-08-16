import json

from rbpf_v2.scripts.run_smoothing import main


def test_run_smoothing_public_entrypoint(tmp_path):
    code = main([
        "--synthetic", "--output-dir", str(tmp_path),
        "--n-particles", "8", "--n-smoother-paths", "8",
        "--n-epochs", "1", "--n-gradient-steps", "1",
    ])
    assert code == 0
    params = json.loads((tmp_path / "em_final_params.json").read_text())
    assert set(params) == {"mean_0", "gamma_0", "B", "kappa", "alpha", "beta"}
    assert (tmp_path / "em_initial_params.json").stat().st_size > 0
    assert (tmp_path / "evaluation_summary.json").stat().st_size > 0
