from rbpf_v2.scripts.smoke_test import main


def test_smoke_public_entrypoint(tmp_path):
    assert main(["--output-dir", str(tmp_path), "--n-particles", "8",
                 "--n-smoother-paths", "8", "--n-gradient-steps", "1"]) == 0
    assert (tmp_path / "evaluation_summary.json").exists()
