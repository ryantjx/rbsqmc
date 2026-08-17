from rbpf_v2.scripts.train import main


def test_train_public_entrypoint(tmp_path):
    assert main(["--output-dir", str(tmp_path), "--n-particles", "8",
                 "--n-smoother-paths", "8", "--n-gradient-steps", "1"]) == 0
    assert (tmp_path / "training_summary.json").exists()
