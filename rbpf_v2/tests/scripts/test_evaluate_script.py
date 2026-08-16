from rbpf_v2.scripts.evaluate import main as evaluate
from rbpf_v2.scripts.train import main as train


def test_evaluate_public_entrypoint(tmp_path):
    model = tmp_path / "model"
    output = tmp_path / "evaluation"
    assert train(["--output-dir", str(model), "--n-particles", "8",
                  "--n-smoother-paths", "8", "--n-gradient-steps", "1"]) == 0
    assert evaluate(["--model-dir", str(model), "--output-dir", str(output),
                     "--n-particles", "8", "--n-smoother-paths", "8"]) == 0
