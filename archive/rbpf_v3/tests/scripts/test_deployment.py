import json
from pathlib import Path
import subprocess
import sys

from rbpf_v3.scripts.run_smoothing_gpu import load_config, training_command


def test_gpu_command_targets_cuthbert_v3_and_forwards_initial_params(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "n_particles": 7,
                "n_smoother_paths": 9,
                "n_epochs": 1,
                "n_gradient_steps": 1,
                "learning_rate": 0.001,
                "max_goals": 8,
                "holdout_days": 1,
                "seed": 1,
                "initial_params": "rbpf_v3/initial.json",
                "output_dir": "rbpf_v3/outputs/smoothing",
                "gpu_type": "L4",
                "colab_timeout": 60,
                "repo_url": "https://example.invalid/repo.git",
            }
        )
    )
    config = load_config(tmp_path, str(config_path))
    command = training_command(config)
    assert command[:4] == [sys.executable, "-u", "-m", "rbpf_v3.scripts.run_smoothing"]
    assert command[command.index("--n-particles") + 1] == "7"
    assert command[command.index("--initial-params") + 1] == "rbpf_v3/initial.json"


def test_colab_launcher_syntax_and_dry_run_from_outside_repo():
    root = Path(__file__).resolve().parents[3]
    launcher = root / "rbpf_v3/run_smoothing_colab.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True, cwd="/tmp")
    completed = subprocess.run(
        ["bash", str(launcher), "--dry-run"],
        check=True,
        cwd="/tmp",
        text=True,
        capture_output=True,
    )
    assert "colab run --gpu L4" in completed.stdout
    assert "rbsqmc-rbpf-v3-smoothing" in completed.stdout
