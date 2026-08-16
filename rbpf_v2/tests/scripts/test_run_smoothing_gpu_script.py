import json

from rbpf_v2.scripts.run_smoothing_gpu import load_config, main, training_command


def test_gpu_runner_builds_run_smoothing_command(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "n_particles": 7,
        "n_smoother_paths": 9,
        "initial_params": "rbpf_v2/initial.json",
    }))
    config = load_config(tmp_path, str(config_path))
    command = training_command(config)
    assert command[:4] == [command[0], "-u", "-m", "rbpf_v2.scripts.run_smoothing"]
    assert command[command.index("--n-particles") + 1] == "7"
    assert command[command.index("--n-smoother-paths") + 1] == "9"
    assert command[command.index("--initial-params") + 1] == "rbpf_v2/initial.json"
    assert main(["--config", str(config_path), "--dry-run"]) == 0
    assert "rbpf_v2.scripts.run_smoothing" in capsys.readouterr().out
