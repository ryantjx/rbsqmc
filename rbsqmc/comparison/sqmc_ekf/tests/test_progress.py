"""Regression checks for Colab polling, diagnostics and expiring credentials."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rbsqmc.comparison.sqmc_ekf.scripts import sqmc_ekf_protocol as protocol
from rbsqmc.comparison.sqmc_ekf.scripts.refresh_colab_proxy import refresh


@pytest.fixture
def local(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(protocol.__file__).parent))
    return importlib.import_module("run_sqmc_ekf_local")


@pytest.fixture
def launcher(local, tmp_path):
    config = dict(session_name="test", run_id="test", source_commit="a" * 40,
                  transfer_timeout=10, colab_timeout=60)
    return local.Launcher(config, tmp_path, sleep=lambda seconds: None)


def test_quiet_command_preserves_original_failure_through_tee(local):
    terminal, log = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(local.Tee(terminal, log)):
        with pytest.raises(subprocess.CalledProcessError) as caught:
            local.command([sys.executable, "-c", "print('HTTP 404'); raise SystemExit(7)"], quiet=True)
    assert caught.value.returncode == 7
    assert caught.value.output == "HTTP 404\n"
    assert terminal.getvalue() == log.getvalue()
    assert "HTTP 404" in terminal.getvalue()


def test_snapshot_filters_remotely_and_preserves_partial_utf8_lines(tmp_path):
    protocol.write_json(tmp_path / "remote_status.json", {"run": {"execution": "running"}})
    log = tmp_path / "remote_logs.txt"
    epoch = "sqmc epoch 16/50: train -15178.6966, test -1083.8652, |grad| 482.8749, 178.5s\n"
    next_epoch = "sqmc epoch 17/50: train -15283.8713, test -1012.9015, |grad| 471.2261, 178.2s\n"
    noise = b"Installing dependency\n" * 10000
    partial = "sqmc epoch 18/50: diagnostic \u03bc".encode()
    log.write_bytes(noise + epoch.encode() + next_epoch.encode() + partial[:-1])
    packet = protocol.progress_snapshot(tmp_path)
    assert packet["lines"] == [epoch, next_epoch]
    assert len(json.dumps(packet)) < 500
    assert packet["offset"] == len(noise) + len(epoch) + len(next_epoch)
    assert protocol.progress_snapshot(tmp_path, packet["offset"])["lines"] == []
    with log.open("ab") as stream:
        stream.write(partial[-1:] + b"\n")
    after = protocol.progress_snapshot(tmp_path, packet["offset"])
    assert after["lines"] == [partial.decode() + "\n"]
    log.write_bytes(b"")
    with pytest.raises(ValueError, match="shrank"):
        protocol.progress_snapshot(tmp_path, after["offset"])


def test_status_only_does_not_read_log(tmp_path):
    protocol.write_json(tmp_path / "remote_status.json", {"run": {"execution": "running"}})
    assert protocol.progress_snapshot(tmp_path, 123, False) == {
        "run": {"execution": "running"}, "offset": 123, "lines": [],
    }


def test_generated_remote_poll_code_prints_each_epoch_once(local, launcher, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(local, "REMOTE_REPO", Path(__file__).resolve().parents[4])
    launcher.remote = tmp_path
    protocol.write_json(tmp_path / "remote_status.json", {"run": {"execution": "running"}})
    epoch = "ekf epoch 1/50: train -10, test -2, |grad| 1, 0.1s\n"
    (tmp_path / "remote_logs.txt").write_text("pip setup chatter\n" + epoch)

    def remote_exec(code):
        return subprocess.check_output([sys.executable, "-c", code], text=True)

    monkeypatch.setattr(launcher, "execute", remote_exec)
    assert launcher.poll_run()["execution"] == "running"
    launcher.poll_run()
    assert capsys.readouterr().out == epoch
    offset = launcher.log_offset
    monkeypatch.setattr(launcher, "execute", lambda code: "remote exception, no packet")
    with pytest.raises(RuntimeError, match="no unique progress"):
        launcher.poll_run()
    assert launcher.log_offset == offset


@pytest.mark.parametrize("persistent", [False, True])
def test_poll_transport_failure_retries_but_never_completes_on_error(launcher, monkeypatch, persistent):
    calls = []

    def poll():
        calls.append(True)
        if persistent or len(calls) < 3:
            raise subprocess.CalledProcessError(1, ["colab", "exec"], output="connection failed")
        return {"execution": "complete", "archive_ready": True}

    launcher.proxy_refresh_at = float("inf")
    monkeypatch.setattr(launcher, "poll_run", poll)
    if persistent:
        with pytest.raises(subprocess.CalledProcessError):
            launcher.wait_run()
        assert launcher.status["run"]["execution"] != "complete"
    else:
        launcher.wait_run()
        assert launcher.status["run"]["execution"] == "complete"
    assert len(calls) == 3
    assert launcher.proxy_refresh_at == 0


def test_remote_model_failure_is_not_retried(launcher, monkeypatch):
    def poll():
        return {"execution": "failed", "archive_ready": True, "error": "Non-finite gradient"}

    monkeypatch.setattr(launcher, "poll_run", poll)
    with pytest.raises(RuntimeError, match="Non-finite gradient"):
        launcher.wait_run()


def test_proxy_refresh_updates_credentials_only_for_owned_endpoint():
    session = SimpleNamespace(endpoint="owned", token="old", url="old-url", kernel_id="kernel")
    writes = []
    proxy = SimpleNamespace(token="new", url="new-url", token_expires_in_seconds=3600)
    assignments = [SimpleNamespace(endpoint="owned", runtime_proxy_info=proxy)]
    state = SimpleNamespace(store=SimpleNamespace(get=lambda name: session, add=writes.append),
                            client=SimpleNamespace(list_assignments=lambda: assignments))
    assert refresh(state, "test", "owned") == {"expires_in_seconds": 3600}
    assert (session.token, session.url, session.kernel_id) == ("new", "new-url", "kernel")
    assert writes == [session]
    with pytest.raises(RuntimeError, match="endpoint changed"):
        refresh(state, "test", "different")
    assignments.clear()
    with pytest.raises(RuntimeError, match="no longer assigned"):
        refresh(state, "test", "owned")
    assert len(writes) == 1


def test_launcher_refreshes_before_expiry_using_cli_interpreter(local, launcher, tmp_path, monkeypatch):
    cli = tmp_path / "colab"
    cli.write_text("#!/cli-env/bin/python3\n")
    launcher.colab, launcher.endpoint = str(cli), "owned"
    now = [100]
    monkeypatch.setattr(local.time, "monotonic", lambda: now[0])
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return '{"expires_in_seconds": 3600}' if argv[0] == "/cli-env/bin/python3" else ""

    launcher.run = run
    launcher.call("download", "first")
    launcher.call("exec", "second")
    assert len(calls) == 3
    assert calls[0][-2:] == ["test", "owned"]
    now[0] += 1201
    launcher.call("download", "third")
    assert len(calls) == 5
    assert calls[3][0] == "/cli-env/bin/python3"
