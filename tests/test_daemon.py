from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench import cli, daemon


def _state_dir_patch(tempdir: str):
    return patch("jongbench.daemon.state_dir", return_value=Path(tempdir))


def test_state_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tempdir, _state_dir_patch(tempdir):
        assert daemon.load_state() is None
        daemon.save_state(1234, ["--port", "8642"], "127.0.0.1", 8642)
        state = daemon.load_state()
        assert state is not None
        assert state["pid"] == 1234
        assert state["args"] == ["--port", "8642"]
        assert state["host"] == "127.0.0.1"
        assert state["port"] == 8642
        daemon.clear_state()
        assert daemon.load_state() is None


def test_corrupt_state_treated_as_missing() -> None:
    with tempfile.TemporaryDirectory() as tempdir, _state_dir_patch(tempdir):
        daemon.state_path().write_text("not json", encoding="utf-8")
        assert daemon.load_state() is None
        daemon.state_path().write_text('{"pid": "nope"}', encoding="utf-8")
        assert daemon.load_state() is None


def test_stale_pid_cleaned_up() -> None:
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    with tempfile.TemporaryDirectory() as tempdir, _state_dir_patch(tempdir):
        daemon.save_state(proc.pid, [], "127.0.0.1", 8642)
        assert daemon.running_state() is None
        assert not daemon.state_path().exists()
        assert daemon.status() == 1
        assert daemon.stop() == 0


def test_start_refuses_when_running() -> None:
    with tempfile.TemporaryDirectory() as tempdir, _state_dir_patch(tempdir):
        daemon.save_state(os.getpid(), [], "127.0.0.1", 8642)
        with patch("jongbench.daemon.subprocess.Popen") as popen:
            assert daemon.start([], "127.0.0.1", 8642) == 1
        popen.assert_not_called()


def test_start_success_records_state() -> None:
    child = MagicMock()
    child.pid = 4321
    child.poll.return_value = None
    with tempfile.TemporaryDirectory() as tempdir, _state_dir_patch(tempdir):
        with (
            patch("jongbench.daemon.subprocess.Popen", return_value=child) as popen,
            patch("jongbench.daemon._port_open", return_value=True),
        ):
            assert daemon.start(["--port", "9000"], "127.0.0.1", 9000) == 0
        argv = popen.call_args.args[0]
        assert argv[:5] == [sys.executable, "-m", "jongbench", "serve", "--foreground"]
        assert argv[5:] == ["--port", "9000"]
        state = daemon.load_state()
        assert state is not None
        assert state["pid"] == 4321
        assert state["args"] == ["--port", "9000"]


def test_start_reports_child_death() -> None:
    child = MagicMock()
    child.pid = 4321
    child.poll.return_value = 1
    with tempfile.TemporaryDirectory() as tempdir, _state_dir_patch(tempdir):
        daemon.log_path().write_text("boom", encoding="utf-8")
        with (
            patch("jongbench.daemon.subprocess.Popen", return_value=child),
            patch("jongbench.daemon._port_open", return_value=False),
        ):
            assert daemon.start([], "127.0.0.1", 8642) == 1
        assert daemon.load_state() is None


def test_restart_requires_saved_state() -> None:
    with tempfile.TemporaryDirectory() as tempdir, _state_dir_patch(tempdir):
        assert daemon.restart() == 1


def test_restart_reuses_saved_args() -> None:
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    with tempfile.TemporaryDirectory() as tempdir, _state_dir_patch(tempdir):
        daemon.save_state(proc.pid, ["--port", "9000"], "0.0.0.0", 9000)
        with patch("jongbench.daemon.start", return_value=0) as start:
            assert daemon.restart() == 0
        start.assert_called_once_with(["--port", "9000"], "0.0.0.0", 9000)


def test_serve_cli_parsing() -> None:
    parser = cli._build_parser()
    parsed = parser.parse_args(["serve"])
    assert parsed.action is None
    assert parsed.foreground is False
    parsed = parser.parse_args(["serve", "stop"])
    assert parsed.action == "stop"
    parsed = parser.parse_args(["serve", "--foreground", "--port", "9000"])
    assert parsed.foreground is True
    assert parsed.port == 9000

    for verb, target in (("stop", "stop"), ("restart", "restart"), ("status", "status")):
        with patch(f"jongbench.daemon.{target}", return_value=0) as func:
            assert cli.main(["serve", verb]) == 0
        func.assert_called_once_with()


def test_serve_cli_start_resolves_paths() -> None:
    with patch("jongbench.daemon.start", return_value=0) as start:
        assert cli.main(["serve", "--runs-root", "runs", "--weights", "weights/mortal.pth"]) == 0
    flags, host, port = start.call_args.args
    assert host == "127.0.0.1"
    assert port == 8642
    runs_root = flags[flags.index("--runs-root") + 1]
    weights = flags[flags.index("--weights") + 1]
    assert Path(runs_root).is_absolute()
    assert Path(weights).is_absolute()


if __name__ == "__main__":
    test_state_roundtrip()
    test_serve_cli_parsing()
    print("OK")
