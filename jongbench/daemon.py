from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

START_TIMEOUT = 10.0
STOP_TIMEOUT = 10.0


def state_dir() -> Path:
    return Path.home() / ".jongbench"


def state_path() -> Path:
    return state_dir() / "serve.json"


def log_path() -> Path:
    return state_dir() / "serve.log"


def load_state() -> dict | None:
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pid"), int):
        return None
    return data


def save_state(pid: int, args: list[str], host: str, port: int) -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    data = {
        "pid": pid,
        "args": args,
        "host": host,
        "port": port,
        "started": datetime.now(timezone.utc).isoformat(),
    }
    state_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_state() -> None:
    state_path().unlink(missing_ok=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_state() -> dict | None:
    state = load_state()
    if state is None:
        return None
    if not pid_alive(state["pid"]):
        clear_state()
        return None
    return state


def _connect_host(host: str) -> str:
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((_connect_host(host), port), timeout=1.0):
            return True
    except OSError:
        return False


def _log_tail(lines: int = 20) -> str:
    try:
        content = log_path().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(content.splitlines()[-lines:])


def start(args: list[str], host: str, port: int) -> int:
    state = running_state()
    if state is not None:
        print(
            f"error: server already running (pid {state['pid']}, "
            f"http://{state['host']}:{state['port']}) — use 'jongbench serve stop'",
            file=sys.stderr,
        )
        return 1

    state_dir().mkdir(parents=True, exist_ok=True)
    with log_path().open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "jongbench", "serve", "--foreground", *args],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    save_state(proc.pid, args, host, port)

    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            clear_state()
            print(f"error: server exited during startup:\n{_log_tail()}", file=sys.stderr)
            return 1
        if _port_open(host, port):
            print(f"serving on http://{host}:{port} (pid {proc.pid}, log {log_path()})")
            return 0
        time.sleep(0.2)

    print(f"error: server did not come up within {START_TIMEOUT:.0f}s:\n{_log_tail()}", file=sys.stderr)
    return 1


def stop() -> int:
    state = running_state()
    if state is None:
        print("not running")
        return 0
    pid = state["pid"]
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            clear_state()
            print(f"stopped (pid {pid})")
            return 0
        time.sleep(0.2)
    os.kill(pid, signal.SIGKILL)
    clear_state()
    print(f"killed (pid {pid})")
    return 0


def restart() -> int:
    state = load_state()
    if state is None:
        print("error: no previous server state — use 'jongbench serve'", file=sys.stderr)
        return 1
    code = stop()
    if code != 0:
        return code
    return start(list(state["args"]), state["host"], state["port"])


def status() -> int:
    state = running_state()
    if state is None:
        print("not running")
        return 1
    started = state.get("started")
    uptime = ""
    if isinstance(started, str):
        try:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(started)
            uptime = f", up {_fmt_duration(delta.total_seconds())}"
        except ValueError:
            pass
    print(
        f"running: http://{state['host']}:{state['port']} "
        f"(pid {state['pid']}{uptime}, log {log_path()})"
    )
    return 0


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
