from __future__ import annotations

import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench import webui


HOST = "127.0.0.1"
PORT = 0


def test_pending_option_metadata() -> None:
    discard = webui._pending_option(
        3,
        2,
        {"label": "discard 0m (red)", "event": {"type": "dahai", "pai": "5mr"}},
    )
    assert discard == {
        "choice": 3,
        "action": "discard",
        "label": "discard 0m (red)",
        "tile": "5mr",
    }
    assert webui._pending_option(0, 2, {"label": "tsumo (win)", "event": {"type": "hora", "target": 2}})["action"] == "tsumo"
    assert webui._pending_option(1, 2, {"label": "ron (win)", "event": {"type": "hora", "target": 1}})["action"] == "ron"
    assert webui._pending_option(2, 2, {"label": "pass", "event": {"type": "none"}})["action"] == "pass"

    for invalid in (True, -1, 2**64, [1.5, 2], [True, 2]):
        try:
            webui._normalize_seed(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid seed: {invalid!r}")


def test_web_server_workflow() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-webui-") as tempdir:
        thread = threading.Thread(
            target=webui.run_server,
            kwargs={
                "host": HOST,
                "port": 0,
                "runs_root": tempdir,
                "no_eval": True,
                "max_concurrent_games": 2,
            },
            daemon=True,
        )
        thread.start()
        base = _wait_for_server()
        parsed = urlparse(base)
        global PORT
        PORT = int(parsed.port or 0)

        status, text = _request_text("GET", "/")
        assert status == 200
        assert "jongbench" in text

        run = _request_json(
            "POST",
            "/api/start",
            {"models": ["random", "random", "random", "random"], "keys": {}, "seed": 4242},
        )
        run_id = run["run_id"]
        state = _wait_done(run_id, 180)
        assert state["status"] == "done", state
        frames = _collect_frames(run_id, 0, 5, 30)
        assert len(frames) >= 5
        seqs = [frame["seq"] for frame in frames]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)
        for frame in frames:
            snapshot = frame.get("snapshot")
            assert isinstance(snapshot, dict)
            assert "seats" in snapshot or "hands" in snapshot

        human = _request_json(
            "POST",
            "/api/start",
            {
                "models": ["human", "random", "random", "random"],
                "keys": {},
                "seed": 555,
                "human_seat": 0,
            },
        )
        human_id = human["run_id"]
        answered = 0
        mask_checked = False
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            state = _request_json("GET", f"/api/state/{human_id}")
            if state["status"] == "error":
                raise AssertionError(state)
            if state["status"] == "done":
                break
            pending = _request_json("GET", f"/api/pending/{human_id}")
            if pending:
                assert isinstance(pending.get("state_text"), str)
                assert pending["state_text"].strip()
                assert isinstance(pending.get("options"), list)
                assert pending["options"]
                for option in pending["options"]:
                    assert isinstance(option.get("choice"), int)
                    assert isinstance(option.get("action"), str)
                    assert isinstance(option.get("label"), str)
                if not mask_checked and answered >= 1:
                    sample = _collect_frames(human_id, 0, 3, 10)
                    mask_checked = _assert_masked_frame(sample)
                result = _request_json(
                    "POST",
                    f"/api/choose/{human_id}",
                    {"generation": pending["generation"], "choice": 0},
                )
                assert result == {"ok": True}, result
                answered += 1
            else:
                time.sleep(0.05)
        else:
            raise AssertionError("human game timed out")

        state = _request_json("GET", f"/api/state/{human_id}")
        assert state["status"] == "done", state
        assert answered >= 5, answered
        if not mask_checked:
            sample = _collect_frames(human_id, 0, 20, 20)
            mask_checked = _assert_masked_frame(sample)
        assert mask_checked

        _test_abort_frees_slot()

    print("OK")


def _test_abort_frees_slot() -> None:
    started = [
        _start_when_allowed(
            {
                "models": ["human", "random", "random", "random"],
                "keys": {},
                "seed": 777 + i,
                "human_seat": 0,
            }
        )
        for i in range(2)
    ]
    conflict, _ = _request_text(
        "POST",
        "/api/start",
        {"models": ["random"] * 4, "keys": {}, "seed": 999},
    )
    assert conflict == 409, conflict

    aborted = _request_json("POST", f"/api/abort/{started[0]}")
    assert aborted == {"ok": True}, aborted
    status, _ = _request_text("GET", f"/api/state/{started[0]}")
    assert status == 404, status
    listed = {item["run_id"] for item in _request_json("GET", "/api/sessions")}
    assert started[0] not in listed, listed
    status, _ = _request_text("POST", f"/api/abort/{started[0]}")
    assert status == 404, status

    replacement = _start_when_allowed({"models": ["random"] * 4, "keys": {}, "seed": 999})
    _request_json("POST", f"/api/abort/{started[1]}")
    _wait_done(replacement, 180)


def _start_when_allowed(payload: dict[str, Any]) -> str:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status, text = _request_text("POST", "/api/start", payload)
        if status == 200:
            return str(json.loads(text)["run_id"])
        assert status in {409, 429}, (status, text)
        time.sleep(1.0)
    raise AssertionError("start was never allowed")


def _wait_for_server() -> str:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        url = webui.last_server_url()
        if url:
            return url
        time.sleep(0.02)
    raise AssertionError("server did not start")


def _request_json(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    status, text = _request_text(method, path, body)
    data = json.loads(text or "null")
    if not 200 <= status < 300:
        raise AssertionError((status, data))
    return data


def _request_text(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, str]:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    headers = {}
    raw: bytes | None = None
    if body is not None:
        raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=raw, headers=headers)
    response = conn.getresponse()
    data = response.read().decode("utf-8")
    status = int(response.status)
    conn.close()
    return status, data


def _wait_done(run_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _request_json("GET", f"/api/state/{run_id}")
        if state["status"] == "error":
            raise AssertionError(state)
        if state["status"] == "done":
            return state
        time.sleep(0.2)
    raise AssertionError(f"run {run_id} timed out")


def _collect_frames(run_id: str, since: int, min_frames: int, timeout: float) -> list[dict[str, Any]]:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    conn.request("GET", f"/api/events/{run_id}?since={since}")
    response = conn.getresponse()
    assert response.status == 200, response.status
    frames: list[dict[str, Any]] = []
    event_type: str | None = None
    data_lines: list[str] = []
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline and len(frames) < min_frames:
            try:
                line = response.readline()
            except TimeoutError:
                break
            if not line:
                break
            text = line.decode("utf-8")
            if text in {"\n", "\r\n"}:
                if event_type == "frame" and data_lines:
                    frames.append(json.loads("".join(data_lines)))
                event_type = None
                data_lines = []
            elif text.startswith("event:"):
                event_type = text.split(":", 1)[1].strip()
            elif text.startswith("data:"):
                data_lines.append(text.split(":", 1)[1].strip())
    finally:
        conn.close()
    return frames


def _assert_masked_frame(frames: list[dict[str, Any]]) -> bool:
    start_event_masked = False
    for frame in frames:
        event = frame.get("event") or {}
        if event.get("type") == "start_kyoku":
            hands = event.get("tehais") or []
            assert len(hands) == 4
            assert any(tile != "?" for tile in hands[0])
            for hand in hands[1:]:
                assert hand and all(tile == "?" for tile in hand), hand
            start_event_masked = True
        if event.get("type") == "tsumo" and event.get("actor") != 0:
            assert "pai" not in event, event

    for frame in frames:
        snapshot = frame.get("snapshot") or {}
        seats = snapshot.get("seats") or []
        if len(seats) != 4:
            continue
        seat0 = seats[0].get("hand") or []
        opponents = [seats[idx].get("hand") or [] for idx in (1, 2, 3)]
        if not seat0 or not all(opponents):
            continue
        assert any(tile != "?" for tile in seat0), seat0
        for hand in opponents:
            assert all(tile == "?" for tile in hand), hand
        for line in snapshot.get("ticker") or []:
            if line.startswith(("P1 drew ", "P2 drew ", "P3 drew ")):
                assert line.endswith("drew a tile"), line
        return start_event_masked
    return False


if __name__ == "__main__":
    test_pending_option_metadata()
    test_web_server_workflow()
