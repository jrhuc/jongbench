from __future__ import annotations

import http.client
import io
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

from jongbench import arena, evaluate, webui  # noqa: E402
from jongbench.engines import GameAborted  # noqa: E402


def _session() -> webui.GameSession:
    return webui.GameSession(
        names=["p0", "p1", "p2", "p3"],
        human_seat=None,
        human_io=webui.WebHumanIO(),
        run_dir="/tmp/unused-jongbench-session",
    )


def test_web_human_cancel_during_render_does_not_miss_waiter() -> None:
    io = webui.WebHumanIO()
    rendering = threading.Event()
    release_render = threading.Event()
    original = webui.prompts.render_state
    outcome: list[BaseException] = []

    def blocked_render(*args: Any, **kwargs: Any) -> str:
        del args, kwargs
        rendering.set()
        assert release_render.wait(5)
        return "state"

    def ask() -> None:
        try:
            io.ask(0, object(), [], [{"label": "pass", "event": {"type": "none"}}])
        except BaseException as exc:
            outcome.append(exc)

    webui.prompts.render_state = blocked_render
    try:
        thread = threading.Thread(target=ask)
        thread.start()
        assert rendering.wait(5)
        io.cancel()
        release_render.set()
        thread.join(5)
        assert not thread.is_alive()
        assert len(outcome) == 1 and isinstance(outcome[0], GameAborted), outcome
    finally:
        webui.prompts.render_state = original


def test_terminal_statuses_and_transition_history_are_stable() -> None:
    session = _session()
    assert session.set_status("running")
    session.set_error("spectator broke")
    assert not session.set_status("evaluating")
    assert not session.set_status("done")
    assert session.status_snapshot() == ("error", "spectator broke")
    assert [status for _, status, _ in session.status_transitions_after(0)] == [
        "starting",
        "running",
        "error",
    ]


def test_frame_retention_is_bounded() -> None:
    session = _session()
    for seq in range(webui._MAX_SESSION_FRAMES + 10):
        session.frames.append({"seq": seq, "event": {}, "snapshot": {}})
    assert len(session.frames) == webui._MAX_SESSION_FRAMES
    assert session.frames[0]["seq"] == 10


def test_terminal_sse_late_join_sends_frames_then_current_status() -> None:
    session = _session()
    session.frames.append({"seq": 1, "event": {}, "snapshot": {}})
    session.set_status("running")
    session.set_status("done")

    handler = object.__new__(webui._make_handler(session))
    handler.wfile = io.BytesIO()
    handler.send_response = lambda *args, **kwargs: None
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda *args, **kwargs: None
    handler._send_events(session, 0)
    body = handler.wfile.getvalue().decode("utf-8")
    assert [
        line.removeprefix("event: ")
        for line in body.splitlines()
        if line.startswith("event: ")
    ] == ["frame", "status"]
    status_payload = json.loads(
        next(
            line.removeprefix("data: ")
            for line in reversed(body.splitlines())
            if line.startswith("data: ")
        )
    )
    assert status_payload["status"] == "done"


def test_evaluation_stops_at_player_boundary_after_abort() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-eval-cancel-") as tempdir:
        run_dir = Path(tempdir)
        (run_dir / "logs").mkdir()
        (run_dir / "review").mkdir()
        (run_dir / "logs" / "game.json").write_text("{}\n", encoding="utf-8")
        cancel = threading.Event()
        reviewed: list[int] = []
        originals = (
            evaluate.load_mjai_log,
            evaluate.load_engine,
            evaluate.review_player,
        )

        def review_player(*args: Any, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            seat = int(args[1])
            reviewed.append(seat)
            cancel.set()
            return {"entries": [], "total_reviewed": 0, "total_matches": 0}

        evaluate.load_mjai_log = lambda path: []
        evaluate.load_engine = lambda weights: object()
        evaluate.review_player = review_player
        try:
            try:
                webui._evaluate_run(
                    run_dir,
                    arena.GameSummary(
                        seed=(1, 1),
                        names=["p0", "p1", "p2", "p3"],
                        scores=[25000] * 4,
                        placements={"p0": 1, "p1": 2, "p2": 3, "p3": 4},
                    ),
                    "unused",
                    cancel_event=cancel,
                )
            except GameAborted:
                pass
            else:
                raise AssertionError("evaluation ignored cancellation")
        finally:
            (
                evaluate.load_mjai_log,
                evaluate.load_engine,
                evaluate.review_player,
            ) = originals
        assert reviewed == [0]

    successful = _session()
    successful.set_status("running")
    successful.set_status("evaluating")
    successful.set_status("done")
    assert [status for _, status, _ in successful.status_transitions_after(0)] == [
        "starting",
        "running",
        "evaluating",
        "done",
    ]


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


def test_engine_names_carry_spec_reasoning_and_dedupe() -> None:
    from jongbench.providers import parse_spec

    parsed = [
        parse_spec("openai/gpt-5.2#high"),
        parse_spec("openai/gpt-5.2"),
        parse_spec("openai/gpt-5.2"),
        parse_spec("human"),
    ]
    assert webui._engine_names(parsed) == [
        "gpt-5.2-high",
        "gpt-5.2",
        "gpt-5.2-2",
        "human",
    ]


def test_error_diagnostic_file() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-error-") as tempdir:
        try:
            raise RuntimeError("deliberate failure")
        except RuntimeError as exc:
            webui._write_run_error(Path(tempdir), "test", exc)
        payload = json.loads(
            (Path(tempdir) / "error.json").read_text(encoding="utf-8")
        )
        assert payload["stage"] == "test"
        assert payload["type"] == "RuntimeError"
        assert payload["message"] == "deliberate failure"
        assert "RuntimeError: deliberate failure" in payload["traceback"]


class _Server:
    def __init__(self, session: webui.GameSession) -> None:
        self.session = session
        self.httpd, url = webui.serve_session(session, "127.0.0.1", 0)
        self.port = int(urlparse(url).port or 0)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def request_text(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, str]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
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

    def request_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        status, text = self.request_text(method, path, body)
        data = json.loads(text or "null")
        if not 200 <= status < 300:
            raise AssertionError((status, data))
        return data

    def wait_done(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.request_json("GET", "/api/state")
            if state["status"] == "error":
                raise AssertionError(state)
            if state["status"] == "done":
                return state
            time.sleep(0.2)
        raise AssertionError("run timed out")

    def collect_frames(self, since: int, min_frames: int, timeout: float) -> list[dict[str, Any]]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        conn.request("GET", f"/api/events?since={since}")
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


def test_replay_server_serves_the_bundle() -> None:
    bundle = {
        "game": "1_1",
        "seed": [1, 1],
        "names": ["p0", "p1", "p2", "p3"],
        "scores": [25000, 25000, 25000, 25000],
        "placements": {"p0": 1, "p1": 2, "p2": 3, "p3": 4},
        "frames": [{"seq": 1, "event": {"type": "start_game"}, "snapshot": {}}],
    }
    httpd, url = webui.serve_replay(bundle, "127.0.0.1", 0)
    port = int(urlparse(url).port or 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("GET", "/api/replay")
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == bundle
        conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("GET", "/api/state")
        assert conn.getresponse().status == 404
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_watch_server_workflow() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-webui-") as tempdir:
        session = webui.start_session(
            ["random"] * 4,
            seed=(4242, 1),
            runs_root=tempdir,
            weights="unused",
            no_eval=True,
            delay=0,
            state_hints=False,
        )
        server = _Server(session)
        try:
            status, text = server.request_text("GET", "/")
            assert status == 200
            assert "jongbench" in text

            state = server.wait_done(180)
            assert state["status"] == "done", state
            frames = server.collect_frames(0, 5, 30)
            assert len(frames) >= 5
            seqs = [frame["seq"] for frame in frames]
            assert seqs == sorted(seqs)
            assert len(set(seqs)) == len(seqs)
            for frame in frames:
                snapshot = frame.get("snapshot")
                assert isinstance(snapshot, dict)
                assert "seats" in snapshot or "hands" in snapshot

            run_dir = Path(session.run_dir)
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            assert config["state_hints"] is False
            partial_path = run_dir / "logs" / "partial.jsonl"
            partial_events = [
                json.loads(line)
                for line in partial_path.read_text(encoding="utf-8").splitlines()
            ]
            assert partial_events[0]["type"] == "start_game"
            assert any(event.get("type") == "start_kyoku" for event in partial_events)
            assert partial_events[-1]["type"] == "end_game"

            status, _ = server.request_text("GET", "/api/review")
            assert status == 404
        finally:
            server.close()

    print("OK")


def test_watch_server_human_seat_masks_and_answers() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-webui-human-") as tempdir:
        session = webui.start_session(
            ["human", "random", "random", "random"],
            seed=(555, 1),
            runs_root=tempdir,
            weights="unused",
            no_eval=True,
            delay=0,
        )
        server = _Server(session)
        try:
            answered = 0
            mask_checked = False
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                state = server.request_json("GET", "/api/state")
                if state["status"] == "error":
                    raise AssertionError(state)
                if state["status"] == "done":
                    break
                pending = server.request_json("GET", "/api/pending")
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
                        sample = server.collect_frames(0, 3, 10)
                        mask_checked = _assert_masked_frame(sample)
                    result = server.request_json(
                        "POST",
                        "/api/choose",
                        {"generation": pending["generation"], "choice": 0},
                    )
                    assert result == {"ok": True}, result
                    answered += 1
                else:
                    time.sleep(0.05)
            else:
                raise AssertionError("human game timed out")

            state = server.request_json("GET", "/api/state")
            assert state["status"] == "done", state
            assert answered >= 5, answered
            if not mask_checked:
                sample = server.collect_frames(0, 20, 20)
                mask_checked = _assert_masked_frame(sample)
            assert mask_checked
        finally:
            server.close()


def test_abort_unblocks_human_seat() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-webui-abort-") as tempdir:
        session = webui.start_session(
            ["human", "random", "random", "random"],
            seed=(777, 1),
            runs_root=tempdir,
            weights="unused",
            no_eval=True,
            delay=0,
        )
        server = _Server(session)
        try:
            deadline = time.monotonic() + 60
            while server.request_json("GET", "/api/pending") is None:
                assert time.monotonic() < deadline, "human decision never arrived"
                time.sleep(0.05)
            assert server.request_json("POST", "/api/abort") == {"ok": True}
            assert session.wait_terminal()[0] == "aborted"
            assert server.request_json("POST", "/api/abort") == {"ok": False}
        finally:
            server.close()


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
    test_error_diagnostic_file()
    test_watch_server_workflow()
