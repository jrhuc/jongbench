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
from jongbench.providers import ProviderSpec  # noqa: E402


HOST = "127.0.0.1"
PORT = 0


def _session(run_id: str = "test") -> webui.GameSession:
    return webui.GameSession(
        run_id=run_id,
        created=time.time(),
        model_specs=["random"] * 4,
        names=["p0", "p1", "p2", "p3"],
        human_seat=None,
        human_io=webui.WebHumanIO(),
        run_dir="/tmp/unused-jongbench-session",
        state_hints=True,
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


def test_web_starts_reject_custom_endpoints_and_require_request_keys() -> None:
    try:
        webui._validate_web_start_credentials(
            ["compat:http://127.0.0.1:9999:victim"] + ["random"] * 3,
            {"compat": "visitor-key"},
        )
    except ValueError as exc:
        assert "compat" in str(exc)
    else:
        raise AssertionError("web start accepted an arbitrary compatibility URL")

    for provider, spec in (
        ("openai", "openai:test"),
        ("anthropic", "anthropic:test"),
        ("google", "google:test"),
        ("xai", "xai:test"),
        ("deepseek", "deepseek:test"),
    ):
        try:
            webui._validate_web_start_credentials([spec] + ["random"] * 3, {})
        except ValueError as exc:
            assert provider in str(exc)
        else:
            raise AssertionError(f"web start accepted missing {provider} request key")
        webui._validate_web_start_credentials(
            [spec] + ["random"] * 3, {provider: "visitor-key"}
        )


def test_preflight_distinguishes_compatibility_base_urls() -> None:
    calls: list[str] = []

    class FakeProvider:
        def __init__(self, url: str) -> None:
            self.url = url

        def complete(self, *args: Any, **kwargs: Any) -> tuple[str, dict[str, int]]:
            del args, kwargs
            calls.append(self.url)
            return "OK", {}

    class FakeEngine:
        def __init__(self, url: str) -> None:
            self.name = url
            self.temperature = 0.0
            self.spec = ProviderSpec("compat", "same-model", url)
            self.provider = FakeProvider(url)

    urls = ["https://one.invalid/v1", "https://two.invalid/v1"]
    webui._preflight_engines([FakeEngine(url) for url in urls])
    assert sorted(calls) == urls


def test_preflight_distinguishes_reasoning_levels() -> None:
    calls: list[str] = []

    class FakeProvider:
        def __init__(self, reasoning: str) -> None:
            self.reasoning = reasoning

        def complete(self, *args: Any, **kwargs: Any) -> tuple[str, dict[str, int]]:
            del args, kwargs
            calls.append(self.reasoning)
            return "OK", {}

    class FakeEngine:
        def __init__(self, reasoning: str) -> None:
            self.name = reasoning
            self.temperature = 0.0
            self.reasoning = reasoning
            self.max_tokens = 8
            self.spec = ProviderSpec("openai", "gpt-5.2")
            self.provider = FakeProvider(reasoning)

    webui._preflight_engines(
        [FakeEngine("medium"), FakeEngine("medium"), FakeEngine("high")]
    )
    assert sorted(calls) == ["high", "medium"]


def test_web_reasoning_validation() -> None:
    invalid = (
        (["medium"] * 3, "list of 4"),
        (["turbo", None, None, None], "supported levels"),
        (["high", None, None, None], "not supported for random"),
    )
    for reasoning, message in invalid:
        try:
            webui.start_game_session(
                ["random"] * 4,
                {},
                1,
                None,
                None,
                runs_root="/tmp/unused-jongbench-test",
                weights="unused",
                no_eval=True,
                delay=0,
                reasoning=reasoning,
            )
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"accepted invalid reasoning: {reasoning!r}")

    try:
        webui.start_game_session(
            ["openai:gpt-5.1", "random", "random", "random"],
            {},
            1,
            None,
            None,
            runs_root="/tmp/unused-jongbench-test",
            weights="unused",
            no_eval=True,
            delay=0,
            reasoning=["xhigh", None, None, None],
        )
    except ValueError as exc:
        assert "available: off, low, medium, high" in str(exc)
    else:
        raise AssertionError("accepted an unsupported model reasoning level")


def test_reasoning_propagates_to_names_and_config() -> None:
    parsed = [
        webui._parse_spec("openai:gpt-5.2"),
        webui._parse_spec("openai:gpt-5.2"),
        webui._parse_spec("random"),
        webui._parse_spec("human"),
    ]
    reasoning = ["high", "off", None, None]
    names = webui._engine_names(parsed, reasoning)
    assert names == ["gpt-5.2-high", "gpt-5.2-off", "random", "human"]

    with tempfile.TemporaryDirectory(prefix="jongbench-reasoning-") as tempdir:
        run_dir = Path(tempdir)
        session = webui.GameSession(
            run_id="reasoning",
            created=time.time(),
            model_specs=["openai:gpt-5.2"] * 2 + ["random", "human"],
            names=names,
            human_seat=3,
            human_io=webui.WebHumanIO(),
            run_dir=tempdir,
            state_hints=True,
        )
        webui._write_config(
            run_dir,
            session,
            (1, 2),
            None,
            0,
            True,
            reasoning,
        )
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert config["reasoning"] == reasoning
        assert config["names"] == names


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


def test_terminal_sse_late_join_sends_frames_then_current_status() -> None:
    state = webui._ServerState(
        runs_root="/tmp/unused-jongbench-runs",
        weights="unused",
        max_concurrent_games=1,
        no_eval=True,
        demo_path=None,
        delay=0,
    )
    session = _session("late")
    session.frames.append({"seq": 1, "event": {}, "snapshot": {}})
    session.set_status("running")
    session.set_status("done")
    session._worker_done.set()
    with state.lock:
        state.sessions[session.id] = session

    handler = object.__new__(webui._make_handler(state))
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

    successful = _session("successful")
    successful.set_status("running")
    successful.set_status("evaluating")
    successful.set_status("done")
    assert [status for _, status, _ in successful.status_transitions_after(0)] == [
        "starting",
        "running",
        "evaluating",
        "done",
    ]


def test_abort_holds_capacity_until_worker_exit_then_cleans_session() -> None:
    state = webui._ServerState(
        runs_root="/tmp/unused-jongbench-runs",
        weights="unused",
        max_concurrent_games=1,
        no_eval=True,
        demo_path=None,
        delay=0,
    )
    session = _session("blocked")
    session.set_status("running")
    release_worker = threading.Event()

    def worker() -> None:
        release_worker.wait()
        session._worker_done.set()

    session.game_thread = threading.Thread(target=worker)
    session.game_thread.start()
    state.add_reserved(session)
    assert session.abort()
    assert state.get_session(session.id) is session
    assert not state.reserve_start()

    release_worker.set()
    session.game_thread.join(5)
    deadline = time.monotonic() + 5
    while state.get_session(session.id) is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert state.get_session(session.id) is None
    assert state.reserve_start()
    state.release_reserved()


def test_session_and_frame_retention_are_bounded() -> None:
    session = _session("frames")
    for seq in range(webui._MAX_SESSION_FRAMES + 10):
        session.frames.append({"seq": seq, "event": {}, "snapshot": {}})
    assert len(session.frames) == webui._MAX_SESSION_FRAMES
    assert session.frames[0]["seq"] == 10

    state = webui._ServerState(
        runs_root="/tmp/unused-jongbench-runs",
        weights="unused",
        max_concurrent_games=1,
        no_eval=True,
        demo_path=None,
        delay=0,
    )
    sessions = []
    for index in range(webui._MAX_RETAINED_FINISHED_SESSIONS + 2):
        finished = _session(f"done-{index}")
        finished.set_status("done")
        finished._worker_done.set()
        assert finished.finished_at is not None
        finished.finished_at += index
        sessions.append(finished)
    expired = _session("expired")
    expired.set_status("done")
    expired._worker_done.set()
    expired.finished_at = time.time() - webui._FINISHED_SESSION_TTL - 1
    with state.lock:
        state.sessions = {item.id: item for item in [*sessions, expired]}
    retained = state.list_sessions()
    assert len(retained) == webui._MAX_RETAINED_FINISHED_SESSIONS
    assert expired not in retained
    assert sessions[-1] in retained


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

    try:
        webui.start_game_session(
            ["random"] * 4,
            {},
            1,
            None,
            None,
            runs_root="/tmp/unused-jongbench-test",
            weights="unused",
            no_eval=True,
            delay=0,
            state_hints="yes",  # type: ignore[arg-type]
        )
    except ValueError as exc:
        assert "state_hints" in str(exc)
    else:
        raise AssertionError("accepted non-boolean state_hints")


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
            {
                "models": ["random", "random", "random", "random"],
                "keys": {},
                "seed": 4242,
                "state_hints": False,
            },
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

        first_run = next(path for path in Path(tempdir).iterdir() if path.is_dir())
        config = json.loads((first_run / "config.json").read_text(encoding="utf-8"))
        assert config["state_hints"] is False
        partial_path = first_run / "logs" / "partial.jsonl"
        partial_events = [
            json.loads(line)
            for line in partial_path.read_text(encoding="utf-8").splitlines()
        ]
        assert partial_events[0]["type"] == "start_game"
        assert any(event.get("type") == "start_kyoku" for event in partial_events)
        assert partial_events[-1]["type"] == "end_game"

        human = _request_json(
            "POST",
            "/api/start",
            {
                "models": ["human", "random", "random", "random"],
                "keys": {},
                "seed": 555,
                "human_seat": 0,
                "state_hints": True,
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
    deadline = time.monotonic() + 10
    while True:
        status, text = _request_text("GET", f"/api/state/{started[0]}")
        if status == 404:
            break
        assert status == 200 and json.loads(text)["status"] == "aborted", (status, text)
        if time.monotonic() >= deadline:
            raise AssertionError("aborted session worker did not exit")
        time.sleep(0.02)
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
    test_error_diagnostic_file()
    test_web_server_workflow()
