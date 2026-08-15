"""Local web spectator for one game.

`jongbench watch --ui web` runs one hanchan on a background thread and serves a
board view of it on localhost. One server, one session: configuration lives on the
CLI and credentials come from the environment, the same as every other run. A
`human` seat blocks the arena on `WebHumanIO` until the browser answers.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
import traceback
import webbrowser
from collections import deque
from datetime import datetime, timezone
from functools import cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import arena, prompts, providers
from .artifacts import decision_filename
from .engines import GameAborted, HumanIO, make_engine, sanitize_events
from .spectator import Spectator, TableState
from .weights import AUTO_MORTAL_WEIGHTS

_ACTIVE_STATUSES = {"starting", "running", "evaluating"}
_TERMINAL_STATUSES = {"done", "error", "aborted"}
_BODY_LIMIT = 64 * 1024
# A hanchan produces about 1,800 frames of ~7KB each (measured: 1,772). At 256 a
# reconnecting viewer only received the last hand. 4096 holds a whole match and still
# bounds an endless session.
_MAX_SESSION_FRAMES = 4096
_DRAW_TICKER_RE = re.compile(r"^P([0-3]) drew .+$")


class WebHumanIO(HumanIO):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._pending: dict[str, Any] | None = None
        self._event: threading.Event | None = None
        self._choice: int | None = None
        self._menu_len = 0
        self._cancelled = False

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            gate = self._event
            self._pending = None
        if gate is not None:
            gate.set()

    def ask(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self._lock:
            if self._cancelled:
                raise GameAborted("game aborted")
        gate = threading.Event()
        state_text = prompts.render_state(player_id, state, _prompt_safe_events(events))
        options = [
            _pending_option(index, player_id, item) for index, item in enumerate(menu)
        ]
        with self._lock:
            # Rendering is deliberately outside the lock, so cancellation must
            # be checked again while atomically registering the waiter.
            if self._cancelled:
                raise GameAborted("game aborted")
            self._generation += 1
            generation = self._generation
            self._event = gate
            self._choice = None
            self._menu_len = len(menu)
            self._pending = {
                "generation": generation,
                "seat": int(player_id),
                "state_text": state_text,
                # Keep the labels for older clients, but give the web UI the
                # raw tile and action metadata it needs for direct tile clicks.
                "menu": [option["label"] for option in options],
                "options": options,
            }

        gate.wait()
        with self._lock:
            if self._cancelled:
                raise GameAborted("game aborted")
            choice = self._choice
        if choice is None or not 0 <= choice < len(menu):
            raise RuntimeError("human decision was not selected")
        return menu[choice]["event"]

    def pending(self) -> dict[str, Any] | None:
        with self._lock:
            if self._pending is None:
                return None
            return copy.deepcopy(self._pending)

    def choose(self, generation: int, index: int) -> bool:
        try:
            generation = int(generation)
            index = int(index)
        except (TypeError, ValueError):
            return False
        with self._lock:
            if self._pending is None:
                return False
            if int(self._pending["generation"]) != generation:
                return False
            if not 0 <= index < self._menu_len:
                return False
            gate = self._event
            self._choice = index
            self._pending = None
        if gate is not None:
            gate.set()
        return True


def _pending_option(index: int, player_id: int, item: dict[str, Any]) -> dict[str, Any]:
    """Expose just enough menu metadata for the browser to render a legal choice."""
    event = item.get("event")
    if not isinstance(event, dict):
        event = {}
    event_type = str(event.get("type") or item.get("kind") or "none")
    if event_type == "dahai":
        action = "discard"
    elif event_type == "hora":
        action = "tsumo" if event.get("target") == player_id else "ron"
    elif event_type == "none":
        action = "pass"
    else:
        action = event_type

    option = {
        "choice": int(index),
        "action": action,
        "label": str(item.get("label", action)),
    }
    if action == "discard" and event.get("pai") is not None:
        option["tile"] = str(event["pai"])
    return option


class GameSession:
    def __init__(
        self,
        *,
        names: list[str],
        human_seat: int | None,
        human_io: WebHumanIO,
        run_dir: str,
    ) -> None:
        self.status = "starting"
        self.error: str | None = None
        self.names = list(names)
        self.human_seat = human_seat
        self.human_io = human_io
        self.frames: deque[dict[str, Any]] = deque(maxlen=_MAX_SESSION_FRAMES)
        self.frames_lock = threading.Lock()
        self.final: dict[str, Any] | None = None
        self.review: dict[str, Any] | None = None
        self.run_dir = run_dir
        self._status_lock = threading.Lock()
        self._status_version = 1
        self._status_history: list[tuple[int, str, str | None]] = [
            (self._status_version, self.status, self.error)
        ]
        self.cancel_event = threading.Event()
        self._game_done = threading.Event()
        self._poller_done = threading.Event()
        self._worker_done = threading.Event()

    def _record_status_locked(self) -> None:
        self._status_version += 1
        self._status_history.append((self._status_version, self.status, self.error))

    def set_status(self, status: str) -> bool:
        with self._status_lock:
            if self.status in _TERMINAL_STATUSES or self.status == status:
                return False
            self.status = status
            self._record_status_locked()
            return True

    def abort(self) -> bool:
        with self._status_lock:
            if self.status not in _ACTIVE_STATUSES:
                return False
            self.status = "aborted"
            self._record_status_locked()
        self.cancel_event.set()
        self.human_io.cancel()
        return True

    def set_error(self, message: str) -> None:
        with self._status_lock:
            if self.status in _TERMINAL_STATUSES:
                return
            self.error = message
            self.status = "error"
            self._record_status_locked()

    def status_snapshot(self) -> tuple[str, str | None]:
        with self._status_lock:
            return self.status, self.error

    def status_transitions_after(
        self, version: int
    ) -> list[tuple[int, str, str | None]]:
        with self._status_lock:
            return [item for item in self._status_history if item[0] > version]

    def latest_status_transition(self) -> tuple[int, str, str | None]:
        with self._status_lock:
            return self._status_history[-1]

    def latest_seq(self) -> int:
        with self.frames_lock:
            return int(self.frames[-1]["seq"]) if self.frames else 0

    def frames_after(self, seq: int) -> list[dict[str, Any]]:
        with self.frames_lock:
            # Frames are immutable after publication. Copy the small container,
            # not every large table snapshot in the SSE backlog.
            return [frame for frame in self.frames if int(frame["seq"]) > seq]

    def wait_terminal(self) -> tuple[str, str | None]:
        while self.status_snapshot()[0] not in _TERMINAL_STATUSES:
            time.sleep(0.2)
        self._worker_done.wait(timeout=10.0)
        return self.status_snapshot()


class _DecisionLogSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def __call__(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)


class _HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def start_session(
    model_specs: list[str],
    *,
    seed: tuple[int, int],
    runs_root: str,
    weights: str,
    no_eval: bool,
    delay: float,
    state_hints: bool = True,
    label: str = "watch",
) -> GameSession:
    if len(model_specs) != 4:
        raise ValueError("models must contain exactly 4 specs")
    parsed = [providers.parse_spec(spec) for spec in model_specs]
    human_seats = [idx for idx, spec in enumerate(parsed) if spec.provider == "human"]
    if len(human_seats) > 1:
        raise ValueError("only one human seat is supported")
    human_seat = human_seats[0] if human_seats else None

    names = _engine_names(parsed)
    run_dir = _make_run_dir(Path(runs_root), label)
    for directory in ("logs", "decisions", "review"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)

    human_io = WebHumanIO()
    spectator = Spectator(delay=delay, names=names)
    engines = [
        _make_engine(
            name=names[seat],
            spec_str=model_specs[seat],
            spec=parsed[seat],
            seat=seat,
            seed=seed,
            spectator=spectator,
            human_io=human_io,
            decisions_dir=run_dir / "decisions",
            state_hints=state_hints,
        )
        for seat in range(4)
    ]

    session = GameSession(
        names=names,
        human_seat=human_seat,
        human_io=human_io,
        run_dir=str(run_dir),
    )
    _write_config(run_dir, model_specs, session, seed, label, no_eval, state_hints)
    for engine in engines:
        engine.cancel_event = session.cancel_event

    threading.Thread(
        target=_poll_spectator,
        args=(session, spectator, seed),
        daemon=True,
        name="jongbench-web-poller",
    ).start()
    threading.Thread(
        target=_run_game_thread,
        args=(session, spectator, engines, seed, run_dir, weights, no_eval),
        daemon=True,
        name="jongbench-web-game",
    ).start()
    return session


def run_watch_server(
    model_specs: list[str],
    *,
    seed: int | tuple[int, int] = (10000, 1),
    host: str = "127.0.0.1",
    port: int = 8642,
    delay: float = 0.4,
    open_browser: bool = True,
    weights: str = AUTO_MORTAL_WEIGHTS,
    runs_root: str = "runs",
    no_eval: bool = False,
    state_hints: bool = True,
    label: str = "watch",
) -> str:
    if isinstance(seed, int):
        seed = (seed, 1)
    session = start_session(
        list(model_specs),
        seed=(int(seed[0]), int(seed[1])),
        runs_root=runs_root,
        weights=weights,
        no_eval=no_eval,
        delay=delay,
        state_hints=state_hints,
        label=label,
    )
    httpd, url = serve_session(session, host, port)
    thread = threading.Thread(
        target=httpd.serve_forever, daemon=True, name="jongbench-watch-web"
    )
    thread.start()
    print(f"watching at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        status, error = session.wait_terminal()
        if status == "error":
            raise RuntimeError(error or "web watch failed")
        return session.run_dir
    finally:
        httpd.shutdown()
        httpd.server_close()


def serve_session(
    session: GameSession, host: str = "127.0.0.1", port: int = 0
) -> tuple[ThreadingHTTPServer, str]:
    httpd = ThreadingHTTPServer((host, int(port)), _make_handler(session))
    httpd.daemon_threads = True
    return httpd, _server_url(httpd, host)


def _server_url(httpd: ThreadingHTTPServer, host: str) -> str:
    actual_port = int(httpd.server_address[1])
    url_host = "127.0.0.1" if host == "0.0.0.0" else host
    if ":" in url_host and not url_host.startswith("["):
        url_host = f"[{url_host}]"
    return f"http://{url_host}:{actual_port}/"


class _BaseHandler(BaseHTTPRequestHandler):
    server_version = "jongbench-webui/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_headers(
        self, status: int, content_type: str, length: int | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self._send_headers(status, content_type, len(body))
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self._send_bytes(status, body, "application/json; charset=utf-8")


def _make_replay_handler(bundle: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    body = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    class ReplayHandler(_BaseHandler):
        def do_GET(self) -> None:
            try:
                path = urlparse(self.path).path
                if path == "/":
                    self._send_bytes(
                        HTTPStatus.OK, _page_bytes(), "text/html; charset=utf-8"
                    )
                elif path == "/api/replay":
                    self._send_bytes(
                        HTTPStatus.OK, body, "application/json; charset=utf-8"
                    )
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ReplayHandler


def serve_replay(
    bundle: dict[str, Any], host: str = "127.0.0.1", port: int = 0
) -> tuple[ThreadingHTTPServer, str]:
    httpd = ThreadingHTTPServer((host, int(port)), _make_replay_handler(bundle))
    httpd.daemon_threads = True
    return httpd, _server_url(httpd, host)


def run_replay_server(
    bundle: dict[str, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 8642,
    open_browser: bool = True,
) -> None:
    httpd, url = serve_replay(bundle, host, port)
    print(f"replay at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def _make_handler(session: GameSession) -> type[BaseHTTPRequestHandler]:
    class Handler(_BaseHandler):
        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/":
                    self._send_bytes(
                        HTTPStatus.OK, _page_bytes(), "text/html; charset=utf-8"
                    )
                elif path == "/api/state":
                    self._send_json(HTTPStatus.OK, _session_state(session))
                elif path == "/api/events":
                    query = parse_qs(parsed.query)
                    since = _int_query(query.get("since", ["0"])[0], 0)
                    since = max(
                        since, _int_query(self.headers.get("Last-Event-ID", "0"), 0)
                    )
                    self._send_events(session, since)
                elif path == "/api/pending":
                    self._send_json(
                        HTTPStatus.OK,
                        session.human_io.pending()
                        if session.human_seat is not None
                        else None,
                    )
                elif path == "/api/review":
                    if session.review is None:
                        raise _HTTPError(
                            HTTPStatus.NOT_FOUND, "review is not available"
                        )
                    self._send_json(HTTPStatus.OK, session.review)
                else:
                    raise _HTTPError(HTTPStatus.NOT_FOUND, "not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except _HTTPError as exc:
                self._send_json(exc.status, {"error": exc.message})
            except Exception as exc:
                try:
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)}
                    )
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/abort":
                    self._send_json(HTTPStatus.OK, {"ok": session.abort()})
                elif parsed.path == "/api/choose":
                    payload = self._read_json_body()
                    ok = session.human_io.choose(
                        payload.get("generation"), payload.get("choice")
                    )
                    self._send_json(HTTPStatus.OK, {"ok": ok})
                else:
                    raise _HTTPError(HTTPStatus.NOT_FOUND, "not found")
            except _HTTPError as exc:
                self._send_json(exc.status, {"error": exc.message})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def _read_json_body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise _HTTPError(
                    HTTPStatus.BAD_REQUEST, "invalid Content-Length"
                ) from exc
            if length < 0:
                raise _HTTPError(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            if length > _BODY_LIMIT:
                raise _HTTPError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large"
                )
            raw = self.rfile.read(length)
            try:
                loaded = json.loads(raw.decode("utf-8") if raw else "{}")
            except json.JSONDecodeError as exc:
                raise _HTTPError(HTTPStatus.BAD_REQUEST, "invalid JSON body") from exc
            if not isinstance(loaded, dict):
                raise _HTTPError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
            return loaded

        def _send_events(self, session: GameSession, since: int) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_ping = time.monotonic()
            status_version, initial_status, initial_error = (
                session.latest_status_transition()
            )
            try:
                # An active late joiner needs the current status before frames,
                # not historical statuses that would regress the UI. A terminal
                # late joiner receives its retained frames before the terminal
                # status closes the stream.
                if initial_status not in _TERMINAL_STATUSES:
                    self._write_sse(
                        "status",
                        _session_state(
                            session, status=initial_status, error=initial_error
                        ),
                    )
                while True:
                    for frame in session.frames_after(since):
                        since = max(since, int(frame["seq"]))
                        self._write_sse("frame", frame, event_id=since)
                    if initial_status in _TERMINAL_STATUSES:
                        self._write_sse(
                            "status",
                            _session_state(
                                session,
                                status=initial_status,
                                error=initial_error,
                            ),
                        )
                        return
                    terminal = False
                    for version, status, error in session.status_transitions_after(
                        status_version
                    ):
                        status_version = version
                        self._write_sse(
                            "status",
                            _session_state(session, status=status, error=error),
                        )
                        terminal = status in _TERMINAL_STATUSES
                    if terminal:
                        return
                    now = time.monotonic()
                    if now - last_ping >= 15.0:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = now
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _write_sse(
            self, event: str, payload: Any, *, event_id: int | None = None
        ) -> None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self.wfile.write(f"event: {event}\n".encode("utf-8"))
            if event_id is not None:
                self.wfile.write(f"id: {event_id}\n".encode("utf-8"))
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler


def _run_game_thread(
    session: GameSession,
    spectator: Spectator,
    engines: list[Any],
    seed: tuple[int, int],
    run_dir: Path,
    weights: str,
    no_eval: bool,
) -> None:
    try:
        session.set_status("running")
        summaries = arena.run_games(
            engines, 1, seed_start=seed, log_dir=str(run_dir / "logs")
        )
        if not summaries:
            raise RuntimeError("arena returned no game summaries")
        summary = summaries[0]
        session.names = list(summary.names)
        session.final = {
            "names": list(summary.names),
            "scores": list(summary.scores),
            "placements": dict(summary.placements),
        }
        spectator.finish(summary.names, summary.scores)
        session._game_done.set()
        session._poller_done.wait(timeout=10.0)
        if session.cancel_event.is_set() or session.status_snapshot()[0] in {
            "error",
            "aborted",
        }:
            return
        if no_eval:
            session.set_status("done")
            return
        session.set_status("evaluating")
        session.review = _evaluate_run(
            run_dir,
            summary,
            weights,
            cancel_event=session.cancel_event,
        )
        session.set_status("done")
    except GameAborted:
        pass
    except Exception as exc:
        if session.cancel_event.is_set():
            return
        _write_run_error(run_dir, "game", exc)
        session.set_error(str(exc))
    finally:
        session._game_done.set()
        session._poller_done.wait(timeout=10.0)
        session._worker_done.set()


def _poll_spectator(
    session: GameSession,
    spectator: Spectator,
    seed: tuple[int, int],
) -> None:
    table = TableState()
    table.names = list(session.names)
    partial_log = _DecisionLogSink(Path(session.run_dir) / "logs" / "partial.jsonl")
    partial_log(
        {
            "type": "start_game",
            "names": list(session.names),
            "seed": [int(seed[0]), int(seed[1])],
        }
    )
    seq = 0
    try:
        while True:
            updates = spectator.events_since(seq, copy_events=False)
            if updates:
                for item in updates:
                    seq = max(seq, int(item["seq"]))
                    event = item["event"]
                    if event.get("type") == "finish":
                        partial_log({"type": "end_game"})
                        table.finish(event.get("names"), event.get("scores"))
                    else:
                        partial_log(event)
                        table.apply(event)
                    snapshot = table.snapshot()
                    public_event = event
                    if session.human_seat is not None:
                        snapshot = _mask_snapshot_in_place(snapshot, session.human_seat)
                        public_event = _prompt_safe_events(
                            sanitize_events([event], session.human_seat)
                        )[0]
                    frame = {
                        "seq": seq,
                        "event": public_event,
                        "snapshot": snapshot,
                    }
                    with session.frames_lock:
                        session.frames.append(frame)
                continue
            if session._game_done.is_set():
                if not spectator.events_since(seq):
                    return
            time.sleep(0.15)
    except Exception as exc:
        _write_run_error(Path(session.run_dir), "spectator", exc)
        session.cancel_event.set()
        session.human_io.cancel()
        session.set_error(str(exc))
    finally:
        session._poller_done.set()


def _write_run_error(run_dir: Path, stage: str, exc: Exception) -> None:
    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    try:
        (run_dir / "error.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        pass


def _mask_snapshot(snapshot: dict[str, Any], human_seat: int) -> dict[str, Any]:
    return _mask_snapshot_in_place(copy.deepcopy(snapshot), human_seat)


def _mask_snapshot_in_place(masked: dict[str, Any], human_seat: int) -> dict[str, Any]:
    ticker = masked.get("ticker")
    if isinstance(ticker, list):
        for index, line in enumerate(ticker):
            if not isinstance(line, str):
                continue
            match = _DRAW_TICKER_RE.fullmatch(line)
            if match is not None and int(match.group(1)) != human_seat:
                ticker[index] = f"P{match.group(1)} drew a tile"
    for seat_info in masked.get("seats", []):
        if not isinstance(seat_info, dict):
            continue
        if int(seat_info.get("seat", -1)) == human_seat:
            continue
        hand = seat_info.get("hand")
        if isinstance(hand, list):
            seat_info["hand"] = ["?"] * len(hand)
    return masked


def _evaluate_run(
    run_dir: Path,
    summary: arena.GameSummary,
    weights: str,
    *,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    from . import evaluate

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise GameAborted("game aborted")

    check_cancelled()
    logs = sorted((run_dir / "logs").glob("*.json.gz")) + sorted(
        (run_dir / "logs").glob("*.json")
    )
    if not logs:
        raise RuntimeError("no mjai log was written")
    events = evaluate.load_mjai_log(str(logs[0]))
    check_cancelled()
    mortal = evaluate.load_engine(weights)
    check_cancelled()
    reviews = evaluate.review_game(events, mortal, check_cancelled=check_cancelled)
    check_cancelled()
    players: dict[str, Any] = {}
    for seat in range(4):
        review = reviews[seat]
        players[str(seat)] = {
            "name": summary.names[seat],
            "review": review,
            "aggregates": evaluate.aggregates(review),
        }
    check_cancelled()
    data = {
        "seed": [int(summary.seed[0]), int(summary.seed[1])],
        "names": list(summary.names),
        "scores": list(summary.scores),
        "placements": dict(summary.placements),
        "players": players,
    }
    path = run_dir / "review" / f"{summary.seed[0]}_{summary.seed[1]}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    response = dict(data)
    response["run_dir"] = str(run_dir)
    return response


def _make_engine(
    *,
    name: str,
    spec_str: str,
    spec: providers.ProviderSpec,
    seat: int,
    seed: tuple[int, int],
    spectator: Spectator,
    human_io: WebHumanIO,
    decisions_dir: Path,
    state_hints: bool,
) -> Any:
    if spec.provider == "human":
        return make_engine(
            name, spec_str, human_io=human_io, spectator=spectator, concurrency=1
        )
    kwargs: dict[str, Any] = {
        "spectator": spectator,
        "state_hints": state_hints,
    }
    if spec.provider == "random":
        kwargs["seed"] = seed[0] + seat * 1009 + seed[1] * 97
    else:
        kwargs["decision_log"] = _DecisionLogSink(
            decisions_dir / decision_filename(name)
        )
    return make_engine(name, spec_str, **kwargs)


def _prompt_safe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = copy.deepcopy(events)
    for event in safe:
        if event.get("type") == "tsumo" and event.get("pai") == "?":
            event.pop("pai", None)
    return safe


def _engine_names(parsed: list[providers.ProviderSpec]) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for seat, spec in enumerate(parsed):
        base = _safe_name(spec.display_name or f"P{seat}")
        count = seen.get(base, 0) + 1
        seen[base] = count
        names.append(base if count == 1 else f"{base}-{count}")
    return names


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip()).strip("._-")
    return (cleaned or "player")[:60]


def _make_run_dir(root: Path, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _safe_name(label) if label.strip() else "watch"
    name = f"{stamp}-{slug}"
    path = root / name
    suffix = 2
    while path.exists():
        path = root / f"{name}-{suffix}"
        suffix += 1
    return path


def _write_config(
    run_dir: Path,
    model_specs: list[str],
    session: GameSession,
    seed: tuple[int, int],
    label: str,
    no_eval: bool,
    state_hints: bool,
) -> None:
    config = {
        "label": label or run_dir.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "models": list(model_specs),
        "names": list(session.names),
        "games": 1,
        "seed_start": [int(seed[0]), int(seed[1])],
        "human_seat": session.human_seat,
        "no_eval": bool(no_eval),
        "state_hints": bool(state_hints),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _session_state(
    session: GameSession,
    *,
    status: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if status is None:
        status, current_error = session.status_snapshot()
        error = current_error
    return {
        "status": status,
        "error": error,
        "names": list(session.names),
        "human_seat": session.human_seat,
        "latest_seq": session.latest_seq(),
        "final": copy.deepcopy(session.final),
    }


@cache
def _page_bytes() -> bytes:
    page = Path(__file__).parent / "webui_page.html"
    if not page.exists():
        raise FileNotFoundError(
            "jongbench/webui_page.html is missing; build it with `cd webui && bun install && bun run build`"
        )
    return page.read_bytes()


def _int_query(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
