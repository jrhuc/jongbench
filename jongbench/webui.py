from __future__ import annotations

import copy
from collections import defaultdict, deque
import hashlib
import ipaddress
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import threading
import time
import traceback
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

from . import arena, prompts, providers
from .artifacts import decision_filename
from .engines import BaseEngine, GameAborted, make_engine, sanitize_events
from .spectator import Spectator, TableState

try:
    from .engines import HumanEngine as _NativeHumanEngine
    from .engines import HumanIO
except ImportError:
    _NativeHumanEngine = None

    class HumanIO:
        def ask(
            self,
            player_id: int,
            state: Any,
            events: list[dict[str, Any]],
            menu: list[dict[str, Any]],
        ) -> dict[str, Any]:
            raise NotImplementedError


_ACTIVE_STATUSES = {"starting", "running", "evaluating"}
_TERMINAL_STATUSES = {"done", "error", "aborted"}
_BODY_LIMIT = 64 * 1024
_MAX_SESSION_FRAMES = 256
_MAX_RETAINED_FINISHED_SESSIONS = 4
_FINISHED_SESSION_TTL = 30 * 60.0
_DRAW_TICKER_RE = re.compile(r"^P([0-3]) drew .+$")
_LAST_SERVER_LOCK = threading.Lock()
_LAST_SERVER_URL: str | None = None
_LAST_SERVER_PORT: int | None = None
_MODEL_CACHE_TTL = 300.0
_MODEL_CACHE_MAX = 32
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]] = {}


def _prune_model_cache_locked(now: float) -> None:
    expired = [
        entry_key
        for entry_key, (stamp, _) in _MODEL_CACHE.items()
        if now - stamp >= _MODEL_CACHE_TTL
    ]
    for entry_key in expired:
        del _MODEL_CACHE[entry_key]


def _cached_list_models(key: str, base_url: str | None = None) -> list[dict[str, Any]]:
    cache_key = (
        base_url or "",
        hashlib.sha256(key.encode("utf-8")).hexdigest(),
    )
    with _MODEL_CACHE_LOCK:
        _prune_model_cache_locked(time.monotonic())
        hit = _MODEL_CACHE.get(cache_key)
        if hit is not None:
            return hit[1]
    models = providers.list_models(key, base_url)
    with _MODEL_CACHE_LOCK:
        now = time.monotonic()
        _prune_model_cache_locked(now)
        while len(_MODEL_CACHE) >= _MODEL_CACHE_MAX and cache_key not in _MODEL_CACHE:
            oldest = min(_MODEL_CACHE.items(), key=lambda item: item[1][0])[0]
            del _MODEL_CACHE[oldest]
        _MODEL_CACHE[cache_key] = (now, models)
    return models


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
            _pending_option(index, player_id, item)
            for index, item in enumerate(menu)
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


class _FallbackHumanEngine(BaseEngine):
    auto_choose_single_menu = False

    def __init__(
        self,
        name: str,
        io: HumanIO,
        spectator: Any | None = None,
        concurrency: int = 1,
        **_: Any,
    ) -> None:
        super().__init__(name, spectator=spectator, concurrency=concurrency)
        self.io = io

    def decide(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.io.ask(player_id, state, events, menu)


class GameSession:
    def __init__(
        self,
        *,
        run_id: str,
        created: float,
        model_specs: list[str],
        names: list[str],
        human_seat: int | None,
        human_io: WebHumanIO,
        run_dir: str,
        state_hints: bool,
    ) -> None:
        self.id = run_id
        self.created = created
        self.status = "starting"
        self.error: str | None = None
        self.model_specs = list(model_specs)
        self.names = list(names)
        self.human_seat = human_seat
        self.human_io = human_io
        self.frames: deque[dict[str, Any]] = deque(maxlen=_MAX_SESSION_FRAMES)
        self.frames_lock = threading.Lock()
        self.final: dict[str, Any] | None = None
        self.review: dict[str, Any] | None = None
        self.run_dir = run_dir
        self.state_hints = bool(state_hints)
        self._status_lock = threading.Lock()
        self._status_version = 1
        self._status_history: list[tuple[int, str, str | None]] = [
            (self._status_version, self.status, self.error)
        ]
        self.finished_at: float | None = None
        self.cancel_event = threading.Event()
        self._game_done = threading.Event()
        self._poller_done = threading.Event()
        self._worker_done = threading.Event()
        self.game_thread: threading.Thread | None = None
        self.poller_thread: threading.Thread | None = None

    def _record_status_locked(self) -> None:
        self._status_version += 1
        self._status_history.append((self._status_version, self.status, self.error))
        if self.status in _TERMINAL_STATUSES and self.finished_at is None:
            self.finished_at = time.time()

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

    def occupies_slot(self) -> bool:
        return not self._worker_done.is_set()

    def latest_seq(self) -> int:
        with self.frames_lock:
            return int(self.frames[-1]["seq"]) if self.frames else 0

    def frames_after(self, seq: int) -> list[dict[str, Any]]:
        with self.frames_lock:
            # Frames are immutable after publication. Copy the small container,
            # not every large table snapshot in the SSE backlog.
            return [frame for frame in self.frames if int(frame["seq"]) > seq]


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


class _ServerState:
    def __init__(
        self,
        *,
        runs_root: str,
        weights: str,
        max_concurrent_games: int,
        no_eval: bool,
        demo_path: str | None,
        delay: float,
        allow_local_endpoints: bool = False,
    ) -> None:
        self.runs_root = runs_root
        self.weights = weights
        self.max_concurrent_games = max(1, int(max_concurrent_games))
        self.no_eval = bool(no_eval)
        self.demo_path = demo_path
        self.delay = float(delay)
        self.allow_local_endpoints = allow_local_endpoints
        self.sessions: dict[str, GameSession] = {}
        self.lock = threading.Lock()
        self._reserved = 0
        self._starts_by_ip: defaultdict[str, deque[float]] = defaultdict(deque)

    def reserve_start(self) -> bool:
        with self.lock:
            self._prune_sessions_locked(time.time())
            active = sum(
                1
                for session in self.sessions.values()
                if session.occupies_slot()
            )
            if active + self._reserved >= self.max_concurrent_games:
                return False
            self._reserved += 1
            return True

    def add_reserved(self, session: GameSession) -> None:
        with self.lock:
            self._reserved = max(0, self._reserved - 1)
            self.sessions[session.id] = session
            self._prune_sessions_locked(time.time())
        thread = threading.Thread(
            target=self._after_worker_exit,
            args=(session,),
            daemon=True,
            name=f"jongbench-web-cleanup-{session.id}",
        )
        thread.start()

    def release_reserved(self) -> None:
        with self.lock:
            self._reserved = max(0, self._reserved - 1)

    def get_session(self, run_id: str) -> GameSession | None:
        with self.lock:
            self._prune_sessions_locked(time.time())
            return self.sessions.get(run_id)

    def list_sessions(self) -> list[GameSession]:
        with self.lock:
            self._prune_sessions_locked(time.time())
            return sorted(self.sessions.values(), key=lambda session: session.created)

    def _after_worker_exit(self, session: GameSession) -> None:
        session._worker_done.wait()
        with self.lock:
            if self.sessions.get(session.id) is not session:
                return
            if session.status_snapshot()[0] == "aborted":
                self.sessions.pop(session.id, None)
            else:
                self._prune_sessions_locked(time.time())

    def _prune_sessions_locked(self, now: float) -> None:
        finished = [
            session
            for session in self.sessions.values()
            if session.status_snapshot()[0] in {"done", "error"}
            and not session.occupies_slot()
        ]
        expired = {
            session.id
            for session in finished
            if session.finished_at is not None
            and session.finished_at <= now - _FINISHED_SESSION_TTL
        }
        retained = sorted(
            (session for session in finished if session.id not in expired),
            key=lambda session: session.finished_at or session.created,
            reverse=True,
        )
        expired.update(
            session.id for session in retained[_MAX_RETAINED_FINISHED_SESSIONS:]
        )
        for run_id in expired:
            self.sessions.pop(run_id, None)

    def allow_start(self, ip: str) -> bool:
        now = time.time()
        with self.lock:
            starts = self._starts_by_ip[ip]
            while starts and starts[0] <= now - 60.0:
                starts.popleft()
            if len(starts) >= 3:
                return False
            starts.append(now)
            return True


class _HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def start_game_session(
    model_specs: list[str],
    keys: dict[str, str],
    seed: int | tuple[int, int] | list[int] | None,
    human_seat: int | None,
    label: str | None,
    *,
    runs_root: str,
    weights: str,
    no_eval: bool,
    delay: float,
    state_hints: bool = True,
    reasoning: list[str | None] | None = None,
) -> GameSession:
    if len(model_specs) != 4:
        raise ValueError("models must contain exactly 4 specs")
    parsed = [_parse_spec(spec) for spec in model_specs]
    human_seats = [idx for idx, spec in enumerate(parsed) if spec.provider == "human"]
    if len(human_seats) > 1:
        raise ValueError("only one human seat is supported")
    if human_seat is not None:
        human_seat = int(human_seat)
        if not 0 <= human_seat < 4:
            raise ValueError("human_seat must be 0, 1, 2, 3, or null")
    if human_seats:
        if human_seat is None:
            human_seat = human_seats[0]
        elif human_seat != human_seats[0]:
            raise ValueError("human_seat must match the human model seat")
    elif human_seat is not None:
        raise ValueError("human_seat was provided but no model spec is human")
    if not isinstance(state_hints, bool):
        raise ValueError("state_hints must be a boolean")
    if reasoning is not None:
        if (
            not isinstance(reasoning, list)
            or len(reasoning) != 4
            or not all(
                item is None
                or (isinstance(item, str) and item in providers.REASONING_LEVELS)
                for item in reasoning
            )
        ):
            raise ValueError(
                "reasoning must be a list of 4 supported levels or nulls"
            )
        # Per-model effort support is OpenRouter's to negotiate: it clamps a level a
        # model cannot honour, so only the seat kind is checked here.
        for idx, level in enumerate(reasoning):
            if level is None:
                continue
            provider = parsed[idx].provider
            if provider in {"random", "human"}:
                raise ValueError(f"reasoning is not supported for {provider}")
    reasoning_levels = reasoning if reasoning is not None else [None, None, None, None]

    seed_tuple = _normalize_seed(seed)
    run_id = secrets.token_hex(4)
    names = _engine_names(parsed, reasoning_levels)
    run_dir = _make_run_dir(Path(runs_root), label, run_id)

    human_io = WebHumanIO()
    spectator = Spectator(delay=delay, names=names)
    engines = [
        _make_engine_for_session(
            name=names[seat],
            spec_str=model_specs[seat],
            spec=parsed[seat],
            seat=seat,
            seed=seed_tuple,
            keys=keys,
            spectator=spectator,
            human_io=human_io,
            decisions_dir=run_dir / "decisions",
            state_hints=state_hints,
            reasoning=reasoning_levels[seat],
        )
        for seat in range(4)
    ]
    _preflight_engines(engines)
    for directory in ("logs", "decisions", "review"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)

    session = GameSession(
        run_id=run_id,
        created=time.time(),
        model_specs=list(model_specs),
        names=names,
        human_seat=human_seat,
        human_io=human_io,
        run_dir=str(run_dir),
        state_hints=state_hints,
    )
    _write_config(run_dir, session, seed_tuple, label, delay, no_eval, reasoning)
    for engine in engines:
        engine.cancel_event = session.cancel_event

    session.poller_thread = threading.Thread(
        target=_poll_spectator,
        args=(session, spectator, seed_tuple),
        daemon=True,
        name=f"jongbench-web-poller-{run_id}",
    )
    session.poller_thread.start()
    session.game_thread = threading.Thread(
        target=_run_game_thread,
        args=(session, spectator, engines, seed_tuple, run_dir, weights, no_eval),
        daemon=True,
        name=f"jongbench-web-game-{run_id}",
    )
    session.game_thread.start()
    return session


def _preflight_engines(engines: list[Any]) -> None:
    """One tiny completion per distinct provider/model so a missing or invalid
    API key fails the start request instead of silently degrading every
    decision to the fallback action."""
    checks: dict[tuple[str, str, str | None, str | None], Any] = {}
    for engine in engines:
        provider = getattr(engine, "provider", None)
        spec = getattr(engine, "spec", None)
        if provider is None or spec is None:
            continue
        checks.setdefault(
            (
                spec.provider,
                spec.model,
                spec.base_url,
                getattr(engine, "reasoning", None),
            ),
            engine,
        )
    if not checks:
        return

    def check(engine: Any) -> None:
        try:
            engine.provider.complete(
                [
                    {"role": "system", "content": "Reply with the word OK."},
                    {"role": "user", "content": "OK?"},
                ],
                max_tokens=getattr(engine, "max_tokens", 8),
                temperature=engine.temperature,
            )
        except Exception as exc:
            reason = str(exc).splitlines()[0][:240] if str(exc) else type(exc).__name__
            raise ValueError(f"API check failed for {engine.name}: {reason}") from exc

    with ThreadPoolExecutor(max_workers=len(checks)) as executor:
        for future in [executor.submit(check, engine) for engine in checks.values()]:
            future.result()


def run_server(
    *,
    host: str = "0.0.0.0",
    port: int = 8642,
    weights: str = "weights/mortal.pth",
    runs_root: str = "runs",
    demo_path: str | None = None,
    max_concurrent_games: int = 1,
    no_eval: bool = False,
) -> None:
    state = _ServerState(
        runs_root=runs_root,
        weights=weights,
        max_concurrent_games=max_concurrent_games,
        no_eval=no_eval,
        demo_path=demo_path,
        delay=0.0,
        allow_local_endpoints=_is_loopback_host(host),
    )
    httpd, url = _make_server(state, host, port)
    print(f"jongbench web UI listening on {url}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def run_watch_server(
    model_specs: list[str],
    *,
    seed: int | tuple[int, int] | list[int] = (10000, 1),
    host: str = "127.0.0.1",
    port: int = 8642,
    delay: float = 0.4,
    open_browser: bool = True,
    weights: str = "weights/mortal.pth",
    runs_root: str = "runs",
    api_keys: dict[str, str] | None = None,
    no_eval: bool = False,
    state_hints: bool = True,
) -> str:
    state = _ServerState(
        runs_root=runs_root,
        weights=weights,
        max_concurrent_games=1,
        no_eval=no_eval,
        demo_path=None,
        delay=delay,
        allow_local_endpoints=_is_loopback_host(host),
    )
    httpd, url = _make_server(state, host, port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="jongbench-watch-web")
    thread.start()
    try:
        session = start_game_session(
            list(model_specs),
            dict(api_keys or {}),
            seed,
            _human_seat_from_specs(list(model_specs)),
            None,
            runs_root=runs_root,
            weights=weights,
            no_eval=no_eval,
            delay=delay,
            state_hints=state_hints,
            reasoning=None,
        )
        with state.lock:
            state.sessions[session.id] = session
        if open_browser:
            webbrowser.open(f"{url}#run={session.id}")
        while session.status_snapshot()[0] not in {"done", "error", "aborted"}:
            time.sleep(0.2)
        status, error = session.status_snapshot()
        if status == "error":
            raise RuntimeError(error or "web watch failed")
        return session.run_dir
    finally:
        httpd.shutdown()
        httpd.server_close()


def last_server_url() -> str | None:
    with _LAST_SERVER_LOCK:
        return _LAST_SERVER_URL


def last_server_port() -> int | None:
    with _LAST_SERVER_LOCK:
        return _LAST_SERVER_PORT


def _make_handler(state: _ServerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "jongbench-webui/0.1"

        def do_HEAD(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    body = _page_bytes()
                    self._send_headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
                    return
                if parsed.path == "/api/demo":
                    demo = _demo_bytes(state)
                    self._send_headers(HTTPStatus.OK, "application/json; charset=utf-8", len(demo))
                    return
                raise _HTTPError(HTTPStatus.NOT_FOUND, "not found")
            except _HTTPError as exc:
                self._send_json(exc.status, {"error": exc.message})

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/":
                    self._send_bytes(HTTPStatus.OK, _page_bytes(), "text/html; charset=utf-8")
                elif path == "/api/sessions":
                    self._send_json(HTTPStatus.OK, [_session_list_item(session) for session in state.list_sessions()])
                elif path.startswith("/api/state/"):
                    session = self._session_from_path(path, "/api/state/")
                    self._send_json(HTTPStatus.OK, _session_state(session))
                elif path.startswith("/api/events/"):
                    session = self._session_from_path(path, "/api/events/")
                    query = parse_qs(parsed.query)
                    since = _int_query(query.get("since", ["0"])[0], 0)
                    self._send_events(session, since)
                elif path.startswith("/api/pending/"):
                    session = self._session_from_path(path, "/api/pending/")
                    self._send_json(HTTPStatus.OK, session.human_io.pending() if session.human_seat is not None else None)
                elif path.startswith("/api/review/"):
                    session = self._session_from_path(path, "/api/review/")
                    if session.review is None:
                        raise _HTTPError(HTTPStatus.NOT_FOUND, "review is not available")
                    self._send_json(HTTPStatus.OK, session.review)
                elif path == "/api/demo":
                    self._send_bytes(HTTPStatus.OK, _demo_bytes(state), "application/json; charset=utf-8")
                elif path == "/api/config":
                    self._send_json(HTTPStatus.OK, {"local_endpoints": state.allow_local_endpoints})
                else:
                    raise _HTTPError(HTTPStatus.NOT_FOUND, "not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except _HTTPError as exc:
                self._send_json(exc.status, {"error": exc.message})
            except Exception as exc:
                try:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self) -> None:
            reserved = False
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/start":
                    payload = self._read_json_body()
                    models = payload.get("models")
                    if not isinstance(models, list) or len(models) != 4 or not all(isinstance(item, str) for item in models):
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, "models must be a list of 4 spec strings")
                    human_seat = payload.get("human_seat")
                    if human_seat is not None and not isinstance(human_seat, int):
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, "human_seat must be an integer or null")
                    keys = payload.get("keys") or {}
                    if not isinstance(keys, dict):
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, "keys must be an object")
                    clean_keys = {str(k): str(v) for k, v in keys.items() if isinstance(v, str)}
                    try:
                        _validate_web_start_credentials(
                            models, clean_keys, state.allow_local_endpoints
                        )
                    except ValueError as exc:
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
                    seed = payload.get("seed")
                    if seed is not None and (
                        not isinstance(seed, int) or isinstance(seed, bool)
                    ):
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, "seed must be an integer or null")
                    label = payload.get("label")
                    if label is not None and not isinstance(label, str):
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, "label must be a string or null")
                    state_hints = payload.get("state_hints", True)
                    if not isinstance(state_hints, bool):
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, "state_hints must be a boolean")
                    if not state.reserve_start():
                        raise _HTTPError(HTTPStatus.CONFLICT, "too many games are already running")
                    reserved = True
                    if not state.allow_start(self.client_address[0]):
                        raise _HTTPError(HTTPStatus.TOO_MANY_REQUESTS, "too many starts from this IP")
                    try:
                        session = start_game_session(
                            models,
                            clean_keys,
                            seed,
                            human_seat,
                            label,
                            runs_root=state.runs_root,
                            weights=state.weights,
                            no_eval=state.no_eval,
                            delay=state.delay,
                            state_hints=state_hints,
                            reasoning=payload.get("reasoning"),
                        )
                    except ValueError as exc:
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
                    state.add_reserved(session)
                    reserved = False
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "run_id": session.id,
                            "names": list(session.names),
                            "human_seat": session.human_seat,
                        },
                    )
                elif parsed.path == "/api/models":
                    payload = self._read_json_body()
                    provider = payload.get("provider")
                    key = payload.get("key")
                    base_url = payload.get("base_url")
                    allowed = {"openrouter"}
                    if provider == "compat":
                        if not state.allow_local_endpoints or not _is_loopback_url(base_url):
                            raise _HTTPError(HTTPStatus.BAD_REQUEST, "local endpoint must use a loopback URL")
                        allowed.add("compat")
                    if not isinstance(provider, str) or provider not in allowed:
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, "provider must be a supported model provider")
                    if not isinstance(key, str) or (provider != "compat" and not key.strip()):
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, "key must be a non-empty string")
                    try:
                        models = _cached_list_models(
                            key.strip(), base_url if provider == "compat" else None
                        )
                    except (ValueError, RuntimeError) as exc:
                        raise _HTTPError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
                    self._send_json(HTTPStatus.OK, {"models": models})
                elif parsed.path.startswith("/api/abort/"):
                    session = self._session_from_path(parsed.path, "/api/abort/")
                    ok = session.abort()
                    self._send_json(HTTPStatus.OK, {"ok": ok})
                elif parsed.path.startswith("/api/choose/"):
                    session = self._session_from_path(parsed.path, "/api/choose/")
                    payload = self._read_json_body()
                    ok = session.human_io.choose(payload.get("generation"), payload.get("choice"))
                    self._send_json(HTTPStatus.OK, {"ok": ok})
                else:
                    raise _HTTPError(HTTPStatus.NOT_FOUND, "not found")
            except _HTTPError as exc:
                if reserved:
                    state.release_reserved()
                self._send_json(exc.status, {"error": exc.message})
            except Exception as exc:
                if reserved:
                    state.release_reserved()
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _session_from_path(self, path: str, prefix: str) -> GameSession:
            run_id = path[len(prefix) :].strip("/")
            if not run_id:
                raise _HTTPError(HTTPStatus.NOT_FOUND, "missing run id")
            session = state.get_session(run_id)
            if session is None:
                raise _HTTPError(HTTPStatus.NOT_FOUND, "unknown run id")
            return session

        def _read_json_body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise _HTTPError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
            if length < 0:
                raise _HTTPError(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            if length > _BODY_LIMIT:
                raise _HTTPError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
            raw = self.rfile.read(length)
            try:
                loaded = json.loads(raw.decode("utf-8") if raw else "{}")
            except json.JSONDecodeError as exc:
                raise _HTTPError(HTTPStatus.BAD_REQUEST, "invalid JSON body") from exc
            if not isinstance(loaded, dict):
                raise _HTTPError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
            return loaded

        def _send_headers(self, status: int, content_type: str, length: int | None = None) -> None:
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
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._send_bytes(status, body, "application/json; charset=utf-8")

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
                        self._write_sse("frame", frame)
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

        def _write_sse(self, event: str, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self.wfile.write(f"event: {event}\n".encode("utf-8"))
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler


def _make_server(state: _ServerState, host: str, port: int) -> tuple[ThreadingHTTPServer, str]:
    httpd = ThreadingHTTPServer((host, int(port)), _make_handler(state))
    httpd.daemon_threads = True
    actual_port = int(httpd.server_address[1])
    url_host = "127.0.0.1" if host == "0.0.0.0" else host
    if ":" in url_host and not url_host.startswith("["):
        url_host = f"[{url_host}]"
    url = f"http://{url_host}:{actual_port}/"
    with _LAST_SERVER_LOCK:
        global _LAST_SERVER_URL, _LAST_SERVER_PORT
        _LAST_SERVER_URL = url
        _LAST_SERVER_PORT = actual_port
    return httpd, url


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
        summaries = arena.run_games(engines, 1, seed_start=seed, log_dir=str(run_dir / "logs"))
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
            updates = spectator.events_since(seq)
            if updates:
                for item in updates:
                    seq = max(seq, int(item["seq"]))
                    event = copy.deepcopy(item["event"])
                    if event.get("type") == "finish":
                        partial_log({"type": "end_game"})
                        table.finish(event.get("names"), event.get("scores"))
                    else:
                        partial_log(event)
                        table.apply(event)
                    snapshot = table.snapshot()
                    public_event = event
                    if session.human_seat is not None:
                        snapshot = _mask_snapshot(snapshot, session.human_seat)
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
    masked = copy.deepcopy(snapshot)
    ticker = masked.get("ticker")
    if isinstance(ticker, list):
        for index, line in enumerate(ticker):
            if not isinstance(line, str):
                continue
            match = _DRAW_TICKER_RE.fullmatch(line)
            if match is not None and int(match.group(1)) != human_seat:
                ticker[index] = f"P{match.group(1)} drew a tile"
    hands = masked.get("hands")
    if isinstance(hands, list):
        for seat, hand in enumerate(hands):
            if seat != human_seat and isinstance(hand, list):
                hands[seat] = ["?"] * len(hand)
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
    logs = sorted((run_dir / "logs").glob("*.json.gz")) + sorted((run_dir / "logs").glob("*.json"))
    if not logs:
        raise RuntimeError("no mjai log was written")
    events = evaluate.load_mjai_log(str(logs[0]))
    check_cancelled()
    mortal = evaluate.load_engine(weights)
    check_cancelled()
    reviews: dict[int, dict[str, Any]] = {}
    for seat in range(4):
        check_cancelled()
        reviews[seat] = evaluate.review_player(events, seat, mortal)
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
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    response = dict(data)
    response["run_dir"] = str(run_dir)
    return response


def _make_engine_for_session(
    *,
    name: str,
    spec_str: str,
    spec: providers.ProviderSpec,
    seat: int,
    seed: tuple[int, int],
    keys: dict[str, str],
    spectator: Spectator,
    human_io: WebHumanIO,
    decisions_dir: Path,
    state_hints: bool,
    reasoning: str | None = None,
) -> Any:
    if spec.provider == "human":
        try:
            return make_engine(name, spec_str, human_io=human_io, spectator=spectator, concurrency=1)
        except (TypeError, ValueError):
            engine_cls = _NativeHumanEngine or _FallbackHumanEngine
            return engine_cls(name, human_io, spectator=spectator)
    kwargs: dict[str, Any] = {
        "spectator": spectator,
        "state_hints": state_hints,
    }
    if spec.provider == "random":
        kwargs["seed"] = seed[0] + seat * 1009 + seed[1] * 97
    else:
        kwargs["api_key"] = keys.get(spec.provider) or None
        kwargs["decision_log"] = _DecisionLogSink(
            decisions_dir / decision_filename(name)
        )
        kwargs["reasoning"] = reasoning
    return make_engine(name, spec_str, **kwargs)


def _prompt_safe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = copy.deepcopy(events)
    for event in safe:
        if event.get("type") == "tsumo" and event.get("pai") == "?":
            event.pop("pai", None)
    return safe


def _parse_spec(spec_str: str) -> providers.ProviderSpec:
    if spec_str == "human":
        try:
            return providers.parse_spec(spec_str)
        except ValueError:
            return providers.ProviderSpec(provider="human", model="human")
    return providers.parse_spec(spec_str)


def _validate_web_start_credentials(
    model_specs: list[str],
    keys: dict[str, str],
    allow_local_endpoints: bool = False,
) -> None:
    """Keep shared HTTP starts on fixed provider endpoints and visitor keys.

    Direct CLI/watch starts intentionally retain environment-key and custom
    compatibility-endpoint support.
    """
    parsed = [_parse_spec(spec) for spec in model_specs]
    local_specs = [spec for spec in parsed if spec.provider == "compat"]
    if local_specs and (
        not allow_local_endpoints
        or any(not _is_loopback_url(spec.base_url) for spec in local_specs)
    ):
        raise ValueError("local models require a loopback OpenAI-compatible URL")
    missing = sorted(
        {
            spec.provider
            for spec in parsed
            if spec.provider not in {"human", "random", "compat"}
            and not keys.get(spec.provider, "").strip()
        }
    )
    if missing:
        raise ValueError(
            "request-supplied API key required for: " + ", ".join(missing)
        )


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_loopback_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    try:
        hostname = parsed.hostname
    except ValueError:
        return False
    return bool(
        parsed.scheme in {"http", "https"}
        and hostname
        and _is_loopback_host(hostname)
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )


def _human_seat_from_specs(model_specs: list[str]) -> int | None:
    for idx, spec in enumerate(model_specs):
        if _parse_spec(spec).provider == "human":
            return idx
    return None


def _normalize_seed(seed: int | tuple[int, int] | list[int] | None) -> tuple[int, int]:
    if seed is None:
        return secrets.randbelow(900_000_000) + 1, 1
    if isinstance(seed, bool):
        raise ValueError("seed must be an int, [int, int], or null")
    if isinstance(seed, int):
        result = (int(seed), 1)
    elif isinstance(seed, (tuple, list)) and len(seed) == 2:
        if any(not isinstance(value, int) or isinstance(value, bool) for value in seed):
            raise ValueError("seed must be an int, [int, int], or null")
        result = (seed[0], seed[1])
    else:
        raise ValueError("seed must be an int, [int, int], or null")
    if any(value < 0 or value > 2**64 - 1 for value in result):
        raise ValueError("seed values must be between 0 and 2^64 - 1")
    return result


def _engine_names(
    parsed: list[providers.ProviderSpec],
    reasoning: list[str | None] | None = None,
) -> list[str]:
    levels = reasoning if reasoning is not None else [None, None, None, None]
    seen: dict[str, int] = {}
    names = []
    for seat, spec in enumerate(parsed):
        base = spec.display_name if spec.provider not in {"human"} else "human"
        base = _safe_name(base or f"P{seat}")
        level = levels[seat] if seat < len(levels) else None
        if level is not None:
            base = f"{base}-{level}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        names.append(base if count == 1 else f"{base}-{count}")
    return names


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip()).strip("._-")
    return (cleaned or "player")[:60]


def _make_run_dir(root: Path, label: str | None, run_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_label = (label or "").strip()
    slug = _safe_name(raw_label) if raw_label else ""
    name = f"{stamp}-{slug}-{run_id}" if slug else f"{stamp}-{run_id}"
    path = root / name
    suffix = 2
    while path.exists():
        path = root / f"{name}-{suffix}"
        suffix += 1
    return path


def _write_config(
    run_dir: Path,
    session: GameSession,
    seed: tuple[int, int],
    label: str | None,
    delay: float,
    no_eval: bool,
    reasoning: list[str | None] | None = None,
) -> None:
    config = {
        "label": label or run_dir.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "models": list(session.model_specs),
        "names": list(session.names),
        "games": 1,
        "seed_start": [int(seed[0]), int(seed[1])],
        "human_seat": session.human_seat,
        "delay": float(delay),
        "no_eval": bool(no_eval),
        "state_hints": session.state_hints,
    }
    if reasoning is not None:
        config["reasoning"] = list(reasoning)
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _session_list_item(session: GameSession) -> dict[str, Any]:
    status, _ = session.status_snapshot()
    return {
        "run_id": session.id,
        "status": status,
        "names": list(session.names),
        "created": session.created,
        "human_seat": session.human_seat,
        "final": copy.deepcopy(session.final) if status in {"done", "evaluating"} else None,
    }


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


def _page_bytes() -> bytes:
    page = Path(__file__).parent / "webui_page.html"
    if not page.exists():
        raise FileNotFoundError(
            "jongbench/webui_page.html is missing; build it with `cd webui && bun install && bun run build`"
        )
    return page.read_bytes()


def _demo_bytes(state: _ServerState) -> bytes:
    if not state.demo_path:
        raise _HTTPError(HTTPStatus.NOT_FOUND, "demo is not configured")
    path = Path(state.demo_path)
    if not path.exists():
        raise _HTTPError(HTTPStatus.NOT_FOUND, "demo was not found")
    return path.read_bytes()


def _int_query(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
