from __future__ import annotations

from collections import deque
from copy import deepcopy
import re
import sys
import threading
import time
from typing import Any, Callable

from .tiles import deaka, fmt_tile, sort_tiles


_SEAT_WINDS = ("E", "S", "W", "N")
_ROUND_NAMES = {"E": "East", "S": "South", "W": "West", "N": "North"}
_RESET = "\x1b[0m"
_RIICHI = "\x1b[1;33mRIICHI\x1b[0m"
_TILE_RE = re.compile(r"\[(\s*)([0-9][mps]|[ESWNPFC])(\s*)\]")
_DRAW_TICKER_RE = re.compile(r"^P([0-3]) drew .+$")


class TableState:
    def __init__(self) -> None:
        self.names = [f"P{seat}" for seat in range(4)]
        self.scores = [25000, 25000, 25000, 25000]
        self.hands: list[list[str]] = [[] for _ in range(4)]
        self.hand = self.hands
        self.melds: list[list[dict[str, Any]]] = [[] for _ in range(4)]
        self.discards: list[list[dict[str, Any]]] = [[] for _ in range(4)]
        self.riichi_declared = [False, False, False, False]
        self.riichi_accepted = [False, False, False, False]
        self.bakaze = "E"
        self.kyoku = 1
        self.honba = 0
        self.kyotaku = 0
        self.oya = 0
        self.dora_indicators: list[str] = []
        self.tiles_left = 70
        self.ticker: deque[str] = deque(maxlen=8)
        self.kyoku_index = 0
        self.done = False
        self.final_names: list[str] | None = None
        self.final_scores: list[int] | None = None
        self._pending_riichi = [False, False, False, False]

    def set_scores(self, scores: list[int]) -> None:
        if len(scores) != 4:
            raise ValueError(f"expected 4 scores, got {len(scores)}")
        self.scores = [int(score) for score in scores]

    def finish(self, names: list[str] | None = None, scores: list[int] | None = None) -> None:
        if names is not None:
            if len(names) != 4:
                raise ValueError(f"expected 4 names, got {len(names)}")
            self.names = [str(name) for name in names]
            self.final_names = list(self.names)
        else:
            self.final_names = list(self.names)
        if scores is not None:
            self.set_scores(scores)
            self.final_scores = list(self.scores)
        else:
            self.final_scores = list(self.scores)
        self.kyotaku = 0
        self.done = True
        self._tick("Game finished")

    def apply(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "start_game":
            self._apply_start_game(event)
        elif event_type == "start_kyoku":
            self._apply_start_kyoku(event)
        elif event_type == "tsumo":
            self._apply_tsumo(event)
        elif event_type == "dahai":
            self._apply_dahai(event)
        elif event_type in {"chi", "pon", "daiminkan"}:
            self._apply_open_call(event)
        elif event_type == "ankan":
            self._apply_ankan(event)
        elif event_type == "kakan":
            self._apply_kakan(event)
        elif event_type == "dora":
            marker = event.get("dora_marker")
            if marker is not None:
                self.dora_indicators.append(str(marker))
                self._tick(f"Dora indicator {fmt_tile(str(marker))}")
        elif event_type == "reach":
            self._apply_reach(event)
        elif event_type == "reach_accepted":
            self._apply_reach_accepted(event)
        elif event_type == "hora":
            self._apply_hora(event)
        elif event_type == "ryukyoku":
            self._apply_ryukyoku(event)
        elif event_type == "end_game":
            if self.kyotaku:
                leader = min(
                    range(4), key=lambda seat: (-self.scores[seat], seat)
                )
                self.scores[leader] += self.kyotaku * 1000
                self.kyotaku = 0
            self.done = True
            self.final_names = list(self.names)
            self.final_scores = list(self.scores)
            self._tick("Game finished")

    def snapshot(self) -> dict[str, Any]:
        seats = []
        for seat in range(4):
            seats.append(
                {
                    "seat": seat,
                    "name": self.names[seat],
                    "wind": self.seat_wind(seat),
                    "score": self.scores[seat],
                    "hand": list(self.hands[seat]),
                    "melds": deepcopy(self.melds[seat]),
                    "discards": deepcopy(self.discards[seat]),
                    "riichi_declared": self.riichi_declared[seat],
                    "riichi_accepted": self.riichi_accepted[seat],
                }
            )
        return {
            "names": list(self.names),
            "scores": list(self.scores),
            "hands": [list(hand) for hand in self.hands],
            "melds": deepcopy(self.melds),
            "discards": deepcopy(self.discards),
            "riichi_declared": list(self.riichi_declared),
            "riichi_accepted": list(self.riichi_accepted),
            "seats": seats,
            "bakaze": self.bakaze,
            "kyoku": self.kyoku,
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "oya": self.oya,
            "dora_indicators": list(self.dora_indicators),
            "tiles_left": self.tiles_left,
            "ticker": list(self.ticker),
            "kyoku_index": self.kyoku_index,
            "done": self.done,
            "final_names": None if self.final_names is None else list(self.final_names),
            "final_scores": None if self.final_scores is None else list(self.final_scores),
        }

    def seat_wind(self, seat: int) -> str:
        return _SEAT_WINDS[(seat - self.oya) % 4]

    def _apply_start_game(self, event: dict[str, Any]) -> None:
        names = event.get("names")
        if isinstance(names, list) and len(names) == 4:
            self.names = [str(name) for name in names]
        self.done = False
        self.final_names = None
        self.final_scores = None
        self._tick("Game started")

    def _apply_start_kyoku(self, event: dict[str, Any]) -> None:
        self.bakaze = str(event.get("bakaze", "E"))
        self.kyoku = int(event.get("kyoku", 1))
        self.honba = int(event.get("honba", 0))
        self.kyotaku = int(event.get("kyotaku", 0))
        self.oya = int(event.get("oya", 0))
        scores = event.get("scores")
        if isinstance(scores, list) and len(scores) == 4:
            self.set_scores(scores)
        tehais = event.get("tehais") or [[], [], [], []]
        self.hands = [
            sort_tiles([str(tile) for tile in tehais[seat]]) if seat < len(tehais) else []
            for seat in range(4)
        ]
        self.hand = self.hands
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.riichi_declared = [False, False, False, False]
        self.riichi_accepted = [False, False, False, False]
        self._pending_riichi = [False, False, False, False]
        marker = event.get("dora_marker")
        self.dora_indicators = [str(marker)] if marker is not None else []
        self.tiles_left = 70
        self.ticker.clear()
        self.kyoku_index += 1
        self.done = False
        self._tick(f"{_ROUND_NAMES.get(self.bakaze, self.bakaze)} {self.kyoku} started")

    def _apply_tsumo(self, event: dict[str, Any]) -> None:
        actor = _actor(event)
        if actor is None:
            return
        pai = event.get("pai")
        if pai is not None:
            self.hands[actor].append(str(pai))
            self.hands[actor] = sort_tiles(self.hands[actor])
            self._tick(f"P{actor} drew {fmt_tile(str(pai))}")
        self.tiles_left = max(0, self.tiles_left - 1)

    def _apply_dahai(self, event: dict[str, Any]) -> None:
        actor = _actor(event)
        pai = event.get("pai")
        if actor is None or pai is None:
            return
        tile = str(pai)
        _remove_tile(self.hands[actor], tile)
        self.hands[actor] = sort_tiles(self.hands[actor])
        riichi = self._pending_riichi[actor]
        self._pending_riichi[actor] = False
        self.discards[actor].append(
            {
                "tile": tile,
                "tsumogiri": bool(event.get("tsumogiri", False)),
                "riichi": riichi,
                "called": False,
            }
        )
        suffix = " (riichi)" if riichi else ""
        self._tick(f"P{actor} discarded {fmt_tile(tile)}{suffix}")

    def _apply_open_call(self, event: dict[str, Any]) -> None:
        actor = _actor(event)
        target = _target(event)
        pai = event.get("pai")
        if actor is None or target is None or pai is None:
            return
        consumed = [str(tile) for tile in event.get("consumed", [])]
        for tile in consumed:
            _remove_tile(self.hands[actor], tile)
        self.hands[actor] = sort_tiles(self.hands[actor])
        self._mark_called(target)
        kind = str(event.get("type"))
        tiles = [*consumed, str(pai)]
        self.melds[actor].append(
            {"kind": kind, "tiles": tiles, "from_seat": target}
        )
        self._tick(f"P{actor} called {kind} on P{target}'s {fmt_tile(str(pai))}")

    def _apply_ankan(self, event: dict[str, Any]) -> None:
        actor = _actor(event)
        if actor is None:
            return
        consumed = [str(tile) for tile in event.get("consumed", [])]
        for tile in consumed:
            _remove_tile(self.hands[actor], tile)
        self.hands[actor] = sort_tiles(self.hands[actor])
        self.melds[actor].append(
            {"kind": "ankan", "tiles": consumed, "from_seat": None}
        )
        shown = fmt_tile(consumed[0]) if consumed else "?"
        self._tick(f"P{actor} declared ankan {shown}")

    def _apply_kakan(self, event: dict[str, Any]) -> None:
        actor = _actor(event)
        pai = event.get("pai")
        if actor is None or pai is None:
            return
        tile = str(pai)
        consumed = [str(item) for item in event.get("consumed", [])]
        _remove_tile(self.hands[actor], tile)
        self.hands[actor] = sort_tiles(self.hands[actor])
        meld = self._find_pon_meld(actor, tile)
        if meld is None:
            self.melds[actor].append(
                {"kind": "kakan", "tiles": [*consumed, tile], "from_seat": None}
            )
        else:
            meld["kind"] = "kakan"
            meld["tiles"] = [*consumed, tile] if consumed else [*meld["tiles"], tile]
        self._tick(f"P{actor} declared kakan {fmt_tile(tile)}")

    def _apply_reach(self, event: dict[str, Any]) -> None:
        actor = _actor(event)
        if actor is None:
            return
        self.riichi_declared[actor] = True
        self._pending_riichi[actor] = True
        self._tick(f"P{actor} declared riichi")

    def _apply_reach_accepted(self, event: dict[str, Any]) -> None:
        actor = _actor(event)
        if actor is None:
            return
        if not self.riichi_accepted[actor]:
            self.scores[actor] -= 1000
            self.kyotaku += 1
        self.riichi_declared[actor] = True
        self.riichi_accepted[actor] = True
        self._tick(f"P{actor} riichi accepted")

    def _apply_hora(self, event: dict[str, Any]) -> None:
        deltas = event.get("deltas")
        if isinstance(deltas, list) and len(deltas) == 4:
            self.scores = [score + int(delta) for score, delta in zip(self.scores, deltas, strict=True)]
        self.kyotaku = 0
        actor = _actor(event)
        target = _target(event)
        if actor is None:
            self._tick("Win")
        elif target == actor:
            self._tick(f"P{actor} won by tsumo")
        elif target is not None:
            self._tick(f"P{actor} won from P{target}")
        else:
            self._tick(f"P{actor} won")

    def _apply_ryukyoku(self, event: dict[str, Any]) -> None:
        deltas = event.get("deltas")
        if isinstance(deltas, list) and len(deltas) == 4:
            self.scores = [score + int(delta) for score, delta in zip(self.scores, deltas, strict=True)]
        self._tick("Hand ended in draw")

    def _find_pon_meld(self, actor: int, tile: str) -> dict[str, Any] | None:
        base = deaka(tile)
        for meld in reversed(self.melds[actor]):
            if meld.get("kind") != "pon":
                continue
            tiles = [str(item) for item in meld.get("tiles", [])]
            if tiles and all(deaka(item) == base for item in tiles):
                return meld
        return None

    def _mark_called(self, target: int) -> None:
        if not self.discards[target]:
            return
        self.discards[target][-1]["called"] = True

    def _tick(self, text: str) -> None:
        self.ticker.append(text)


class Spectator:
    def __init__(
        self,
        *,
        delay: float = 0.0,
        on_update: Callable[[TableState], None] | None = None,
        names: list[str] | None = None,
    ) -> None:
        self.table = TableState()
        if names is not None:
            if len(names) != 4:
                raise ValueError(f"expected 4 names, got {len(names)}")
            self.table.names = [str(name) for name in names]
        self.delay = delay
        self.on_update = on_update
        self._lock = threading.Lock()
        self._kyoku_first_event: dict[str, Any] | None = None
        self._applied_len = 0
        self._kyoku_counter = 0
        self._feed: list[dict[str, Any]] = []
        self._next_seq = 1

    def publish(self, game_index: int, seat: int, events: list[dict[str, Any]]) -> None:
        del seat
        if game_index != 0 or not events:
            return
        with self._lock:
            first = events[0]
            if self._kyoku_first_event != first:
                self._kyoku_first_event = deepcopy(first)
                self._applied_len = 0
                self._kyoku_counter += 1
            if len(events) <= self._applied_len:
                return
            new_events = events[self._applied_len :]
            for event in new_events:
                self.table.apply(event)
                self._feed.append({"seq": self._next_seq, "event": deepcopy(event)})
                self._next_seq += 1
                if self.on_update is not None:
                    self.on_update(self.table)
                delay = event.get("delay", self.delay)
                try:
                    delay_seconds = float(delay)
                except (TypeError, ValueError):
                    delay_seconds = 0.0
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
            self._applied_len = len(events)

    def events_since(self, seq: int) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in self._feed if int(item["seq"]) > seq]

    def finish(
        self,
        names: list[str] | None = None,
        scores: list[int] | None = None,
    ) -> None:
        with self._lock:
            self.table.finish(names, scores)
            self._feed.append(
                {
                    "seq": self._next_seq,
                    "event": {
                        "type": "finish",
                        "names": list(self.table.names),
                        "scores": list(self.table.scores),
                    },
                }
            )
            self._next_seq += 1


def render_table(
    table: TableState,
    glyphs: bool = False,
    *,
    reveal: bool = True,
    pov: int = 0,
) -> str:
    if not 0 <= pov < 4:
        raise ValueError(f"expected pov in 0..3, got {pov}")
    bottom = pov
    right = (pov + 1) % 4
    top = (pov + 2) % 4
    left = (pov + 3) % 4
    width = 100
    height = 40
    grid = [[" " for _ in range(width)] for _ in range(height)]

    _put_center(grid, 0, _seat_title(table, top), width)
    _put_center(grid, 1, _hand_line(table, top, reveal, glyphs, pov), width)
    _put_center(grid, 2, _meld_line(table, top, glyphs), width)
    for row, line in enumerate(
        _discard_lines(table.discards[top], glyphs, max_rows=4), start=4
    ):
        _put_center(grid, row, line, width)

    _put_center(grid, 38, _hand_line(table, bottom, reveal, glyphs, pov), width)
    _put_center(grid, 37, _meld_line(table, bottom, glyphs), width)
    _put_center(grid, 39, _seat_title(table, bottom), width)
    bottom_discards = _discard_lines(table.discards[bottom], glyphs, max_rows=4)
    bottom_start = 32 + max(0, 4 - len(bottom_discards))
    for row, line in enumerate(bottom_discards, start=bottom_start):
        _put_center(grid, row, line, width)

    _put(grid, 8, 0, _fit(_seat_title(table, left), 28))
    _put_vertical_hand(grid, table, left, 0, 10, reveal, glyphs, pov)
    _put_side_discards(grid, table.discards[left], 7, 12, glyphs)
    _put(grid, 26, 0, _fit(_meld_line(table, left, glyphs), 30))

    _put_right(grid, 8, _fit(_seat_title(table, right), 28), width)
    _put_vertical_hand(grid, table, right, width - 5, 10, reveal, glyphs, pov)
    _put_side_discards(grid, table.discards[right], 70, 12, glyphs)
    _put_right(grid, 26, _fit(_meld_line(table, right, glyphs), 30), width)

    _put_center_box(grid, table, glyphs, pov)

    rendered = "\n".join("".join(row).rstrip() for row in grid)
    return _colorize(rendered)


class TerminalRenderer:
    def __init__(
        self,
        *,
        glyphs: bool = False,
        reveal: bool = True,
        pov: int = 0,
        stream: Any | None = None,
        min_interval: float = 0.1,
    ) -> None:
        self.glyphs = glyphs
        self.reveal = reveal
        self.pov = pov
        self.stream = stream if stream is not None else sys.stdout
        self.min_interval = min_interval
        self._last_draw = 0.0

    def __call__(self, table: TableState) -> None:
        self.on_update(table)

    def on_update(self, table: TableState) -> None:
        now = time.monotonic()
        if now - self._last_draw < self.min_interval:
            return
        self._last_draw = now
        print("\x1b[H\x1b[2J", end="", file=self.stream)
        print(
            render_table(
                table,
                glyphs=self.glyphs,
                reveal=self.reveal,
                pov=self.pov,
            ),
            file=self.stream,
        )
        print("", file=self.stream)
        for line in table.ticker:
            print(_visible_ticker_line(line, self.reveal, self.pov), file=self.stream)
        self.stream.flush()


def _actor(event: dict[str, Any]) -> int | None:
    actor = event.get("actor")
    if actor is None:
        return None
    actor = int(actor)
    if not 0 <= actor < 4:
        return None
    return actor


def _target(event: dict[str, Any]) -> int | None:
    target = event.get("target")
    if target is None:
        return None
    target = int(target)
    if not 0 <= target < 4:
        return None
    return target


def _remove_tile(hand: list[str], tile: str) -> None:
    for idx, candidate in enumerate(hand):
        if candidate == tile:
            del hand[idx]
            return
    base = deaka(tile)
    for idx, candidate in enumerate(hand):
        if deaka(candidate) == base:
            del hand[idx]
            return


def _seat_title(table: TableState, seat: int) -> str:
    tag = " RIICHI" if table.riichi_declared[seat] or table.riichi_accepted[seat] else ""
    return f"P{seat} {table.names[seat]}{tag}"


def _tile_cell(tile: str | None, glyphs: bool, *, concealed: bool = False) -> str:
    shown = "\u25aa" if concealed else fmt_tile(str(tile), glyphs=glyphs)
    return f"[{shown:^3}]"


def _hand_line(
    table: TableState,
    seat: int,
    reveal: bool,
    glyphs: bool,
    pov: int,
) -> str:
    hand = table.hands[seat]
    concealed = not reveal and seat != pov
    return " ".join(_tile_cell(tile, glyphs, concealed=concealed) for tile in hand)


def _meld_line(table: TableState, seat: int, glyphs: bool) -> str:
    melds = []
    for meld in table.melds[seat]:
        tiles = "".join(_tile_cell(str(tile), glyphs) for tile in meld.get("tiles", []))
        source = meld.get("from_seat")
        suffix = "" if source is None else f"<P{source}"
        melds.append(f"{meld.get('kind', '?')} {tiles}{suffix}")
    return " | ".join(melds)


def _discard_cell(discard: dict[str, Any], glyphs: bool) -> str:
    marks = ""
    if discard.get("riichi"):
        marks += "*"
    if discard.get("tsumogiri"):
        marks += "'"
    if discard.get("called"):
        marks += "x"
    return f"{_tile_cell(str(discard.get('tile', '?')), glyphs)}{marks:<3}"


def _discard_lines(
    discards: list[dict[str, Any]],
    glyphs: bool,
    *,
    max_rows: int,
) -> list[str]:
    visible = discards[-max_rows * 6 :]
    rows = []
    for start in range(0, len(visible), 6):
        rows.append(" ".join(_discard_cell(discard, glyphs) for discard in visible[start : start + 6]))
    return rows


def _put_vertical_hand(
    grid: list[list[str]],
    table: TableState,
    seat: int,
    col: int,
    row: int,
    reveal: bool,
    glyphs: bool,
    pov: int,
) -> None:
    concealed = not reveal and seat != pov
    for idx, tile in enumerate(table.hands[seat][:14]):
        _put(grid, row + idx, col, _tile_cell(tile, glyphs, concealed=concealed))


def _put_side_discards(
    grid: list[list[str]],
    discards: list[dict[str, Any]],
    col: int,
    row: int,
    glyphs: bool,
) -> None:
    for idx, discard in enumerate(discards[-18:]):
        _put(grid, row + idx % 6, col + (idx // 6) * 8, _discard_cell(discard, glyphs))


def _put_center_box(
    grid: list[list[str]],
    table: TableState,
    glyphs: bool,
    pov: int,
) -> None:
    left = 31
    top = 15
    box_width = 38
    inner = box_width - 2
    rows = [
        "+" + "-" * inner + "+",
        _box_line(_round_label(table), inner),
        _box_line(f"honba {table.honba}  kyotaku {table.kyotaku}", inner),
        _box_line(f"tiles left {table.tiles_left}", inner),
        _box_line("dora " + " ".join(_tile_cell(tile, glyphs) for tile in table.dora_indicators), inner),
        _box_line(_score_line(table, (pov + 2) % 4), inner),
        _box_line(
            f"{_score_line(table, (pov + 3) % 4):<17}"
            f"{_score_line(table, (pov + 1) % 4):>17}",
            inner,
        ),
        _box_line(_score_line(table, pov), inner),
        "+" + "-" * inner + "+",
    ]
    for idx, row in enumerate(rows):
        _put(grid, top + idx, left, row)


def _round_label(table: TableState) -> str:
    return f"{_ROUND_NAMES.get(table.bakaze, table.bakaze)} {table.kyoku}"


def _score_line(table: TableState, seat: int) -> str:
    return f"P{seat} {table.seat_wind(seat)} {table.scores[seat]}"


def _visible_ticker_line(line: str, reveal: bool, pov: int) -> str:
    if reveal:
        return line
    match = _DRAW_TICKER_RE.fullmatch(line)
    if match is not None and int(match.group(1)) != pov:
        return f"P{match.group(1)} drew a tile"
    return line


def _box_line(text: str, inner: int) -> str:
    text = _fit(text, inner)
    return "|" + text.center(inner) + "|"


def _put(grid: list[list[str]], row: int, col: int, text: str) -> None:
    if not 0 <= row < len(grid):
        return
    for idx, char in enumerate(text):
        target_col = col + idx
        if 0 <= target_col < len(grid[row]):
            grid[row][target_col] = char


def _put_center(grid: list[list[str]], row: int, text: str, width: int) -> None:
    text = _fit(text, width)
    _put(grid, row, max(0, (width - len(text)) // 2), text)


def _put_right(grid: list[list[str]], row: int, text: str, width: int) -> None:
    text = _fit(text, width)
    _put(grid, row, max(0, width - len(text)), text)


def _fit(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    if width <= 3:
        return "." * width
    return text[: width - 3] + "..."


def _colorize(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        left, token, right = match.groups()
        return "[" + left + _color_token(token) + right + "]"

    text = _TILE_RE.sub(replace, text)
    text = text.replace("RIICHI", _RIICHI)
    face_down = "[ " + "\u25aa" + " ]"
    text = text.replace(face_down, "\x1b[90m" + face_down + "\x1b[0m")
    return text


def _color_token(token: str) -> str:
    if token.startswith("0"):
        return f"\x1b[1;91m{token}{_RESET}"
    if token.endswith("m"):
        return f"\x1b[38;5;203m{token}{_RESET}"
    if token.endswith("p"):
        return f"\x1b[38;5;75m{token}{_RESET}"
    if token.endswith("s"):
        return f"\x1b[38;5;78m{token}{_RESET}"
    return f"\x1b[1;97m{token}{_RESET}"
