from __future__ import annotations

import json
import re
from typing import Any

try:
    from jongbench.tiles import fmt_hand, fmt_tile, tiles_from_counts
except ImportError:
    _BASE_TILES = [
        *(f"{n}m" for n in range(1, 10)),
        *(f"{n}p" for n in range(1, 10)),
        *(f"{n}s" for n in range(1, 10)),
        "E",
        "S",
        "W",
        "N",
        "P",
        "F",
        "C",
    ]
    _RED_FIVES = ("5mr", "5pr", "5sr")
    _ORDER = {tile: i for i, tile in enumerate(_BASE_TILES + list(_RED_FIVES))}

    def _deaka(tile: str) -> str:
        if tile in _RED_FIVES:
            return tile[0] + tile[1]
        return tile

    def _tile_key(tile: str) -> tuple[int, int]:
        base = _deaka(tile)
        return (_ORDER.get(base, 100), 0 if tile in _RED_FIVES else 1)

    def fmt_tile(s: str, glyphs: bool = False) -> str:
        del glyphs
        if s in _RED_FIVES:
            return "0" + s[1]
        return s

    def tiles_from_counts(counts: list[int], akas: list[bool]) -> list[str]:
        tiles: list[str] = []
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            tile = _BASE_TILES[i]
            red_index = {"5m": 0, "5p": 1, "5s": 2}.get(tile)
            if red_index is not None and akas[red_index]:
                tiles.append(_RED_FIVES[red_index])
                count -= 1
            tiles.extend([tile] * count)
        return sorted(tiles, key=_tile_key)

    def fmt_hand(tiles: list[str], glyphs: bool = False) -> str:
        del glyphs
        sorted_tiles = sorted(tiles, key=_tile_key)
        suited: dict[str, list[str]] = {"m": [], "p": [], "s": []}
        honors: list[str] = []
        for tile in sorted_tiles:
            shown = fmt_tile(tile)
            if len(shown) == 2 and shown[1] in suited:
                suited[shown[1]].append(shown[0])
            else:
                honors.append(shown)
        parts = [
            "".join(nums) + suit
            for suit, nums in suited.items()
            if nums
        ]
        if honors:
            parts.append("".join(honors))
        return " ".join(parts)


SYSTEM = """You are an expert riichi mahjong player in a benchmark match.
Rules: Tenhou-style four-player hanchan, red fives, and open tanyao allowed.
Play to maximize final placement, not just hand value.
Balance speed, value, defense, dealer pressure, riichi sticks, honba, and endgame standings.
Do not assume hidden information beyond your hand and public table state.
Tile notation: 1m-9m are manzu, 1p-9p are pinzu, 1s-9s are souzu.
Red fives are 0m/0p/0s in prompts and may appear as 5mr/5pr/5sr in raw events.
Honors: E S W N are winds; P F C are white, green, and red dragons.
Think briefly if needed, then end your reply with a single line containing only JSON of the form {"choice": N} where N is the number of your chosen option. No other text may follow that line."""


_ROUND_NAMES = {
    "E": "East",
    "S": "South",
    "W": "West",
    "N": "North",
}
_SEAT_WINDS = ("E", "S", "W", "N")


def kyoku_label(bakaze: str, kyoku: int, honba: int) -> str:
    wind = _ROUND_NAMES.get(bakaze, bakaze)
    return f"{wind} {kyoku} (honba {honba})"


def render_state(player_id: int, state: Any, events: list[dict[str, Any]]) -> str:
    start = _start_kyoku(events)
    bakaze = start.get("bakaze", "?")
    kyoku = int(start.get("kyoku", 0))
    honba = int(start.get("honba", 0))
    kyotaku = int(start.get("kyotaku", 0))
    oya = int(start.get("oya", 0))
    scores = list(start.get("scores", [0, 0, 0, 0]))

    lines = [
        f"Round: {kyoku_label(bakaze, kyoku, honba)}, kyotaku {kyotaku}, dealer P{oya}",
        "Seats: " + _seat_summary(player_id, oya, scores),
        "Dora indicators: " + _dora_indicators(events),
        f"Tiles remaining in wall: {70 - sum(1 for ev in events if ev.get('type') == 'tsumo')}",
        "Legend: ' after a discard means tsumogiri; * marks the riichi declaration discard.",
    ]

    melds = _melds_by_player(events)
    discards, riichi = _discards_and_riichi(events)
    for seat in range(4):
        marker = " (you)" if seat == player_id else ""
        wind = _SEAT_WINDS[(seat - oya) % 4]
        meld_text = ", ".join(melds[seat]) if melds[seat] else "-"
        discard_text = " ".join(discards[seat]) if discards[seat] else "-"
        riichi_text = "yes" if riichi[seat] else "no"
        score = scores[seat] if seat < len(scores) else 0
        lines.append(
            f"P{seat} {wind}{marker} score {score}: riichi {riichi_text}; "
            f"melds {meld_text}; discards {discard_text}"
        )

    hand = fmt_hand(
        tiles_from_counts(list(state.tehai), list(state.akas_in_hand))
    )
    hand_line = f"Your hand: {hand}"
    if (
        events
        and events[-1].get("type") == "tsumo"
        and int(events[-1].get("actor", -1)) == player_id
    ):
        drawn = state.last_self_tsumo() or events[-1].get("pai")
        if drawn:
            hand_line += f"; just drew: {fmt_tile(str(drawn))}"
    lines.append(hand_line)

    recent = [_event_words(ev) for ev in events[-3:]]
    if recent:
        lines.append("Last events: " + "; ".join(recent))

    return "\n".join(lines)


def render_menu(menu: list[dict[str, Any]]) -> str:
    return "\n".join(f"{i}: {item['label']}" for i, item in enumerate(menu))


def build_user_prompt(
    player_id: int,
    state: Any,
    events: list[dict[str, Any]],
    menu: list[dict[str, Any]],
    error_feedback: str | None = None,
) -> str:
    lines = [
        render_state(player_id, state, events),
        "",
        "Choose your action:",
        render_menu(menu),
    ]
    if error_feedback:
        lines.extend(
            [
                "",
                f"Your previous reply was invalid: {error_feedback}. Reply again, following the format.",
            ]
        )
    lines.extend(
        [
            "",
            'End with a single line containing only JSON: {"choice": N}',
        ]
    )
    return "\n".join(lines)


def extract_choice(text: str, n_options: int) -> int:
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL))):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "choice" in obj:
            choice = obj["choice"]
            if not isinstance(choice, int) or isinstance(choice, bool):
                raise ValueError("choice must be an integer")
            if not 0 <= choice < n_options:
                raise ValueError("choice out of range")
            return choice

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and re.fullmatch(r"\d+", lines[-1]):
        choice = int(lines[-1])
        if not 0 <= choice < n_options:
            raise ValueError("choice out of range")
        return choice

    raise ValueError("no choice found")


def _start_kyoku(events: list[dict[str, Any]]) -> dict[str, Any]:
    for ev in events:
        if ev.get("type") == "start_kyoku":
            return ev
    raise ValueError("events must include start_kyoku")


def _seat_summary(player_id: int, oya: int, scores: list[int]) -> str:
    parts = []
    for seat in range(4):
        wind = _SEAT_WINDS[(seat - oya) % 4]
        you = " (you)" if seat == player_id else ""
        score = scores[seat] if seat < len(scores) else 0
        parts.append(f"P{seat}={wind} {score}{you}")
    return "; ".join(parts)


def _dora_indicators(events: list[dict[str, Any]]) -> str:
    indicators = []
    start = _start_kyoku(events)
    if "dora_marker" in start:
        indicators.append(start["dora_marker"])
    indicators.extend(
        ev["dora_marker"]
        for ev in events
        if ev.get("type") == "dora" and "dora_marker" in ev
    )
    return ", ".join(fmt_tile(str(tile)) for tile in indicators) if indicators else "-"


def _melds_by_player(events: list[dict[str, Any]]) -> list[list[str]]:
    melds: list[list[str]] = [[] for _ in range(4)]
    for ev in events:
        event_type = ev.get("type")
        actor = ev.get("actor")
        if actor is None or not 0 <= int(actor) < 4:
            continue
        seat = int(actor)
        if event_type in {"chi", "pon", "daiminkan"}:
            tiles = [*ev.get("consumed", []), ev.get("pai")]
            label = event_type
            melds[seat].append(
                f"{label} {_format_tiles(tiles)} (from P{ev.get('target')})"
            )
        elif event_type == "kakan":
            tiles = [*ev.get("consumed", []), ev.get("pai")]
            melds[seat].append(f"kakan {_format_tiles(tiles)}")
        elif event_type == "ankan":
            melds[seat].append(f"ankan {_format_tiles(ev.get('consumed', []))}")
    return melds


def _discards_and_riichi(
    events: list[dict[str, Any]],
) -> tuple[list[list[str]], list[bool]]:
    discards: list[list[str]] = [[] for _ in range(4)]
    riichi = [False] * 4
    pending_reach = [False] * 4
    for ev in events:
        event_type = ev.get("type")
        actor = ev.get("actor")
        if actor is None or not 0 <= int(actor) < 4:
            continue
        seat = int(actor)
        if event_type == "reach":
            pending_reach[seat] = True
        elif event_type == "reach_accepted":
            riichi[seat] = True
        elif event_type == "dahai":
            tile = fmt_tile(str(ev.get("pai", "?")))
            if ev.get("tsumogiri"):
                tile += "'"
            if pending_reach[seat]:
                tile += "*"
                pending_reach[seat] = False
            discards[seat].append(tile)
    return discards, riichi


def _format_tiles(tiles: list[Any]) -> str:
    return fmt_hand([str(tile) for tile in tiles if tile])


def _event_words(ev: dict[str, Any]) -> str:
    event_type = ev.get("type")
    actor = ev.get("actor")
    if event_type == "start_kyoku":
        return "Round started"
    if event_type == "tsumo":
        if "pai" in ev:
            return f"P{actor} drew {fmt_tile(str(ev['pai']))}"
        return f"P{actor} drew"
    if event_type == "dahai":
        tile = fmt_tile(str(ev.get("pai", "?")))
        return f"P{actor} discarded {tile}"
    if event_type == "chi":
        return f"P{actor} called chi on {fmt_tile(str(ev.get('pai', '?')))}"
    if event_type == "pon":
        return f"P{actor} called pon on {fmt_tile(str(ev.get('pai', '?')))}"
    if event_type == "daiminkan":
        return f"P{actor} called daiminkan on {fmt_tile(str(ev.get('pai', '?')))}"
    if event_type == "kakan":
        return f"P{actor} declared kakan {fmt_tile(str(ev.get('pai', '?')))}"
    if event_type == "ankan":
        return f"P{actor} declared ankan"
    if event_type == "reach":
        return f"P{actor} declared riichi"
    if event_type == "reach_accepted":
        return f"P{actor} riichi accepted"
    if event_type == "dora":
        return f"Dora indicator revealed {fmt_tile(str(ev.get('dora_marker', '?')))}"
    if event_type == "hora":
        return f"P{actor} won"
    if event_type == "ryukyoku":
        return "Hand ended in exhaustive draw"
    return str(event_type or "event")
