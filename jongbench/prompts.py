from __future__ import annotations

import json
import re
from typing import Any

try:
    from jongbench.tiles import deaka, fmt_tile, label_to_tile, tiles_from_counts
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

    deaka = _deaka

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

    def label_to_tile(label: int) -> str:
        return _BASE_TILES[label]


SYSTEM = """You are an expert riichi mahjong player in a benchmark match.
Rules: Tenhou-style four-player hanchan, red fives, and open tanyao allowed.
Play to maximize final placement, not just hand value.
Balance speed, value, defense, dealer pressure, riichi sticks, honba, and endgame standings.
Do not assume hidden information beyond your hand and public table state.
Every concealed tile is written separately. Suited tiles use 1m-9m, 1p-9p, and 1s-9s.
Red fives are written as 5m(red), 5p(red), and 5s(red); they are still rank 5.
Honors: E S W N are winds; P F C are white, green, and red dragons.
A dora indicator is not itself dora; the next tile in its cycle is dora. Dora alone is not a yaku.
All listed actions are legal. A win option is listed only when the hand may legally win now.
Structural tenpai and waits do not by themselves guarantee a yaku or legal win.
Passing an available ron causes temporary furiten, or permanent furiten after riichi.
Riichi is a two-step action: choose riichi first, then choose a legal discard in the next prompt.
A standard complete hand has four melds and one pair; listed melds are already complete.
Reason internally. Reply with exactly one JSON object of the form {"choice": N} and no prose."""


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


def render_state(
    player_id: int,
    state: Any,
    events: list[dict[str, Any]],
    *,
    state_hints: bool = False,
) -> str:
    start = _start_kyoku(events)
    bakaze = start.get("bakaze", "?")
    kyoku = int(start.get("kyoku", 0))
    honba = int(start.get("honba", 0))
    oya = int(start.get("oya", 0))
    scores, kyotaku = _scores_and_kyotaku(start, events)

    lines = [
        f"Round: {kyoku_label(bakaze, kyoku, honba)}, kyotaku {kyotaku}, dealer P{oya}",
        "Seats: " + _seat_summary(player_id, oya, scores),
        "Dora indicators: " + _dora_indicators(events),
        f"Tiles remaining in wall: {70 - sum(1 for ev in events if ev.get('type') == 'tsumo')}",
        "Legend: ' after a discard means tsumogiri; * marks the riichi declaration discard.",
    ]
    if state_hints:
        lines.extend(_state_hint_lines(state))

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

    hand_tiles = tiles_from_counts(list(state.tehai), list(state.akas_in_hand))
    meld_count = len(melds[player_id])
    hand_line = (
        f"Your concealed hand ({len(hand_tiles)} tiles; "
        f"{meld_count} completed melds): "
        f"{_format_tiles(hand_tiles)}"
    )
    if (
        events
        and events[-1].get("type") == "tsumo"
        and int(events[-1].get("actor", -1)) == player_id
    ):
        drawn = state.last_self_tsumo() or events[-1].get("pai")
        if drawn:
            hand_line += f"; just drew: {_prompt_tile(str(drawn))}"
    lines.append(hand_line)

    recent = [_event_words(ev) for ev in events[-3:]]
    if recent:
        lines.append("Last events: " + "; ".join(recent))

    return "\n".join(lines)


def render_menu(
    menu: list[dict[str, Any]],
    *,
    state: Any | None = None,
    state_hints: bool = False,
) -> str:
    lines = []
    for index, item in enumerate(menu):
        label = _prompt_action_label(str(item["label"]))
        hint = _action_hint(state, item) if state_hints and state is not None else ""
        lines.append(f"{index}: {label}{hint}")
    return "\n".join(lines)


def build_user_prompt(
    player_id: int,
    state: Any,
    events: list[dict[str, Any]],
    menu: list[dict[str, Any]],
    error_feedback: str | None = None,
    *,
    state_hints: bool = False,
) -> str:
    lines = [
        render_state(player_id, state, events, state_hints=state_hints),
        "",
        "Choose your action:",
        render_menu(menu, state=state, state_hints=state_hints),
    ]
    if error_feedback:
        lines.extend(
            [
                "",
                f"Your previous reply was invalid: {error_feedback}. Reply again, following the format.",
            ]
        )
    lines.extend(
        ["", 'Reply with exactly: {"choice": N}'],
    )
    return "\n".join(lines)


def extract_choice(text: str, n_options: int) -> int:
    if n_options <= 0:
        raise ValueError("no options available")
    choices: list[int] = []
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
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
            choices.append(choice)

    if choices:
        if len(set(choices)) != 1:
            raise ValueError("conflicting choice values")
        return choices[-1]

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
    return ", ".join(_prompt_tile(str(tile)) for tile in indicators) if indicators else "-"


def _melds_by_player(events: list[dict[str, Any]]) -> list[list[str]]:
    raw_melds: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    for ev in events:
        event_type = ev.get("type")
        actor = ev.get("actor")
        if actor is None or not 0 <= int(actor) < 4:
            continue
        seat = int(actor)
        if event_type in {"chi", "pon", "daiminkan"}:
            raw_melds[seat].append(ev)
        elif event_type == "kakan":
            base = deaka(str(ev.get("pai", "")))
            pon_index = next(
                (
                    index
                    for index in range(len(raw_melds[seat]) - 1, -1, -1)
                    if raw_melds[seat][index].get("type") == "pon"
                    and all(
                        deaka(str(tile)) == base
                        for tile in [
                            *raw_melds[seat][index].get("consumed", []),
                            raw_melds[seat][index].get("pai"),
                        ]
                        if tile is not None
                    )
                ),
                None,
            )
            if pon_index is None:
                raw_melds[seat].append(ev)
            else:
                raw_melds[seat][pon_index] = ev
        elif event_type == "ankan":
            raw_melds[seat].append(ev)

    melds: list[list[str]] = [[] for _ in range(4)]
    for seat, seat_melds in enumerate(raw_melds):
        for ev in seat_melds:
            event_type = str(ev.get("type"))
            if event_type in {"chi", "pon", "daiminkan"}:
                tiles = [*ev.get("consumed", []), ev.get("pai")]
                melds[seat].append(
                    f"{event_type} {_format_tiles(tiles)} (from P{ev.get('target')})"
                )
            elif event_type == "kakan":
                tiles = [*ev.get("consumed", []), ev.get("pai")]
                melds[seat].append(f"kakan {_format_tiles(tiles)}")
            elif event_type == "ankan":
                melds[seat].append(
                    f"ankan {_format_tiles(ev.get('consumed', []))}"
                )
    return melds


def _scores_and_kyotaku(
    start: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[list[int], int]:
    scores = [int(score) for score in start.get("scores", [0, 0, 0, 0])]
    kyotaku = int(start.get("kyotaku", 0))
    for event in events:
        if event.get("type") != "reach_accepted":
            continue
        actor = event.get("actor")
        if isinstance(actor, int) and 0 <= actor < len(scores):
            scores[actor] -= 1000
            kyotaku += 1
    return scores, kyotaku


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
            tile = _prompt_tile(str(ev.get("pai", "?")))
            if ev.get("tsumogiri"):
                tile += "'"
            if pending_reach[seat]:
                tile += "*"
                pending_reach[seat] = False
            discards[seat].append(tile)
    return discards, riichi


def _format_tiles(tiles: list[Any]) -> str:
    shown = [_prompt_tile(str(tile)) for tile in tiles if tile]
    return " ".join(shown) if shown else "-"


def _event_words(ev: dict[str, Any]) -> str:
    event_type = ev.get("type")
    actor = ev.get("actor")
    if event_type == "start_kyoku":
        return "Round started"
    if event_type == "tsumo":
        if "pai" in ev:
            return f"P{actor} drew {_prompt_tile(str(ev['pai']))}"
        return f"P{actor} drew"
    if event_type == "dahai":
        tile = _prompt_tile(str(ev.get("pai", "?")))
        return f"P{actor} discarded {tile}"
    if event_type == "chi":
        return f"P{actor} called chi on {_prompt_tile(str(ev.get('pai', '?')))}"
    if event_type == "pon":
        return f"P{actor} called pon on {_prompt_tile(str(ev.get('pai', '?')))}"
    if event_type == "daiminkan":
        return f"P{actor} called daiminkan on {_prompt_tile(str(ev.get('pai', '?')))}"
    if event_type == "kakan":
        return f"P{actor} declared kakan {_prompt_tile(str(ev.get('pai', '?')))}"
    if event_type == "ankan":
        return f"P{actor} declared ankan"
    if event_type == "reach":
        return f"P{actor} declared riichi"
    if event_type == "reach_accepted":
        return f"P{actor} riichi accepted"
    if event_type == "dora":
        return f"Dora indicator revealed {_prompt_tile(str(ev.get('dora_marker', '?')))}"
    if event_type == "hora":
        return f"P{actor} won"
    if event_type == "ryukyoku":
        return "Hand ended in exhaustive draw"
    return str(event_type or "event")


def _prompt_tile(tile: str) -> str:
    if tile in {"5mr", "5pr", "5sr"}:
        return f"{tile[:2]}(red)"
    return fmt_tile(tile)


def _prompt_action_label(label: str) -> str:
    return re.sub(r"0([mps])(?: \(red\))?", r"5\1(red)", label)


def _state_hint_lines(state: Any) -> list[str]:
    shanten = _real_time_shanten(state)
    furiten = bool(getattr(state, "at_furiten", False))
    can_discard = bool(getattr(getattr(state, "last_cans", None), "can_discard", False))
    if can_discard:
        status = f"best result after the required discard: {_shanten_text(shanten)}"
        waits: list[str] = []
    else:
        status = f"current structure: {_shanten_text(shanten)}"
        waits = _wait_tiles(getattr(state, "waits", [])) if shanten == 0 else []
    details = [status]
    if shanten == 0 and not can_discard:
        details.append("waits: " + (" ".join(waits) if waits else "none (dead wait)"))
    details.append(f"furiten: {'yes' if furiten else 'no'}")
    return [
        "Engine-derived state hints (structural only; no EV or recommendation):",
        "- " + "; ".join(details),
    ]


def _action_hint(state: Any, item: dict[str, Any]) -> str:
    event = item.get("event")
    if not isinstance(event, dict) or event.get("type") != "dahai":
        return ""
    summarize = getattr(state, "reaction_summary", None)
    if not callable(summarize):
        return ""
    try:
        shanten, waits_mask, furiten = summarize(
            json.dumps(event, separators=(",", ":"))
        )
    except Exception:
        return ""
    parts = [_shanten_text(int(shanten))]
    waits = _wait_tiles(waits_mask) if int(shanten) == 0 else []
    if int(shanten) == 0:
        parts.append("waits " + (" ".join(waits) if waits else "none (dead wait)"))
        parts.append(f"furiten {'yes' if furiten else 'no'}")
    return " [after: " + "; ".join(parts) + "]"


def _real_time_shanten(state: Any) -> int:
    calculate = getattr(state, "real_time_shanten", None)
    if callable(calculate):
        return int(calculate())
    return int(getattr(state, "shanten", 0))


def _wait_tiles(mask: Any) -> list[str]:
    return [
        _prompt_tile(label_to_tile(index))
        for index, waiting in enumerate(mask)
        if bool(waiting)
    ]


def _shanten_text(shanten: int) -> str:
    if shanten < 0:
        return "complete hand"
    if shanten == 0:
        return "tenpai (0-shanten)"
    return f"{shanten}-shanten"
