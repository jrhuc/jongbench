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

NOTATION
Every concealed tile is written separately. Suited tiles use 1m-9m, 1p-9p, and 1s-9s.
Red fives are written as 5m(red), 5p(red), and 5s(red); they are still rank 5.
Honors: E S W N are winds; P F C are white, green, and red dragons.
A discard marked ' was drawn and immediately discarded; one marked * declared riichi.

HAND STRUCTURE
A standard complete hand has four melds and one pair; listed melds are already complete.
A meld is a run of three consecutive tiles in one suit or a triplet of identical tiles.
Runs cannot wrap 9-1 and honors form no runs. The exceptions to the four-melds-and-a-pair
shape are chiitoitsu (seven distinct pairs) and kokushi musou.

YAKU
A winning hand needs at least one yaku. Dora is not a yaku: a hand with only dora cannot win.
Closed only: riichi 1, ippatsu 1, menzen tsumo 1, pinfu 1, iipeiko 1, ryanpeikou 3,
chiitoitsu 2, and the closed-hand-only yakuman below.
Any hand: tanyao 1 (no terminals or honors), yakuhai 1 per set of dragons, seat wind, or
round wind, haitei/houtei 1, rinshan 1, chankan 1, toitoi 2, sanankou 2, sanshoku doukou 2,
sankantsu 2, honroutou 2, shousangen 2 (plus its yakuhai).
Losing one han when open: sanshoku doujun 2, ittsuu 2, chanta 2, junchan 3, honitsu 3,
chinitsu 6.
Yakuman: kokushi musou, suuankou, daisangen, shousuushii, daisuushii, tsuuiisou,
chinroutou, ryuuiisou, chuuren poutou, suukantsu.
A dora indicator is not itself dora; the next tile in its cycle is dora, wrapping 9 to 1,
E-S-W-N-E, and P-F-C-P. Red fives are each one extra dora.

WAITS
Ryanmen waits on both ends of two consecutive tiles (8 tiles live). Kanchan waits on the
inside of a gap and penchan on the one end a 12 or 89 partial run allows (4 tiles each).
Shanpon holds two pairs and waits to triple either (4 tiles). Tanki waits on a lone tile
to pair it (3 tiles). A ryanmen that would complete a run only outside 1-9 is still a
ryanmen. Chiitoitsu is always a tanki wait.

SCORING
Non-dealer ron: mangan 8000 at 5 han (also 4 han 30 fu and 3 han 60 fu), haneman 12000 at
6-7, baiman 16000 at 8-10, sanbaiman 24000 at 11-12, yakuman 32000 at 13+.
Dealer wins pay 1.5x those amounts and the dealer repeats. Each riichi stick on the table
goes to the winner; each honba adds 300 to a ron and 100 per player to a tsumo.
Fu below mangan starts at 20, plus 10 for a closed ron, 2 for a tsumo, and 2 for a kanchan,
penchan or tanki wait. Triplets add 2 open or 4 closed, doubled for terminals and honors,
and kans add four times the matching triplet. A yakuhai pair adds 2. The total rounds up to
the next 10. Pinfu is a flat 20 fu on ron and 30 by closed tsumo; chiitoitsu is always 25.

CALLS AND RIICHI
Chi, pon and open kan reveal the meld and end a closed hand: riichi, menzen tsumo, ippatsu
and pinfu are then unavailable, and the yaku above marked as losing a han are reduced.
Chi may only take the discard of the player to your left. Pon and kan may take any discard.
Closed kan keeps the hand closed. Every kan flips a new dora indicator.
Riichi requires a closed tenpai hand and 1000 points, and locks your discards: after
declaring you must discard every tile you draw unless it completes the hand or a closed kan
that does not change the wait.
Passing an available ron causes temporary furiten until your next discard, or permanent
furiten for the rest of the hand if you have declared riichi. You are also permanently
furiten while any tile of your own wait sits in your own discards.

DRAWS
When the wall runs out the hand is drawn. Players tenpai at that point are paid by those
who are not, splitting 3000 between them: one tenpai player takes 3000, two take 1500 each,
three take 1000 each. The dealer repeats if the dealer was tenpai. A player whose discards
are all terminals and honours, none of them called, instead scores nagashi mangan.
The hand also aborts when the first four discards are the same wind, and when a fourth kan
is declared by more than one player.

CALL POLICY
On your own turn you may set whether you are willing to call on other players' discards,
the way a client's furo toggle works. Add "calls":"off" to your reply to stop being
offered chi, pon and open kan, or "calls":"on" to resume. It stays as you set it until
you change it or the hand ends, and it never affects winning: ron and tsumo are always
offered, as are your own draws, closed kan and added kan.

THIS PROMPT
All listed actions are legal. A win option is listed only when the hand may legally win now.
Structural tenpai and waits do not by themselves guarantee a yaku or legal win.
Riichi is a two-step action: choose riichi first, then choose a legal discard in the next prompt.
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

    lines.append(render_hand(player_id, state, events))

    recent = [_event_words(ev) for ev in events[-3:]]
    if recent:
        lines.append("Last events: " + "; ".join(recent))

    return "\n".join(lines)


def render_hand(player_id: int, state: Any, events: list[dict[str, Any]]) -> str:
    """`events` need only reach back to this seat's last draw; the meld count comes from
    the state, so a delta slice renders the same line a full history would."""
    meld_count = len(state.chis) + len(state.pons) + len(state.minkans) + len(state.ankans)
    hand_tiles = tiles_from_counts(list(state.tehai), list(state.akas_in_hand))
    line = (
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
            line += f"; just drew: {_prompt_tile(str(drawn))}"
    return line


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
    calls_enabled: bool | None = None,
) -> str:
    lines = [
        render_state(player_id, state, events, state_hints=state_hints),
    ]
    if calls_enabled is not None:
        lines.append(call_policy_line(calls_enabled))
    lines.extend(
        ["", "Choose your action:", render_menu(menu, state=state, state_hints=state_hints)]
    )
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


def build_followup_prompt(
    player_id: int,
    state: Any,
    new_events: list[dict[str, Any]],
    menu: list[dict[str, Any]],
    error_feedback: str | None = None,
    *,
    state_hints: bool = False,
    calls_enabled: bool | None = None,
) -> str:
    """A later turn of a per-kyoku conversation: only what changed since this seat last
    acted. The opening turn carried the whole board, and the transcript above still holds
    it, so resending it would only cost tokens and break the cached prefix."""
    lines = []
    narrated = [_event_words(ev) for ev in new_events]
    lines.append("Since your last action: " + ("; ".join(narrated) if narrated else "nothing"))
    lines.append(render_hand(player_id, state, new_events))
    if state_hints:
        lines.extend(_state_hint_lines(state))
    if calls_enabled is not None:
        lines.append(call_policy_line(calls_enabled))
    lines.extend(
        ["", "Choose your action:", render_menu(menu, state=state, state_hints=state_hints)]
    )
    if error_feedback:
        lines.extend(
            [
                "",
                f"Your previous reply was invalid: {error_feedback}. Reply again, following the format.",
            ]
        )
    lines.extend(["", 'Reply with exactly: {"choice": N}'])
    return "\n".join(lines)


TOOLS_APPENDIX = """
TOOLS
Board-query tools replace inline state hints; call them when you need information.
board() re-renders the full table. discards(player) is one player's discard row.
waits() is your own shape: shanten, waits and furiten. simulate(tile) is the shape
after discarding that tile from your hand now. note(text) saves a private note that
survives into later hands of this match; notes() returns everything you have saved.
Tool results describe the current decision point only.
Call any tools first, then end with the same single JSON object reply."""


def decision_snapshot(
    player_id: int,
    state: Any,
    events: list[dict[str, Any]],
    menu: list[dict[str, Any]],
) -> dict[str, Any]:
    """Everything the board-query tools may be asked at this decision point, precomputed.
    The toolset answering the model runs in another process and cannot reach the live
    Rust state, so each answer is rendered here — from exactly the information the
    prompt path may use — and served from the rollout's state channel."""
    discard_rows, riichi = _discards_and_riichi(events)
    discards = {
        f"P{seat}": (
            f"riichi {'yes' if riichi[seat] else 'no'}; discards "
            + (" ".join(discard_rows[seat]) if discard_rows[seat] else "-")
        )
        for seat in range(4)
    }
    simulate: dict[str, str] = {}
    for item in menu:
        event = item.get("event")
        if not isinstance(event, dict) or event.get("type") != "dahai":
            continue
        summary = _after_summary(state, event)
        if summary:
            simulate[_prompt_tile(str(event.get("pai", "?")))] = summary
    return {
        "board": render_state(player_id, state, events),
        "discards": discards,
        "waits": shape_summary(state),
        "simulate": simulate,
    }


def call_policy_line(calls_enabled: bool) -> str:
    state = "accepting" if calls_enabled else "declining"
    other = "off" if calls_enabled else "on"
    return (
        f'Call policy: currently {state} calls on other players\' discards. '
        f'Add "calls":"{other}" to your reply to change it; ron and tsumo are unaffected.'
    )


def extract_call_policy(text: str) -> bool | None:
    """The seat's requested furo setting, or None if it did not ask to change one.
    Conflicting settings in one reply are ignored rather than guessed at."""
    found = set()
    for match in re.finditer(r'"calls"\s*:\s*"(on|off)"', text, flags=re.IGNORECASE):
        found.add(match.group(1).lower() == "on")
    if len(found) != 1:
        return None
    return found.pop()


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
    return [
        "Engine-derived state hints (structural only; no EV or recommendation):",
        "- " + shape_summary(state),
    ]


def shape_summary(state: Any) -> str:
    """The seat's own shape: shanten, waits when tenpai, furiten. One line of text."""
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
    return "; ".join(details)


def _action_hint(state: Any, item: dict[str, Any]) -> str:
    summary = _after_summary(state, item.get("event"))
    return f" [after: {summary}]" if summary else ""


def _after_summary(state: Any, event: Any) -> str | None:
    """Shanten/waits/furiten after playing a discard event, or None when unavailable."""
    if not isinstance(event, dict) or event.get("type") != "dahai":
        return None
    summarize = getattr(state, "reaction_summary", None)
    if not callable(summarize):
        return None
    try:
        shanten, waits_mask, furiten = summarize(
            json.dumps(event, separators=(",", ":"))
        )
    except Exception:
        return None
    parts = [_shanten_text(int(shanten))]
    if int(shanten) == 0:
        waits = _wait_tiles(waits_mask)
        parts.append("waits " + (" ".join(waits) if waits else "none (dead wait)"))
        parts.append(f"furiten {'yes' if furiten else 'no'}")
    return "; ".join(parts)


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
