"""Log post-processing that the harness already paid for: fingerprint, style, Q-loss."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from .tiles import deaka

_CALL_TYPES = frozenset({"chi", "pon", "daiminkan", "kakan", "ankan"})
_KIND_ALIASES = {
    "dahai": "discard",
    "reach": "riichi",
    "daiminkan": "kan",
    "ankan": "kan",
    "kakan": "kan",
}


def behavioral_fingerprint(
    events: Sequence[dict[str, Any]], player_id: int
) -> dict[str, float]:
    """Rates and values a seat actually produced in a finished log."""
    kyoku = 0
    wins = 0
    deal_ins = 0
    riichi = 0
    calls = 0
    discards = 0
    damaten = 0
    tenpai_draws = 0
    draws = 0
    win_values: list[float] = []
    deal_in_values: list[float] = []
    riichi_turns: list[float] = []
    fold_chances = 0
    folds = 0
    kyoku_deltas: list[float] = []
    declared_riichi = False
    closed = True
    opponent_riichi: set[int] = set()
    opponent_discards: dict[int, set[str]] = defaultdict(set)
    junme = 0

    def reset_hand() -> None:
        nonlocal declared_riichi, closed, junme
        declared_riichi = False
        closed = True
        opponent_riichi.clear()
        opponent_discards.clear()
        junme = 0

    for event in events:
        event_type = event.get("type")
        actor = event.get("actor")
        if event_type == "start_kyoku":
            kyoku += 1
            reset_hand()
            continue
        if event_type == "tsumo" and actor == player_id:
            junme += 1
            continue
        if event_type == "dahai":
            tile = deaka(str(event.get("pai", "")))
            if actor == player_id:
                discards += 1
                if opponent_riichi:
                    fold_chances += 1
                    if any(tile in opponent_discards[seat] for seat in opponent_riichi):
                        folds += 1
            elif isinstance(actor, int):
                opponent_discards[actor].add(tile)
            continue
        if event_type == "reach" and actor == player_id:
            riichi += 1
            declared_riichi = True
            riichi_turns.append(float(junme or 1))
            continue
        if event_type == "reach_accepted" and actor != player_id and isinstance(actor, int):
            opponent_riichi.add(actor)
            continue
        if event_type in _CALL_TYPES and actor == player_id:
            calls += 1
            if event_type in {"chi", "pon", "daiminkan"}:
                closed = False
            continue
        if event_type == "hora":
            deltas = event.get("deltas") or [0, 0, 0, 0]
            value = float(deltas[player_id]) if player_id < len(deltas) else 0.0
            kyoku_deltas.append(value)
            if actor == player_id:
                wins += 1
                win_values.append(value)
                if closed and not declared_riichi:
                    damaten += 1
            elif int(event.get("target", -1)) == player_id:
                deal_ins += 1
                deal_in_values.append(-value)
            continue
        if event_type == "ryukyoku":
            draws += 1
            deltas = event.get("deltas") or [0, 0, 0, 0]
            if player_id < len(deltas):
                kyoku_deltas.append(float(deltas[player_id]))
            if player_id < len(deltas) and int(deltas[player_id]) > 0:
                tenpai_draws += 1
            continue

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    hands = max(kyoku, 1)
    return {
        "kyoku": float(kyoku),
        "win_rate": wins / hands,
        "deal_in_rate": deal_ins / hands,
        "riichi_rate": riichi / hands,
        "call_rate": calls / hands,
        "tenpai_at_draw_rate": tenpai_draws / draws if draws else 0.0,
        "avg_win_value": mean(win_values),
        "avg_deal_in_value": mean(deal_in_values),
        "avg_riichi_turn": mean(riichi_turns),
        "damaten_rate": damaten / wins if wins else 0.0,
        "fold_rate": folds / fold_chances if fold_chances else 0.0,
        "discards": float(discards),
        "avg_kyoku_point_delta": mean(kyoku_deltas),
    }


def style_delta(review: MappingLike) -> dict[str, float]:
    """Model minus reviewer action-kind rates on the same reviewed boards."""
    model = Counter()
    reviewer = Counter()
    for entry in review.get("entries") or []:
        actual = _kind(entry.get("actual") or {})
        expected = _kind(entry.get("expected") or {})
        model[actual] += 1
        reviewer[expected] += 1
    n = max(sum(model.values()), 1)
    kinds = sorted(set(model) | set(reviewer) | {"discard", "riichi", "chi", "pon", "kan", "none"})
    return {
        kind: (model[kind] - reviewer[kind]) / n
        for kind in kinds
    }


def cumulative_q_loss(review: MappingLike) -> dict[str, float]:
    """Sum of best_q - chosen_q over reviewed decisions, in reviewer return units."""
    total = 0.0
    count = 0
    for entry in review.get("entries") or []:
        details = entry.get("details") or []
        index = entry.get("actual_index")
        if not details or not isinstance(index, int) or not 0 <= index < len(details):
            continue
        best = max(float(detail["q_value"]) for detail in details)
        chosen = float(details[index]["q_value"])
        total += best - chosen
        count += 1
    return {
        "q_loss": total,
        "q_loss_per_decision": total / count if count else 0.0,
        "decisions": float(count),
    }


def q_loss_of_choice(q_values: Sequence[float], choice: int) -> float:
    if not q_values:
        raise ValueError("q_values is empty")
    if not 0 <= choice < len(q_values):
        return max(q_values) - min(q_values)
    return max(q_values) - float(q_values[choice])


def calibrate_q_loss(
    rows: Sequence[MappingLike],
) -> dict[str, float]:
    """Ordinary least squares: realized_points ~ a + b * q_loss.

    ``rows`` are ``{"q_loss", "points"}`` observations, typically one per kyoku
    or per hanchan. A slope near zero is the finding, not a failure of the fit.
    """
    pairs = [
        (float(row["q_loss"]), float(row["points"]))
        for row in rows
        if "q_loss" in row and "points" in row
    ]
    n = len(pairs)
    if n < 2:
        raise ValueError("calibration needs at least two observations")
    mean_x = sum(x for x, _ in pairs) / n
    mean_y = sum(y for _, y in pairs) / n
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    if var_x == 0:
        raise ValueError("q_loss has zero variance")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for _, y in pairs)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in pairs)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {
        "n": float(n),
        "intercept": intercept,
        "slope": slope,
        "r2": r2,
        "points_per_q": slope,
    }


def _kind(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "none")
    return _KIND_ALIASES.get(event_type, event_type)


MappingLike = dict[str, Any]
