"""Dense diagnostics from logs and reviewer traces the harness already produced.

The functions in this module deliberately keep descriptive behaviour, reviewer
judgment, and realised outcomes separate.  A profile is useful only when a metric's
semantics are explicit enough that two runs can be compared without guessing what was
counted.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from .tiles import deaka

_CALL_TYPES = frozenset({"chi", "pon", "daiminkan", "kakan", "ankan"})
_OPEN_CALL_TYPES = frozenset({"chi", "pon", "daiminkan", "kakan"})
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
    """Return hand-level rates and values for one seat in a finished MJAI log.

    Deal-in rate and point delta are counted once per hand, including double-ron hands.
    ``fold_rate`` is intentionally the narrow, exactly checkable rate at which a discard
    after one or more opposing riichi declarations was genbutsu against *every* riichi
    player.  It is not a claim that every other discard was a push.
    """
    kyoku = 0
    wins = 0
    deal_in_hands = 0
    riichi = 0
    calls = 0
    opened_hands = 0
    discards = 0
    damaten = 0
    tenpai_draws = 0
    observed_draw_tenpai = 0
    draws = 0
    win_values: list[float] = []
    deal_in_values: list[float] = []
    riichi_turns: list[float] = []
    fold_chances = 0
    all_riichi_genbutsu = 0
    kyoku_deltas: list[float] = []

    active_hand = False
    declared_riichi = False
    closed = True
    opened_this_hand = False
    dealt_in_this_hand = False
    deal_in_loss = 0.0
    hand_delta = 0.0
    opponent_riichi: set[int] = set()
    opponent_discards: dict[int, set[str]] = defaultdict(set)
    junme = 0

    def flush_hand() -> None:
        nonlocal active_hand, opened_hands, deal_in_hands
        nonlocal dealt_in_this_hand, deal_in_loss, hand_delta
        if not active_hand:
            return
        kyoku_deltas.append(hand_delta)
        if opened_this_hand:
            opened_hands += 1
        if dealt_in_this_hand:
            deal_in_hands += 1
            deal_in_values.append(deal_in_loss)
        active_hand = False

    def reset_hand() -> None:
        nonlocal active_hand, declared_riichi, closed, opened_this_hand
        nonlocal dealt_in_this_hand, deal_in_loss, hand_delta, junme
        active_hand = True
        declared_riichi = False
        closed = True
        opened_this_hand = False
        dealt_in_this_hand = False
        deal_in_loss = 0.0
        hand_delta = 0.0
        opponent_riichi.clear()
        opponent_discards.clear()
        junme = 0

    for event in events:
        event_type = event.get("type")
        actor = event.get("actor")
        if event_type == "start_kyoku":
            flush_hand()
            kyoku += 1
            reset_hand()
            continue
        if not active_hand:
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
                    if all(
                        tile in opponent_discards[seat]
                        for seat in opponent_riichi
                    ):
                        all_riichi_genbutsu += 1
            elif isinstance(actor, int):
                opponent_discards[actor].add(tile)
            continue
        if event_type == "reach" and actor == player_id:
            riichi += 1
            declared_riichi = True
            riichi_turns.append(float(junme or 1))
            continue
        if (
            event_type == "reach_accepted"
            and actor != player_id
            and isinstance(actor, int)
        ):
            opponent_riichi.add(actor)
            continue
        if event_type in _CALL_TYPES and actor == player_id:
            calls += 1
            if event_type in _OPEN_CALL_TYPES:
                opened_this_hand = True
                closed = False
            continue
        if event_type == "hora":
            deltas = event.get("deltas") or [0, 0, 0, 0]
            value = float(deltas[player_id]) if player_id < len(deltas) else 0.0
            hand_delta += value
            if actor == player_id:
                wins += 1
                win_values.append(value)
                if closed and not declared_riichi:
                    damaten += 1
            elif int(event.get("target", -1)) == player_id:
                dealt_in_this_hand = True
                deal_in_loss += max(0.0, -value)
            continue
        if event_type == "ryukyoku":
            draws += 1
            deltas = event.get("deltas") or [0, 0, 0, 0]
            value = float(deltas[player_id]) if player_id < len(deltas) else 0.0
            hand_delta += value
            tenpai = _draw_tenpai(event, player_id, fallback_delta=value)
            if tenpai is not None:
                observed_draw_tenpai += 1
                if tenpai:
                    tenpai_draws += 1
            continue
        if event_type == "end_kyoku":
            flush_hand()

    flush_hand()

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    hands = max(kyoku, 1)
    exact_fold_rate = (
        all_riichi_genbutsu / fold_chances if fold_chances else 0.0
    )
    call_actions_per_hand = calls / hands
    return {
        "kyoku": float(kyoku),
        "win_rate": wins / hands,
        "deal_in_rate": deal_in_hands / hands,
        "riichi_rate": riichi / hands,
        # Kept for compatibility; the explicit name below is preferred because a
        # player can call more than once in a hand.
        "call_rate": call_actions_per_hand,
        "call_actions_per_hand": call_actions_per_hand,
        "open_hand_rate": opened_hands / hands,
        "tenpai_at_draw_rate": (
            tenpai_draws / observed_draw_tenpai if observed_draw_tenpai else 0.0
        ),
        "tenpai_at_draw_coverage": (
            observed_draw_tenpai / draws if draws else 0.0
        ),
        "avg_win_value": mean(win_values),
        "avg_deal_in_value": mean(deal_in_values),
        "avg_riichi_turn": mean(riichi_turns),
        "damaten_rate": damaten / wins if wins else 0.0,
        # Compatibility alias. This is exact genbutsu-against-all-riichi, not a
        # general push/fold classifier.
        "fold_rate": exact_fold_rate,
        "all_riichi_genbutsu_rate": exact_fold_rate,
        "discards": float(discards),
        "avg_kyoku_point_delta": mean(kyoku_deltas),
    }


def style_delta(review: MappingLike) -> dict[str, float]:
    """Model minus reviewer *distribution* by action kind on the same boards.

    Review entries normally carry softmax probabilities over every legal action.  Older
    reviews without those probabilities fall back to the reviewer's single expected
    action, preserving compatibility without silently changing current semantics.
    """
    model: Counter[str] = Counter()
    reviewer: Counter[str] = Counter()
    entries = list(review.get("entries") or [])
    for entry in entries:
        model[_kind(entry.get("actual") or {})] += 1.0
        details = list(entry.get("details") or [])
        probabilities = [
            max(0.0, float(detail.get("prob", 0.0))) for detail in details
        ]
        total_probability = sum(probabilities)
        if details and total_probability > 0.0:
            for detail, probability in zip(details, probabilities, strict=True):
                reviewer[_kind(detail.get("event") or {})] += (
                    probability / total_probability
                )
        else:
            reviewer[_kind(entry.get("expected") or {})] += 1.0
    n = max(len(entries), 1)
    kinds = sorted(
        set(model)
        | set(reviewer)
        | {"discard", "riichi", "chi", "pon", "kan", "none"}
    )
    return {kind: (model[kind] - reviewer[kind]) / n for kind in kinds}


def cumulative_q_loss(review: MappingLike) -> dict[str, float]:
    """Summarise reviewer regret without discarding the scale of each decision.

    Raw Q-loss is useful only after calibration establishes that the checkpoint's Q
    scale is comparable across positions.  The normalised companion is bounded per
    decision and the mean span exposes whether a run was dominated by low-stakes menus.
    """
    total = 0.0
    normalised_total = 0.0
    span_total = 0.0
    count = 0
    for entry in review.get("entries") or []:
        details = entry.get("details") or []
        index = entry.get("actual_index")
        if not details or not isinstance(index, int) or not 0 <= index < len(details):
            continue
        q_values = [float(detail["q_value"]) for detail in details]
        high = max(q_values)
        low = min(q_values)
        loss = high - q_values[index]
        span = high - low
        total += loss
        span_total += span
        if span > 0.0:
            normalised_total += loss / span
        count += 1
    return {
        "q_loss": total,
        "q_loss_per_decision": total / count if count else 0.0,
        "normalised_q_loss": normalised_total,
        "normalised_q_loss_per_decision": (
            normalised_total / count if count else 0.0
        ),
        "mean_q_span": span_total / count if count else 0.0,
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
    """Exploratory OLS fit: ``realised_points ~ a + b * q_loss``.

    This deliberately does not manufacture a confidence interval from correlated
    decisions.  Publication-grade calibration must fit on held-out games and bootstrap
    whole game/seed clusters; the returned residual diagnostics make that limitation
    visible to callers.
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
    residuals = [y - (intercept + slope * x) for x, y in pairs]
    ss_tot = sum((y - mean_y) ** 2 for _, y in pairs)
    ss_res = sum(value**2 for value in residuals)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    rmse = math.sqrt(ss_res / n)
    return {
        "n": float(n),
        "intercept": intercept,
        "slope": slope,
        "r2": r2,
        "rmse": rmse,
        "points_per_q": slope,
    }


def _draw_tenpai(
    event: dict[str, Any], player_id: int, *, fallback_delta: float
) -> bool | None:
    """Prefer explicit draw-tenpai metadata; return unknown for zero-delta legacy draws."""
    for key in ("tenpais", "tenpai"):
        values = event.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if player_id < len(values):
                value = values[player_id]
                if isinstance(value, bool):
                    return value
    # Non-zero noten payments identify the seat in the common one-to-three and
    # two-to-two cases. Zero means either all-tenpai or all-noten, so it is unknown.
    if fallback_delta > 0.0:
        return True
    if fallback_delta < 0.0:
        return False
    return None


def _kind(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "none")
    return _KIND_ALIASES.get(event_type, event_type)


MappingLike = dict[str, Any]
