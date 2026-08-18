"""Rule-derived competence tags and exact play checks.

These diagnostics do not consult a reviewer.  They isolate arithmetic and legality
from strategic judgment: same-shanten ukeire loss, needless shanten regression,
self-inflicted furiten, dead-wait tenpai, avoidable deal-in, and an unriichiable
closed tenpai with no inherent ron yaku.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from . import actions, prompts
from .bank_schema import COMPETENCE_TAGS
from .tiles import deaka, tile_to_label

_CALL_KINDS = frozenset({"chi", "pon", "daiminkan", "kakan", "ankan"})
_KAN_KINDS = frozenset({"daiminkan", "kakan", "ankan"})


def competence_tags(
    state: Any,
    menu: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> list[str]:
    """Tag a decision by the kind of judgment it asks for."""
    kinds = {str(item.get("kind")) for item in menu}
    tags: list[str] = []
    player_id = int(state.player_id)
    opponent_riichi = any(
        bool(flag)
        for seat, flag in enumerate(list(state.riichi_accepted))
        if seat != player_id
    )
    dangerous_table = opponent_riichi or _opponent_has_two_melds(events, player_id)
    can_discard = bool(getattr(state.last_cans, "can_discard", False))
    shanten = int(state.real_time_shanten())
    if "riichi" in kinds:
        tags.append("riichi_choice")
    if kinds & _CALL_KINDS:
        tags.append("call_choice")
    if kinds & _KAN_KINDS:
        tags.append("kan_choice")
    if bool(state.at_furiten):
        tags.append("furiten")
    if int(state.tiles_left) <= 8:
        tags.append("last_turns")
    if bool(state.is_all_last):
        tags.append("oorasu_placement")
    if can_discard and dangerous_table and shanten > 0:
        tags.append("pushfold")
    if can_discard and opponent_riichi and shanten >= 3:
        tags.append("defense_only")
    if can_discard and not dangerous_table and "riichi" not in kinds and shanten >= 1:
        tags.append("efficiency")
    ordered = [tag for tag in COMPETENCE_TAGS if tag in tags]
    ordered.extend(sorted(set(tags) - set(COMPETENCE_TAGS)))
    return ordered


def analyze_choice(
    state: Any, menu: Sequence[dict[str, Any]], chosen: dict[str, Any]
) -> dict[str, Any]:
    """Compare one chosen action to legal alternatives on the same board.

    Ukeire is meaningful only between alternatives that leave the same best shanten.
    Comparing a one-shanten shape's acceptance count to a two-shanten shape's count is
    a category error, so shanten regression and ukeire loss are deliberately separate.
    """
    analyses = [_analyze_item(state, item) for item in menu]
    chosen_index = _index_of(menu, chosen)
    chosen_analysis = analyses[chosen_index] if chosen_index is not None else None
    discard_rows = [
        row
        for row in analyses
        if row is not None
        and row["kind"] == "discard"
        and row["shanten"] is not None
    ]
    best_shanten = min((int(row["shanten"]) for row in discard_rows), default=None)
    best_rows = (
        [row for row in discard_rows if int(row["shanten"]) == best_shanten]
        if best_shanten is not None
        else []
    )
    best_ukeire = max((int(row["ukeire"]) for row in best_rows), default=None)
    live_wait_exists = any(
        int(row["shanten"]) == 0 and int(row["ukeire"]) > 0
        for row in discard_rows
    )
    unfuriten_tenpai_exists = any(
        int(row["shanten"]) == 0 and not bool(row["furiten"])
        for row in discard_rows
    )
    can_riichi = any(item.get("kind") == "riichi" for item in menu)
    flags: dict[str, int | bool] = {
        "ukeire_loss": 0,
        "needless_shanten_regression": False,
        "self_inflicted_furiten": False,
        "dead_wait_tenpai": False,
        "yakuless_tenpai": False,
    }
    if (
        chosen_analysis is not None
        and chosen_analysis["kind"] == "discard"
        and chosen_analysis["shanten"] is not None
    ):
        chosen_shanten = int(chosen_analysis["shanten"])
        if best_shanten is not None and chosen_shanten > best_shanten:
            flags["needless_shanten_regression"] = True
        elif best_ukeire is not None and chosen_shanten == best_shanten:
            flags["ukeire_loss"] = max(
                0, int(best_ukeire) - int(chosen_analysis["ukeire"])
            )
        if (
            chosen_shanten == 0
            and bool(chosen_analysis["furiten"])
            and unfuriten_tenpai_exists
        ):
            flags["self_inflicted_furiten"] = True
        if (
            chosen_shanten == 0
            and int(chosen_analysis["ukeire"]) == 0
            and live_wait_exists
        ):
            flags["dead_wait_tenpai"] = True
        # ``wait_has_yaku`` is intentionally a ron-yaku check. Menzen tsumo may
        # still win; this flag means the closed tenpai cannot ron as-is and the
        # menu offered no riichi action to repair that.
        if (
            chosen_shanten == 0
            and bool(chosen_analysis["menzen"])
            and not bool(chosen_analysis["wait_has_yaku"])
            and not can_riichi
        ):
            flags["yakuless_tenpai"] = True
    return {
        "chosen": chosen_analysis,
        "alternatives": analyses,
        **flags,
    }


def scorecard(events: Sequence[dict[str, Any]], player_id: int) -> dict[str, Any]:
    """Exact rule checks for one seat across a finished log.

    The state is advanced once through the log.  The previous implementation rebuilt
    it from the prefix at every discard, making this pass quadratic in episode length.
    """
    import librichi

    totals = {
        "decisions": 0,
        "discards": 0,
        "ukeire_loss": 0,
        "needless_shanten_regression": 0,
        "self_inflicted_furiten": 0,
        "dead_wait_tenpai": 0,
        "yakuless_tenpai": 0,
        "avoidable_deal_in": 0,
        "deal_in": 0,
    }
    state = libriichi.state.PlayerState(player_id)
    pending_discard: dict[str, Any] | None = None
    for index, event in enumerate(events):
        event_type = event.get("type")
        if event_type == "start_kyoku":
            pending_discard = None
        elif event_type == "dahai" and int(event.get("actor", -1)) == player_id:
            menu = actions.build_menu(state)
            if len(menu) >= 2:
                analysis = analyze_choice(state, menu, event)
                totals["decisions"] += 1
                totals["discards"] += 1
                totals["ukeire_loss"] += int(analysis["ukeire_loss"])
                for key in (
                    "needless_shanten_regression",
                    "self_inflicted_furiten",
                    "dead_wait_tenpai",
                    "yakuless_tenpai",
                ):
                    totals[key] += int(bool(analysis[key]))
            pending_discard = {"event": event, "menu": menu}
        elif event_type == "hora" and int(event.get("target", -1)) == player_id:
            if int(event.get("actor", -1)) == player_id:
                pending_discard = None
            else:
                totals["deal_in"] += 1
                if pending_discard is not None and _avoidable_deal_in(
                    pending_discard["menu"],
                    pending_discard["event"],
                    int(event["actor"]),
                    events[: index + 1],
                ):
                    totals["avoidable_deal_in"] += 1
                pending_discard = None
        elif event_type in {"hora", "ryukyoku", "end_kyoku"}:
            pending_discard = None
        state.update(json.dumps(event, separators=(",", ":")))

    discards = max(int(totals["discards"]), 1)
    rates = {
        "ukeire_loss_per_discard": totals["ukeire_loss"] / discards,
        "needless_shanten_regression_rate": totals["needless_shanten_regression"]
        / discards,
        "self_inflicted_furiten_rate": totals["self_inflicted_furiten"] / discards,
        "dead_wait_tenpai_rate": totals["dead_wait_tenpai"] / discards,
        "yakuless_tenpai_rate": totals["yakuless_tenpai"] / discards,
        "avoidable_deal_in_rate": (
            totals["avoidable_deal_in"] / totals["deal_in"]
            if totals["deal_in"]
            else 0.0
        ),
    }
    return {**totals, **rates}


def _analyze_item(state: Any, item: dict[str, Any]) -> dict[str, Any] | None:
    event = item.get("event")
    if not isinstance(event, dict):
        return None
    kind = str(item.get("kind") or event.get("type") or "none")
    if event.get("type") != "dahai":
        return {
            "kind": kind,
            "shanten": None,
            "ukeire": 0,
            "furiten": False,
            "wait_has_yaku": False,
            "menzen": bool(getattr(state, "is_menzen", False)),
        }
    analyze = getattr(state, "reaction_analysis", None)
    if not callable(analyze):
        raise RuntimeError(
            "exact scorecard analysis requires PlayerState.reaction_analysis "
            "from jongbench core >=0.1.1"
        )
    shanten, _waits, furiten, ukeire, wait_has_yaku, menzen = analyze(
        json.dumps(event, separators=(",", ":"))
    )
    return {
        "kind": kind,
        "shanten": int(shanten),
        "ukeire": int(ukeire),
        "furiten": bool(furiten),
        "wait_has_yaku": bool(wait_has_yaku),
        "menzen": bool(menzen),
    }


def _index_of(menu: Sequence[dict[str, Any]], chosen: dict[str, Any]) -> int | None:
    from . import evaluate

    for index, item in enumerate(menu):
        event = item.get("event")
        if isinstance(event, dict) and evaluate.equal_ignore_aka_consumed(event, chosen):
            return index
    return None


def _avoidable_deal_in(
    menu: Sequence[dict[str, Any]],
    discarded: dict[str, Any],
    winner: int,
    events: Sequence[dict[str, Any]],
) -> bool:
    kyoku = prompts.this_kyoku(list(events))
    genbutsu = {
        deaka(str(event["pai"]))
        for event in kyoku
        if event.get("type") == "dahai" and int(event.get("actor", -1)) == winner
    }
    discarded_tile = deaka(str(discarded.get("pai", "")))
    if discarded_tile in genbutsu:
        return False
    for item in menu:
        if item.get("kind") != "discard":
            continue
        event = item.get("event") or {}
        tile = deaka(str(event.get("pai", "")))
        if tile in genbutsu:
            try:
                tile_to_label(str(event.get("pai", "")))
            except ValueError:
                continue
            return True
    return False


def _opponent_has_two_melds(events: Sequence[dict[str, Any]], player_id: int) -> bool:
    counts = [0, 0, 0, 0]
    for event in prompts.this_kyoku(list(events)):
        actor = event.get("actor")
        if actor is None or int(actor) == player_id:
            continue
        if event.get("type") in {"chi", "pon", "daiminkan", "ankan"}:
            counts[int(actor)] += 1
    return any(count >= 2 for count in counts)
