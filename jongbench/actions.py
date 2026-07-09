from __future__ import annotations

import json
from typing import Any

from .tiles import akaize, deaka, fmt_tile, is_aka, label_to_tile, sort_tiles, tile_to_label
from .tiles import tiles_from_counts

MenuItem = dict[str, Any]

_AKA_INDEX = {"5m": 0, "5p": 1, "5s": 2}


def build_menu(state: Any) -> list[MenuItem]:
    cans = state.last_cans
    actor = int(state.player_id)
    target = int(cans.target_actor)
    mask = _action_mask(state, at_kan_select=False)
    menu: list[MenuItem] = []

    if cans.can_tsumo_agari:
        _add(menu, state, "tsumo (win)", {"type": "hora", "actor": actor, "target": target}, "hora")
    if cans.can_ron_agari:
        _add(menu, state, "ron (win)", {"type": "hora", "actor": actor, "target": target}, "hora")

    if cans.can_riichi and _mask_allows(mask, 37):
        _add(menu, state, "riichi", {"type": "reach", "actor": actor}, "riichi")

    if cans.can_discard:
        _add_discards(menu, state, actor, mask)

    pai = state.last_kawa_tile()
    if pai is not None:
        if cans.can_chi_low and _mask_allows(mask, 38):
            consumed = _chi_consumed(state, pai, "low")
            event = _call_event("chi", actor, target, pai, consumed)
            _add(menu, state, f"chi {fmt_tile(pai)} with {_tiles_text(consumed)}", event, "chi")
        if cans.can_chi_mid and _mask_allows(mask, 39):
            consumed = _chi_consumed(state, pai, "mid")
            event = _call_event("chi", actor, target, pai, consumed)
            _add(menu, state, f"chi {fmt_tile(pai)} with {_tiles_text(consumed)}", event, "chi")
        if cans.can_chi_high and _mask_allows(mask, 40):
            consumed = _chi_consumed(state, pai, "high")
            event = _call_event("chi", actor, target, pai, consumed)
            _add(menu, state, f"chi {fmt_tile(pai)} with {_tiles_text(consumed)}", event, "chi")

        if cans.can_pon and _mask_allows(mask, 41):
            for consumed in _pon_consumed_variants(state, pai):
                event = _call_event("pon", actor, target, pai, consumed)
                if _add(menu, state, f"pon {fmt_tile(pai)} with {_tiles_text(consumed)}", event, "pon"):
                    break

        if cans.can_daiminkan and _mask_allows(mask, 42):
            for consumed in _daiminkan_consumed_variants(state, pai):
                event = _call_event("daiminkan", actor, target, pai, consumed)
                label = f"daiminkan {fmt_tile(pai)} with {_tiles_text(consumed)}"
                if _add(menu, state, label, event, "daiminkan"):
                    break

    if cans.can_ankan and _mask_allows(mask, 42):
        for tile in state.ankan_candidates():
            base = deaka(tile)
            for consumed in _ankan_consumed_variants(state, base):
                event = {"type": "ankan", "actor": actor, "consumed": consumed}
                if _add(menu, state, f"ankan {fmt_tile(base)}", event, "ankan"):
                    break

    if cans.can_kakan and _mask_allows(mask, 42):
        for tile in state.kakan_candidates():
            base = deaka(tile)
            for pai2, consumed in _kakan_variants(state, base):
                event = {"type": "kakan", "actor": actor, "pai": pai2, "consumed": consumed}
                label = f"kakan {fmt_tile(pai2)}"
                if is_aka(pai2):
                    label += " (red)"
                if _add(menu, state, label, event, "kakan"):
                    break

    if cans.can_ryukyoku and _mask_allows(mask, 44):
        _add(menu, state, "abort hand (nine terminals)", {"type": "ryukyoku"}, "ryukyoku")

    if not cans.can_discard and bool(cans.can_pass) and _mask_allows(mask, 45):
        _add(menu, state, "pass", {"type": "none"}, "none")

    return menu


def _add(menu: list[MenuItem], state: Any, label: str, event: dict[str, Any], kind: str) -> bool:
    if not _valid(state, event):
        return False
    menu.append({"label": label, "event": event, "kind": kind})
    return True


def _valid(state: Any, event: dict[str, Any]) -> bool:
    try:
        state.validate_reaction(json.dumps(event, separators=(",", ":")))
    except Exception:
        return False
    return True


def _action_mask(state: Any, at_kan_select: bool) -> list[bool] | None:
    try:
        _, mask = state.encode_obs(2, at_kan_select)
        values = mask.tolist()
    except Exception:
        return None
    return [bool(value) for value in values]


def _mask_allows(mask: list[bool] | None, idx: int) -> bool:
    return mask is None or idx < len(mask) and mask[idx]


def _add_discards(menu: list[MenuItem], state: Any, actor: int, mask: list[bool] | None) -> None:
    drawn = state.last_self_tsumo()
    seen: set[str] = set()
    tiles = tiles_from_counts(list(state.tehai), list(state.akas_in_hand))
    for tile in sort_tiles(tiles):
        if tile in seen:
            continue
        seen.add(tile)
        if not _mask_allows(mask, tile_to_label(tile)):
            continue

        tsumogiri = tile == drawn
        event = {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": tsumogiri}
        label = f"discard {fmt_tile(tile)}"
        if is_aka(tile):
            label += " (red)"
        if tsumogiri:
            label += " (drawn)"
        _add(menu, state, label, event, "discard")


def _call_event(kind: str, actor: int, target: int, pai: str, consumed: list[str]) -> dict[str, Any]:
    return {"type": kind, "actor": actor, "target": target, "pai": pai, "consumed": consumed}


def _chi_consumed(state: Any, pai: str, which: str) -> list[str]:
    akas = list(state.akas_in_hand)
    if which == "low":
        first = _next_tile(pai)
        second = _next_tile(first)
        can_akaize = _has_chi_aka(akas, pai, {"3m", "4m"}, {"3p", "4p"}, {"3s", "4s"})
    elif which == "mid":
        first = _prev_tile(pai)
        second = _next_tile(pai)
        can_akaize = _has_chi_aka(akas, pai, {"4m", "6m"}, {"4p", "6p"}, {"4s", "6s"})
    else:
        second = _prev_tile(pai)
        first = _prev_tile(second)
        can_akaize = _has_chi_aka(akas, pai, {"6m", "7m"}, {"6p", "7p"}, {"6s", "7s"})

    if can_akaize:
        return [akaize(first), akaize(second)]
    return [first, second]


def _has_chi_aka(
    akas: list[bool],
    pai: str,
    man: set[str],
    pin: set[str],
    sou: set[str],
) -> bool:
    return (
        pai in man
        and akas[0]
        or pai in pin
        and akas[1]
        or pai in sou
        and akas[2]
    )


def _pon_consumed_variants(state: Any, pai: str) -> list[list[str]]:
    base = deaka(pai)
    if base in _AKA_INDEX and not is_aka(pai) and list(state.akas_in_hand)[_AKA_INDEX[base]]:
        return [[akaize(base), base], [base, base]]
    return [[base, base]]


def _daiminkan_consumed_variants(state: Any, pai: str) -> list[list[str]]:
    base = deaka(pai)
    if base not in _AKA_INDEX:
        return [[base, base, base]]
    if is_aka(pai):
        return [[base, base, base]]
    if list(state.akas_in_hand)[_AKA_INDEX[base]]:
        return [[akaize(base), base, base], [base, base, base]]
    return [[base, base, base], [akaize(base), base, base]]


def _ankan_consumed_variants(state: Any, tile: str) -> list[list[str]]:
    base = deaka(tile)
    if base in _AKA_INDEX:
        if list(state.akas_in_hand)[_AKA_INDEX[base]]:
            return [[akaize(base), base, base, base], [base, base, base, base]]
        return [[base, base, base, base], [akaize(base), base, base, base]]
    return [[base, base, base, base]]


def _kakan_variants(state: Any, tile: str) -> list[tuple[str, list[str]]]:
    base = deaka(tile)
    if base not in _AKA_INDEX:
        return [(base, [base, base, base])]
    if list(state.akas_in_hand)[_AKA_INDEX[base]]:
        return [(akaize(base), [base, base, base]), (base, [akaize(base), base, base])]
    return [(base, [akaize(base), base, base]), (base, [base, base, base])]


def _next_tile(tile: str) -> str:
    label = tile_to_label(deaka(tile))
    if label >= 27:
        raise ValueError(f"cannot get suit successor for {tile}")
    suit = label // 9
    num = label % 9
    return label_to_tile(suit * 9 + (num + 1) % 9)


def _prev_tile(tile: str) -> str:
    label = tile_to_label(deaka(tile))
    if label >= 27:
        raise ValueError(f"cannot get suit predecessor for {tile}")
    suit = label // 9
    num = label % 9
    return label_to_tile(suit * 9 + (num + 8) % 9)


def _tiles_text(tiles: list[str]) -> str:
    return "".join(fmt_tile(tile) for tile in tiles)
