from __future__ import annotations

import gzip
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

import libriichi
from jongbench.mortal_engine import MortalEngine
from jongbench.mortal_model import DQN, AuxNet, Brain, ConfidenceHead, PolicyHead

TILES = [
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
    "5mr",
    "5pr",
    "5sr",
]
TILE_TO_LABEL = {tile: idx for idx, tile in enumerate(TILES)}
AKA_TO_BASE = {"5mr": "5m", "5pr": "5p", "5sr": "5s"}
BASE_TO_AKA = {"5m": "5mr", "5p": "5pr", "5s": "5sr"}


def resolve_device(name: str | torch.device | None = "cpu") -> torch.device:
    if name is None or name == "cpu":
        return torch.device("cpu")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(
    weights_path: str, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    return torch.load(weights_path, weights_only=True, map_location=map_location)


def networks_from_checkpoint(
    ckpt: dict[str, Any],
) -> tuple[
    Brain,
    DQN,
    PolicyHead | None,
    AuxNet | None,
    ConfidenceHead | None,
    int,
]:
    cfg = ckpt["config"]
    version = cfg["control"].get("version", 1)
    brain = Brain(
        version=version,
        num_blocks=cfg["resnet"]["num_blocks"],
        conv_channels=cfg["resnet"]["conv_channels"],
    )
    dqn = DQN(version=version)
    brain.load_state_dict(ckpt["mortal"])
    dqn.load_state_dict(ckpt["current_dqn"])

    policy = None
    if "policy" in ckpt:
        policy = PolicyHead()
        policy.load_state_dict(ckpt["policy"])

    aux_net = None
    if "aux_net" in ckpt:
        aux_net = AuxNet((4,))
        aux_net.load_state_dict(ckpt["aux_net"])

    confidence = None
    if "confidence" in ckpt:
        confidence = ConfidenceHead()
        confidence.load_state_dict(ckpt["confidence"])

    return (
        brain.eval(),
        dqn.eval(),
        policy.eval() if policy is not None else None,
        aux_net.eval() if aux_net is not None else None,
        confidence.eval() if confidence is not None else None,
        version,
    )


def load_engine(
    weights_path: str,
    *,
    device: str | torch.device | None = None,
    use_policy: bool = False,
    enable_quick_eval: bool = False,
    enable_amp: bool | None = None,
    enable_rule_based_agari_guard: bool = True,
    boltzmann_epsilon: float = 0,
    boltzmann_temp: float = 1,
    name: str = "mortal",
) -> MortalEngine:
    # Single-position Mortal inference regresses above two intra-op CPU threads.
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    ckpt = load_checkpoint(weights_path, map_location="cpu")
    default_device = "auto" if "policy" in ckpt else "cpu"
    device_t = resolve_device(default_device if device is None else device)
    brain, dqn, policy, aux_net, confidence, version = networks_from_checkpoint(ckpt)
    if enable_amp is None:
        enable_amp = device_t.type == "cuda"

    return MortalEngine(
        brain,
        dqn,
        is_oracle=False,
        device=device_t,
        enable_amp=enable_amp,
        enable_quick_eval=enable_quick_eval,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        name=name,
        version=version,
        policy=policy,
        aux_net=aux_net,
        confidence=confidence,
        use_policy=use_policy,
        boltzmann_epsilon=boltzmann_epsilon,
        boltzmann_temp=boltzmann_temp,
    )


def review_player(
    events: list[dict[str, Any]],
    player_id: int,
    engine: MortalEngine,
    temperature: float = 0.1,
    *,
    _lines: list[str] | None = None,
    _reactions: list[str | None] | None = None,
) -> dict[str, Any]:
    if engine.use_policy:
        raise ValueError(
            "review requires Q-value play mode; load with use_policy=False"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if _lines is not None and len(_lines) != len(events):
        raise ValueError("serialized event count does not match events")
    if _reactions is not None and len(_reactions) != len(events):
        raise ValueError("reaction count does not match events")
    bot = None if _reactions is not None else libriichi.mjai.Bot(engine, player_id)
    state = libriichi.state.PlayerState(player_id)

    total_reviewed = 0
    total_matches = 0
    raw_rating = 0.0
    kyokus: list[dict[str, Any]] = []
    all_entries: list[dict[str, Any]] = []
    current_entries: list[dict[str, Any]] = []
    policy_jobs: list[dict[str, Any]] = []
    kyoku_review = _new_kyoku_review()

    junme = 0
    tiles_left = 70
    last_tsumo_or_discard: str | None = None
    last_actor = 0

    for i, event in enumerate(events):
        line = (
            _lines[i]
            if _lines is not None
            else json.dumps(event, separators=(",", ":"))
        )
        reaction = _reactions[i] if _reactions is not None else bot.react(line)
        state.update(line)

        event_type = event.get("type")
        if event_type == "start_kyoku":
            kyoku_review["kyoku"] = (
                (tile_to_label(event["bakaze"]) - tile_to_label("E")) * 4
                + int(event["kyoku"])
                - 1
            )
            kyoku_review["honba"] = int(event["honba"])
            scores = list(event["scores"])
            kyoku_review["relative_scores"] = scores[player_id:] + scores[:player_id]
            tiles_left = 70
        elif event_type == "end_kyoku":
            kyoku_review["entries"] = current_entries
            kyokus.append(kyoku_review)
            kyoku_review = _new_kyoku_review()
            current_entries = []
            junme = 0
        elif event_type in {"hora", "ryukyoku"}:
            kyoku_review["end_status"].append(_strip_meta(event))
        elif event_type == "tsumo":
            if event.get("actor") == player_id:
                last_tsumo_or_discard = event["pai"]
                junme += 1
            tiles_left -= 1
        elif event_type in {"chi", "pon"} and event.get("actor") == player_id:
            junme += 1
        elif event_type in {"dahai", "kakan"}:
            last_tsumo_or_discard = event["pai"]

        actor = event.get("actor")
        if actor is not None:
            last_actor = int(actor)

        if event_type in {"start_game", "start_kyoku", "end_kyoku", "end_game"}:
            continue
        if reaction is None:
            continue

        output = json.loads(reaction)
        meta = output.get("meta")
        if meta is None:
            continue

        mask_bits = int(meta["mask_bits"])
        if mask_bits.bit_count() <= 1:
            continue

        masks = masks_from_bits(mask_bits)
        can_pon_or_daiminkan = masks[41] or masks[42]
        can_agari = masks[43]
        can_ryukyoku = masks[44]

        actual = next_action(
            events[i + 1 :],
            player_id,
            can_pon_or_daiminkan,
            can_agari,
            can_ryukyoku,
        )
        if actual is None:
            continue

        actual_label = to_label(actual)
        if not masks[actual_label]:
            raise ValueError(f"{actual!r} is not a valid reaction")

        shanten = int(meta["shanten"])
        at_furiten = bool(meta["at_furiten"])
        q_values = list(meta["q_values"])
        details: list[dict[str, Any]] = []
        min_q = math.inf
        max_q = -math.inf
        actual_q_value: float | None = None

        for label in range(45, -1, -1):
            if not masks[label]:
                continue
            if not q_values:
                raise ValueError("q_values vec underflow")
            q_value = float(q_values.pop())
            min_q = min(min_q, q_value)
            max_q = max(max_q, q_value)
            action = to_event(state, label, last_actor, last_tsumo_or_discard, False)
            if label == actual_label:
                actual_q_value = q_value
            details.append(
                {
                    "event": action,
                    "q_value": q_value,
                    "prob": 0.0,
                    "_label": ("general", label),
                }
            )

        actual_kan_label = to_kan_label(actual)
        kan_select = meta.get("kan_select")
        if kan_select is not None:
            kan_mask_bits = int(kan_select["mask_bits"])
            num_kans = kan_mask_bits.bit_count()
            if num_kans <= 0:
                raise ValueError(
                    f"expected `num_kans > 0`, got mask_bits = {kan_mask_bits}"
                )

            orig_kan_idx = next(
                (
                    idx
                    for idx, detail in enumerate(details)
                    if detail["event"]["type"] == "ankan"
                ),
                None,
            )
            if orig_kan_idx is None:
                raise ValueError("in kan_select but no kan found in root")
            orig_kan_q_value = float(details[orig_kan_idx]["q_value"])
            orig_kan_kind, orig_kan_label = details[orig_kan_idx]["_label"]
            if orig_kan_kind != "general":
                raise ValueError("kan root action is not in the general action space")
            del details[orig_kan_idx]

            kan_masks = masks_from_bits(kan_mask_bits)
            kan_q_values = list(kan_select.get("q_values") or [])
            for kan_label in range(45, -1, -1):
                if not kan_masks[kan_label]:
                    continue
                if num_kans == 1:
                    q_value = orig_kan_q_value
                else:
                    if not kan_q_values:
                        raise ValueError("q_values vec underflow")
                    q_value = float(kan_q_values.pop())
                min_q = min(min_q, q_value)
                max_q = max(max_q, q_value)
                action = to_event(
                    state,
                    kan_label,
                    last_actor,
                    last_tsumo_or_discard,
                    True,
                )
                if (
                    num_kans > 1
                    and actual_kan_label is not None
                    and actual_kan_label == kan_label
                ):
                    actual_q_value = q_value
                details.append(
                    {
                        "event": action,
                        "q_value": q_value,
                        "prob": 0.0,
                        "_label": ("kan_select", kan_label),
                        "_root_label": int(orig_kan_label),
                    }
                )

        probs = softmax([float(detail["q_value"]) for detail in details], temperature)
        for detail, prob in zip(details, probs, strict=True):
            detail["prob"] = prob
        policy_inputs = _capture_policy_inputs(engine, state, details)
        details.sort(key=lambda detail: detail["q_value"], reverse=True)
        actual_index = _actual_index(details, actual_label, actual_kan_label)
        if actual_index is None:
            raise ValueError(
                f"failed to find action ({actual_label}, {actual_kan_label!r})"
            )

        expected = _strip_meta(output)
        is_equal = equal_ignore_aka_consumed(expected, actual)
        if actual_q_value is None:
            raise ValueError(f"failed to find q value of actual action {actual!r}")
        if is_equal:
            raw_rating += 1.0
            total_matches += 1
        else:
            raw_rating += (actual_q_value - min_q) / max(max_q - min_q, 1e-6)

        total_reviewed += 1

        if last_tsumo_or_discard is None:
            raise ValueError("missing last tsumo or discard")

        public_details = []
        for detail in details:
            item = {
                "event": detail["event"],
                "q_value": detail["q_value"],
                "prob": detail["prob"],
            }
            if "policy_prob" in detail:
                item["policy_prob"] = detail["policy_prob"]
            public_details.append(item)
        entry = {
            # Index into `events` of the decision point, so a caller can replay the log
            # to this action and rebuild the board without redoing the review.
            "event_index": i,
            "kyoku": kyoku_review["kyoku"],
            "honba": kyoku_review["honba"],
            "junme": junme,
            "tiles_left": tiles_left,
            "last_actor": last_actor,
            "tile": last_tsumo_or_discard,
            "actual": _strip_meta(actual),
            "expected": expected,
            "is_equal": is_equal,
            "actual_index": actual_index,
            "shanten": shanten,
            "at_furiten": at_furiten,
            "details": public_details,
        }
        if policy_inputs is not None:
            policy_jobs.append(
                {
                    "entry": entry,
                    "details": details,
                    "actual_index": actual_index,
                    "inputs": policy_inputs,
                }
            )
        current_entries.append(entry)
        all_entries.append(entry)

    _attach_policy_probs_batch(engine, policy_jobs)
    rating = (raw_rating / total_reviewed) ** 2 if total_reviewed else 0.0
    return {
        "rating": rating,
        "total_reviewed": total_reviewed,
        "total_matches": total_matches,
        "temperature": temperature,
        "entries": all_entries,
        "kyokus": kyokus,
    }


def review_game(
    events: list[dict[str, Any]],
    engine: MortalEngine,
    temperature: float = 0.1,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> dict[int, dict[str, Any]]:
    lines = [json.dumps(event, separators=(",", ":")) for event in events]
    bot = libriichi.mjai.BatchBot(engine, [0, 1, 2, 3])
    reactions: list[list[str | None]] = [[] for _ in range(4)]
    for line in lines:
        if check_cancelled is not None:
            check_cancelled()
        for player_id, reaction in enumerate(bot.react(line)):
            reactions[player_id].append(reaction)
    return {
        player_id: review_player(
            events,
            player_id,
            engine,
            temperature,
            _lines=lines,
            _reactions=reactions[player_id],
        )
        for player_id in range(4)
    }


def aggregates(review: dict[str, Any]) -> dict[str, Any]:
    entries = list(review.get("entries", []))
    losses = [_prob_loss(entry) for entry in entries]
    policy_entries = [entry for entry in entries if "policy_loss" in entry]
    policy_losses = [float(entry["policy_loss"]) for entry in policy_entries]
    policy_nlls = [
        -math.log(max(float(entry["policy_actual_prob"]), 1e-12))
        for entry in policy_entries
    ]
    total_reviewed = int(review.get("total_reviewed", 0))
    total_matches = int(review.get("total_matches", 0))
    kinds = ["dahai", "reach", "chi", "pon", "kan", "hora", "ryukyoku", "none"]
    by_kind: dict[str, dict[str, Any]] = {
        kind: {"count": 0, "matches": 0, "mean_loss": 0.0} for kind in kinds
    }
    loss_sums = {kind: 0.0 for kind in kinds}

    for entry, loss in zip(entries, losses, strict=True):
        kind = _kind(entry["actual"])
        if kind not in by_kind:
            by_kind[kind] = {"count": 0, "matches": 0, "mean_loss": 0.0}
            loss_sums[kind] = 0.0
        by_kind[kind]["count"] += 1
        by_kind[kind]["matches"] += int(bool(entry["is_equal"]))
        loss_sums[kind] += loss

    for kind, stats in by_kind.items():
        count = int(stats["count"])
        stats["mean_loss"] = loss_sums[kind] / count if count else 0.0

    worst = []
    for entry, loss in sorted(
        zip(entries, losses, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:10]:
        worst.append(
            {
                "kyoku": entry["kyoku"],
                "junme": entry["junme"],
                "actual": entry["actual"],
                "expected": entry["expected"],
                "loss": loss,
            }
        )

    policy_count = len(policy_entries)
    policy_matches = sum(
        int(bool(entry["policy_is_equal"])) for entry in policy_entries
    )
    confidences = [
        float(entry["policy_confidence"])
        for entry in policy_entries
        if "policy_confidence" in entry
    ]
    return {
        "match_rate": total_matches / total_reviewed if total_reviewed else 0.0,
        "mean_prob_loss": sum(losses) / len(losses) if losses else 0.0,
        "mean_q_weight_loss": sum(losses) / len(losses) if losses else 0.0,
        "policy_count": policy_count,
        "policy_match_rate": (policy_matches / policy_count if policy_count else None),
        "mean_policy_loss": (
            sum(policy_losses) / policy_count if policy_count else None
        ),
        "mean_policy_nll": sum(policy_nlls) / policy_count if policy_count else None,
        "mean_policy_confidence": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        "worst": worst,
        "by_kind": by_kind,
    }


def load_mjai_log(path: str) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if path_obj.suffix == ".gz":
        with gzip.open(path_obj, "rt", encoding="utf-8") as f:
            text = f.read()
    else:
        text = path_obj.read_text(encoding="utf-8")

    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] == "[":
        loaded = json.loads(stripped)
        if not isinstance(loaded, list):
            raise ValueError("expected a JSON array log")
        return loaded
    return [json.loads(line) for line in stripped.splitlines() if line.strip()]


def _new_kyoku_review() -> dict[str, Any]:
    return {
        "kyoku": 0,
        "honba": 0,
        "end_status": [],
        "relative_scores": [0, 0, 0, 0],
        "entries": [],
    }


def _strip_meta(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "meta"}


def masks_from_bits(bits: int) -> list[bool]:
    return [((bits >> idx) & 1) == 1 for idx in range(46)]


def next_action(
    events: list[dict[str, Any]],
    player_id: int,
    can_pon_or_daiminkan: bool,
    can_agari: bool,
    can_ryukyoku: bool,
) -> dict[str, Any] | None:
    if not events:
        return None

    event = events[0]
    event_type = event.get("type")
    if event_type in {"dora", "reach_accepted"}:
        return next_action(
            events[1:],
            player_id,
            can_pon_or_daiminkan,
            can_agari,
            can_ryukyoku,
        )
    if event_type == "tsumo":
        return {"type": "none"}
    if event_type == "hora":
        for action in events[:3]:
            if action.get("type") == "hora" and action.get("actor") == player_id:
                return _strip_meta(action)
        return {"type": "none"} if can_agari else None
    if event_type == "ryukyoku":
        return _strip_meta(event) if can_ryukyoku else None

    actor = event.get("actor")
    if actor is not None and actor != player_id:
        if can_agari or can_pon_or_daiminkan:
            return {"type": "none"}
        return None
    return _strip_meta(event)


def to_label(event: dict[str, Any]) -> int:
    event_type = event.get("type")
    if event_type == "dahai":
        return tile_to_label(event["pai"])
    if event_type == "reach":
        return 37
    if event_type == "chi":
        consumed = event["consumed"]
        a = tile_to_label(deaka(consumed[0]))
        b = tile_to_label(deaka(consumed[1]))
        low = min(a, b)
        high = max(a, b)
        called = tile_to_label(deaka(event["pai"]))
        if called < low:
            return 38
        if called < high:
            return 39
        return 40
    if event_type == "pon":
        return 41
    if event_type in {"daiminkan", "ankan", "kakan"}:
        return 42
    if event_type == "hora":
        return 43
    if event_type == "ryukyoku":
        return 44
    return 45


def to_kan_label(event: dict[str, Any]) -> int | None:
    event_type = event.get("type")
    if event_type == "ankan":
        return tile_to_label(deaka(event["consumed"][0]))
    if event_type == "kakan":
        return tile_to_label(deaka(event["pai"]))
    return None


def to_event(
    state: Any,
    label: int,
    target: int,
    last_tsumo_or_discard: str | None,
    at_kan_select: bool,
) -> dict[str, Any]:
    actor = int(state.player_id)

    if at_kan_select:
        if label >= 34:
            raise ValueError(f"invalid kan label {label}")
        tile = label_to_tile(label)
        return {"type": "ankan", "actor": actor, "consumed": [tile] * 4}

    if 0 <= label <= 36:
        tile = label_to_tile(label)
        return {
            "type": "dahai",
            "actor": actor,
            "pai": tile,
            "tsumogiri": last_tsumo_or_discard == tile,
        }
    if label == 37:
        return {"type": "reach", "actor": actor}
    if label == 38:
        pai = require_tile(last_tsumo_or_discard, "missing last discard for Chi")
        if pai in {"3m", "4m", "3p", "4p", "3s", "4s"} and has_red_for(pai, state):
            consumed = [akaize(next_tile(pai)), akaize(next_tile(next_tile(pai)))]
        else:
            consumed = [next_tile(pai), next_tile(next_tile(pai))]
        return {
            "type": "chi",
            "actor": actor,
            "target": target,
            "pai": pai,
            "consumed": consumed,
        }
    if label == 39:
        pai = require_tile(last_tsumo_or_discard, "missing last discard for Chi")
        if pai in {"4m", "6m", "4p", "6p", "4s", "6s"} and has_red_for(pai, state):
            consumed = [akaize(prev_tile(pai)), akaize(next_tile(pai))]
        else:
            consumed = [prev_tile(pai), next_tile(pai)]
        return {
            "type": "chi",
            "actor": actor,
            "target": target,
            "pai": pai,
            "consumed": consumed,
        }
    if label == 40:
        pai = require_tile(last_tsumo_or_discard, "missing last discard for Chi")
        if pai in {"6m", "7m", "6p", "7p", "6s", "7s"} and has_red_for(pai, state):
            consumed = [akaize(prev_tile(prev_tile(pai))), akaize(prev_tile(pai))]
        else:
            consumed = [prev_tile(prev_tile(pai)), prev_tile(pai)]
        return {
            "type": "chi",
            "actor": actor,
            "target": target,
            "pai": pai,
            "consumed": consumed,
        }
    if label == 41:
        pai = require_tile(last_tsumo_or_discard, "missing last discard for Pon")
        if pai in {"5m", "5p", "5s"} and has_red_for(pai, state):
            consumed = [akaize(pai), deaka(pai)]
        else:
            consumed = [deaka(pai), deaka(pai)]
        return {
            "type": "pon",
            "actor": actor,
            "target": target,
            "pai": pai,
            "consumed": consumed,
        }
    if label == 42:
        tile = require_tile(last_tsumo_or_discard, "missing last discard for Daiminkan")
        return {
            "type": "ankan",
            "actor": actor,
            "consumed": [tile, deaka(tile), deaka(tile), deaka(tile)],
        }
    if label == 43:
        return {"type": "hora", "actor": actor, "target": target}
    if label == 44:
        return {"type": "ryukyoku"}
    if label == 45:
        return {"type": "none"}
    raise ValueError(f"unexpected label {label}")


def equal_ignore_aka_consumed(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_type = a.get("type")
    b_type = b.get("type")
    if a_type != b_type:
        return False
    if a_type in {"dahai", "kakan"}:
        return a.get("pai") == b.get("pai")
    if a_type in {"chi", "pon", "daiminkan", "ankan"}:
        return sorted(map(deaka, a.get("consumed", []))) == sorted(
            map(deaka, b.get("consumed", []))
        )
    return a_type in {"reach", "hora", "ryukyoku", "none"}


def _capture_policy_inputs(
    engine: MortalEngine, state: Any, details: list[dict[str, Any]]
) -> dict[bool, tuple[Any, Any]] | None:
    if engine.policy is None or not details:
        return None
    needs_kan = any(detail["_label"][0] == "kan_select" for detail in details)
    return {
        at_kan: state.encode_obs(engine.version, at_kan)
        for at_kan in (False, True)
        if not at_kan or needs_kan
    }


def _attach_policy_probs_batch(
    engine: MortalEngine, jobs: list[dict[str, Any]]
) -> None:
    if not jobs or engine.policy is None:
        return
    records = [
        (job_index, at_kan, obs, mask)
        for job_index, job in enumerate(jobs)
        for at_kan, (obs, mask) in job["inputs"].items()
    ]
    outputs: dict[tuple[int, bool], dict[str, Any]] = {}
    batch_size = 128 if engine.device.type == "cuda" else 16
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        obs = torch.as_tensor(
            np.stack([record[2] for record in batch]),
            dtype=torch.float32,
            device=engine.device,
        )
        masks = torch.as_tensor(
            np.stack([record[3] for record in batch]),
            dtype=torch.bool,
            device=engine.device,
        )
        with (
            torch.inference_mode(),
            torch.autocast(engine.device.type, enabled=engine.enable_amp),
        ):
            phi = engine.brain(obs)
            probs = torch.softmax(engine.policy(phi, masks), dim=-1)
            confidence = (
                engine.confidence(phi) if engine.confidence is not None else None
            )
            rank_probs = None
            if engine.aux_net is not None:
                (rank_logits,) = engine.aux_net(phi)
                rank_probs = torch.softmax(rank_logits, dim=-1)
        probs = probs.float().cpu()
        if confidence is not None:
            confidence = confidence.float().cpu()
        if rank_probs is not None:
            rank_probs = rank_probs.float().cpu()
        for row, (job_index, at_kan, _, _) in enumerate(batch):
            result: dict[str, Any] = {"probs": probs[row]}
            if not at_kan and confidence is not None:
                result["confidence"] = float(confidence[row])
            if not at_kan and rank_probs is not None:
                result["rank_probs"] = [
                    float(value) for value in rank_probs[row].tolist()
                ]
            outputs[(job_index, at_kan)] = result

    for job_index, job in enumerate(jobs):
        entry = job["entry"]
        details = job["details"]
        root = outputs[(job_index, False)]
        kan = outputs.get((job_index, True))
        for detail, public_detail in zip(details, entry["details"], strict=True):
            kind, label = detail["_label"]
            if kind == "general":
                probability = root["probs"][int(label)]
            else:
                if kan is None:
                    raise ValueError("missing conditional kan policy")
                probability = (
                    root["probs"][detail["_root_label"]] * kan["probs"][int(label)]
                )
            public_detail["policy_prob"] = float(probability)

        policy_best_index = max(
            range(len(entry["details"])),
            key=lambda index: entry["details"][index]["policy_prob"],
        )
        policy_best = entry["details"][policy_best_index]
        policy_actual = entry["details"][job["actual_index"]]
        policy_best_prob = float(policy_best["policy_prob"])
        policy_actual_prob = float(policy_actual["policy_prob"])
        entry.update(
            {
                "policy_expected": policy_best["event"],
                "policy_is_equal": policy_best_index == job["actual_index"],
                "policy_actual_prob": policy_actual_prob,
                "policy_loss": max(0.0, policy_best_prob - policy_actual_prob),
            }
        )
        if "confidence" in root:
            entry["policy_confidence"] = root["confidence"]
        if "rank_probs" in root:
            entry["next_rank_probs"] = root["rank_probs"]


def softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    if temperature != 1.0:
        values = [value / temperature for value in values]
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    offset = max_value + math.log(sum(exp_values))
    return [min(max(math.exp(value - offset), 0.0), 1.0) for value in values]


def tile_to_label(tile: str) -> int:
    return TILE_TO_LABEL[tile]


def label_to_tile(label: int) -> str:
    return TILES[label]


def deaka(tile: str) -> str:
    return AKA_TO_BASE.get(tile, tile)


def akaize(tile: str) -> str:
    return BASE_TO_AKA.get(tile, tile)


def next_tile(tile: str) -> str:
    return _shift_tile(tile, 1)


def prev_tile(tile: str) -> str:
    return _shift_tile(tile, -1)


def require_tile(tile: str | None, message: str) -> str:
    if tile is None:
        raise ValueError(message)
    return tile


def has_red_for(tile: str, state: Any) -> bool:
    suit = deaka(tile)[-1]
    if suit == "m":
        return bool(state.akas_in_hand[0])
    if suit == "p":
        return bool(state.akas_in_hand[1])
    if suit == "s":
        return bool(state.akas_in_hand[2])
    return False


def _shift_tile(tile: str, delta: int) -> str:
    base = deaka(tile)
    label = tile_to_label(base)
    if label < 27:
        suit_start = (label // 9) * 9
        shifted = suit_start + ((label - suit_start + delta) % 9)
    elif label < 31:
        shifted = 27 + ((label - 27 + delta) % 4)
    elif label < 34:
        shifted = 31 + ((label - 31 + delta) % 3)
    else:
        shifted = label
    return label_to_tile(shifted)


def _actual_index(
    details: list[dict[str, Any]],
    actual_label: int,
    actual_kan_label: int | None,
) -> int | None:
    for idx, detail in enumerate(details):
        kind, label = detail["_label"]
        if kind == "general" and actual_kan_label is None and label == actual_label:
            return idx
        if (
            kind == "kan_select"
            and actual_kan_label is not None
            and label == actual_kan_label
        ):
            return idx
    return None


def _prob_loss(entry: dict[str, Any]) -> float:
    details = entry["details"]
    return float(details[0]["prob"]) - float(details[entry["actual_index"]]["prob"])


def _kind(event: dict[str, Any]) -> str:
    event_type = event.get("type", "none")
    if event_type in {"daiminkan", "ankan", "kakan"}:
        return "kan"
    return str(event_type)
