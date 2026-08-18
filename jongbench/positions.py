"""Graded single-decision positions extracted from a finished game.

A hanchan costs ~1,000 model calls to play but yields several hundred separately
gradeable decisions. Replaying those decisions as standalone tasks measures the same
thing the full-game rating measures - agreement with Mortal - at one call each, with
byte-identical prompts across models, which a live game can never give you.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import jongbench  # noqa: F401  (sets up the libriichi import path)
import libriichi

from . import actions, engines, evaluate, prompts, scorecard
from .bank_schema import (
    BANK_FORMAT,
    BANK_SCHEMA_VERSION,
    REWARD_NAME,
    REWARD_NORMALIZATION,
    BankManifest,
    BankRow,
    DecisionInfo,
    SourceArtifact,
    SourceProvenance,
    bank_manifest,
    board_id,
    dump_bank,
    load_bank,
    position_id,
    prompt_id,
    sample_rows,
    validate_bank_manifest,
    validate_bank_row,
)

__all__ = [
    "BANK_FORMAT",
    "BANK_SCHEMA_VERSION",
    "REWARD_NAME",
    "REWARD_NORMALIZATION",
    "BankManifest",
    "BankRow",
    "DecisionInfo",
    "MortalArenaEngine",
    "Position",
    "SourceArtifact",
    "SourceProvenance",
    "bank_manifest",
    "board_id",
    "dump_bank",
    "extract_positions",
    "load_bank",
    "position_id",
    "prompt_id",
    "rebuild_state",
    "sample_rows",
    "validate_bank_manifest",
    "validate_bank_row",
]


@dataclass
class Position:
    """One decision, with Mortal's opinion of every legal answer.

    `rewards[i]` is the normalised Q-advantage of `menu[i]`: 1.0 for Mortal's
    own choice, 0.0 for its worst, linear between. Raw `q_values` are the
    reviewer's own return units, stored so a consumer can re-weight by stakes.
    """

    player_id: int
    events: list[dict[str, Any]]
    menu: list[str]
    rewards: list[float]
    q_values: list[float]
    best_index: int
    kyoku: int
    honba: int
    junme: int
    tiles_left: int
    shanten: int
    at_furiten: bool
    game_id: str = ""
    source_log: str | None = None
    tags: list[str] = field(default_factory=list)
    reviewer_confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def state(self) -> Any:
        return rebuild_state(self.events, self.player_id)

    def prompt(self, *, state_hints: bool = True) -> str:
        state = self.state()
        menu = actions.build_menu(state)
        labels = [str(item["label"]) for item in menu]
        if labels != self.menu:
            raise ValueError(
                f"stored menu does not match the rebuilt position: "
                f"{self.menu!r} != {labels!r}"
            )
        return prompts.build_user_prompt(
            self.player_id,
            state,
            engines._prompt_safe_events(self.events),
            menu,
            state_hints=state_hints,
        )

    def to_task_dict(self) -> BankRow:
        row: dict[str, Any] = {
            "record_type": "position",
            "schema_version": BANK_SCHEMA_VERSION,
            "prompt": self.prompt(state_hints=True),
            "prompt_without_state_hints": self.prompt(state_hints=False),
            "system_prompt": prompts.SYSTEM,
            "menu": list(self.menu),
            "rewards": list(self.rewards),
            "q_values": list(self.q_values),
            "best_index": self.best_index,
            "game_id": self.game_id or board_id(seat=self.player_id, events=self.events),
            "tags": list(self.tags),
            "info": {
                "seat": self.player_id,
                "kyoku": self.kyoku,
                "honba": self.honba,
                "junme": self.junme,
                "tiles_left": self.tiles_left,
                "shanten": self.shanten,
                "at_furiten": self.at_furiten,
            },
        }
        if self.source_log:
            row["source_log"] = self.source_log
        if self.reviewer_confidence is not None:
            row["reviewer_confidence"] = float(self.reviewer_confidence)
        row["board_id"] = board_id(seat=self.player_id, events=self.events)
        row["prompt_id"] = prompt_id(row)
        row["id"] = position_id(row)
        suffix = str(row["id"]).removeprefix("sha256:")[:12]
        row["name"] = (
            f"kyoku{self.kyoku}-honba{self.honba}-junme{self.junme}-"
            f"seat{self.player_id}-{suffix}"
        )
        return validate_bank_row(row)


class MortalArenaEngine(engines.BaseEngine):
    """Mortal as an arena seat, so a bank can be built from boards a strong player would
    actually reach. `libriichi.mjai.Bot` wants every event in order while the arena only
    calls an engine at its own decision points, so each seat replays the events it has
    not seen yet and takes the reaction from the last one."""

    def __init__(self, name: str, engine: Any, **kwargs: Any) -> None:
        kwargs.pop("concurrency", None)
        super().__init__(name, spectator=kwargs.get("spectator"), concurrency=1)
        self._engine = engine
        self.checkpoint = getattr(engine, "checkpoint", None)
        self._bots: dict[int, tuple[tuple[Any, ...] | None, Any, int]] = {}

    def decide(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[actions.MenuItem],
        game_index: int = 0,
    ) -> dict[str, Any]:
        kyoku = engines._kyoku_id(events)
        cached = self._bots.get(game_index)
        if cached is None or cached[0] != kyoku:
            bot = libriichi.mjai.Bot(self._engine, player_id)
            fed = 0
        else:
            _, bot, fed = cached

        reaction = None
        for event in events[fed:]:
            reaction = bot.react(json.dumps(event, separators=(",", ":")))
        self._bots[game_index] = (kyoku, bot, len(events))

        if reaction is None:
            return next(
                (item["event"] for item in menu if item.get("kind") == "none"),
                menu[0]["event"],
            )
        decoded = json.loads(reaction)
        decoded.pop("meta", None)
        return decoded


def rebuild_state(events: list[dict[str, Any]], player_id: int) -> Any:
    state = libriichi.state.PlayerState(player_id)
    for event in events:
        state.update(json.dumps(event, separators=(",", ":")))
    return state


def extract_positions(
    events: list[dict[str, Any]],
    engine: Any,
    *,
    seats: tuple[int, ...] = (0, 1, 2, 3),
    temperature: float = 0.1,
    min_options: int = 2,
    game_id: str | None = None,
    source_log: str | None = None,
) -> list[Position]:
    """Every gradeable decision in `events`, from each seat's own point of view."""
    positions: list[Position] = []
    reviews = (
        evaluate.review_game(events, engine, temperature)
        if seats == (0, 1, 2, 3)
        else {
            player_id: evaluate.review_player(events, player_id, engine, temperature)
            for player_id in seats
        }
    )
    resolved_game_id = game_id or board_id(seat=0, events=events)
    for player_id in seats:
        review = reviews[player_id]
        by_index = {entry["event_index"]: entry for entry in review["entries"]}
        if not by_index:
            continue

        state = libriichi.state.PlayerState(player_id)
        for index, event in enumerate(events):
            state.update(json.dumps(event, separators=(",", ":")))
            entry = by_index.get(index)
            if entry is None:
                continue
            menu = actions.build_menu(state)
            if len(menu) < min_options:
                continue
            scored = _score_menu(menu, entry["details"])
            if scored is None:
                continue
            q_values, rewards = scored
            pov_events = engines.sanitize_events(events[: index + 1], player_id)
            confidence = entry.get("policy_confidence")
            positions.append(
                Position(
                    player_id=player_id,
                    events=pov_events,
                    menu=[str(item["label"]) for item in menu],
                    rewards=rewards,
                    q_values=q_values,
                    best_index=max(range(len(q_values)), key=q_values.__getitem__),
                    kyoku=int(entry["kyoku"]),
                    honba=int(entry["honba"]),
                    junme=int(entry["junme"]),
                    tiles_left=int(entry["tiles_left"]),
                    shanten=int(entry["shanten"]),
                    at_furiten=bool(entry["at_furiten"]),
                    game_id=resolved_game_id,
                    source_log=source_log,
                    tags=scorecard.competence_tags(state, menu, pov_events),
                    reviewer_confidence=(
                        float(confidence) if confidence is not None else None
                    ),
                    metadata={"expected": entry["expected"], "actual": entry["actual"]},
                )
            )
    return positions


def _score_menu(
    menu: list[actions.MenuItem], details: list[dict[str, Any]]
) -> tuple[list[float], list[float]] | None:
    """Raw and normalised Q per menu entry, or None when Mortal did not price every option.

    A partial mapping would silently reward whichever actions happened to match, so a
    position that cannot be fully scored is dropped instead. Zero-span menus are also
    dropped: they have no decision.
    """
    q_values: list[float] = []
    for item in menu:
        q = _lookup(item["event"], details)
        if q is None:
            return None
        q_values.append(q)
    low, high = min(q_values), max(q_values)
    span = high - low
    if span <= 0:
        return None
    return q_values, [(q - low) / span for q in q_values]


def _lookup(event: dict[str, Any], details: list[dict[str, Any]]) -> float | None:
    for detail in details:
        if evaluate.equal_ignore_aka_consumed(detail["event"], event):
            return float(detail["q_value"])
    return None
