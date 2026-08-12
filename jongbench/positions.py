"""Graded single-decision positions extracted from a finished game.

A hanchan costs ~1,000 model calls to play but yields several hundred separately
gradeable decisions. Replaying those decisions as standalone tasks measures the same
thing the full-game rating measures - agreement with Mortal - at one call each, with
byte-identical prompts across models, which a live game can never give you.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import jongbench  # noqa: F401  (sets up the libriichi import path)
import libriichi

from . import actions, engines, evaluate, prompts


@dataclass
class Position:
    """One decision, with Mortal's opinion of every legal answer.

    `rewards[i]` is the normalised Q-advantage of `menu[i]`: 1.0 for Mortal's own choice,
    0.0 for its worst, linear between. That is the per-decision term of the mjai-reviewer
    rating, so a taskset scored on it and a full-game rating measure the same quantity.
    """

    player_id: int
    events: list[dict[str, Any]]
    menu: list[str]
    rewards: list[float]
    best_index: int
    kyoku: int
    honba: int
    junme: int
    tiles_left: int
    shanten: int
    at_furiten: bool
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        return cls(**data)


class MortalArenaEngine(engines.BaseEngine):
    """Mortal as an arena seat, so a bank can be built from boards a strong player would
    actually reach. `libriichi.mjai.Bot` wants every event in order while the arena only
    calls an engine at its own decision points, so each seat replays the events it has not
    seen yet and takes the reaction from the last one."""

    def __init__(self, name: str, engine: Any, **kwargs: Any) -> None:
        kwargs.pop("concurrency", None)
        super().__init__(name, spectator=kwargs.get("spectator"), concurrency=1)
        self._engine = engine
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
            rewards = _score_menu(menu, entry["details"])
            if rewards is None:
                continue
            positions.append(
                Position(
                    player_id=player_id,
                    events=engines.sanitize_events(events[: index + 1], player_id),
                    menu=[str(item["label"]) for item in menu],
                    rewards=rewards,
                    best_index=max(range(len(rewards)), key=rewards.__getitem__),
                    kyoku=int(entry["kyoku"]),
                    honba=int(entry["honba"]),
                    junme=int(entry["junme"]),
                    tiles_left=int(entry["tiles_left"]),
                    shanten=int(entry["shanten"]),
                    at_furiten=bool(entry["at_furiten"]),
                    metadata={"expected": entry["expected"], "actual": entry["actual"]},
                )
            )
    return positions


def _score_menu(
    menu: list[actions.MenuItem], details: list[dict[str, Any]]
) -> list[float] | None:
    """Normalised Q per menu entry, or None when Mortal did not price every option.

    A partial mapping would silently reward whichever actions happened to match, so a
    position that cannot be fully scored is dropped instead.
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
    return [(q - low) / span for q in q_values]


def _lookup(event: dict[str, Any], details: list[dict[str, Any]]) -> float | None:
    for detail in details:
        if evaluate.equal_ignore_aka_consumed(detail["event"], event):
            return float(detail["q_value"])
    return None
