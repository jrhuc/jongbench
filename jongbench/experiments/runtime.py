"""Deterministic replay handoff for branch experiments."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

from .. import actions, engines
from .capsule import ScriptedDecision

_DECLINABLE_REACTIONS = frozenset({"none", "chi", "pon", "daiminkan"})
_BAKAZE_ORDER = {"E": 0, "S": 1, "W": 2, "N": 3}


class ReplayDiverged(RuntimeError):
    """The deterministic arena no longer matches a recorded decision prefix."""


def _absolute_kyoku(event: dict[str, Any]) -> int:
    return (
        _BAKAZE_ORDER.get(str(event.get("bakaze")), 0) * 4
        + int(event.get("kyoku", 1))
        - 1
    )


def current_opportunity(
    events: Sequence[dict[str, Any]],
) -> tuple[int, int, int]:
    """Coordinates used by journal rows for the current arena reaction cycle."""
    start = next(
        (event for event in reversed(events) if event.get("type") == "start_kyoku"),
        None,
    )
    if start is None:
        raise ReplayDiverged("arena decision has no start_kyoku event")
    return (
        _absolute_kyoku(start),
        int(start.get("honba", 0)),
        len(events),
    )


class ReplayThenEngine(engines.BaseEngine):
    """Replay model-mediated decisions through one global cut, then delegate.

    Controls are never scripted: their deterministic policy is recomputed. Live model
    seats replay both recorded decisions and omitted auto-pass reactions until the
    target opportunity, even when that seat has no later journal row of its own.
    """

    def __init__(
        self,
        name: str,
        script: Sequence[ScriptedDecision],
        continuation: engines.BaseEngine,
        *,
        replay_until: tuple[int, int, int] | None = None,
        scripted_policy: bool = True,
    ) -> None:
        super().__init__(name, spectator=None, concurrency=1)
        if any(row.seat != name for row in script):
            raise ValueError(f"script for {name!r} contains another seat")
        if not scripted_policy and script:
            raise ValueError("fixed controls must not carry a model decision script")
        self._script = deque(script)
        self._continuation = continuation
        self._scripted_policy = bool(scripted_policy)
        self._replay_until = (
            replay_until
            if replay_until is not None
            else (script[-1].opportunity_key if script else None)
        )
        if self._scripted_policy and self._replay_until is None:
            raise ValueError("a scripted policy needs a global replay cut")
        self.engine_type = getattr(continuation, "engine_type", "replay-then-control")

    @property
    def remaining(self) -> int:
        return len(self._script)

    def set_player_ids(self, player_ids: list[int]) -> None:
        super().set_player_ids(player_ids)
        self._continuation.set_player_ids(player_ids)

    def start_game(self, game_idx: int) -> None:
        self._continuation.start_game(game_idx)

    def end_kyoku(self, game_idx: int) -> None:
        self._continuation.end_kyoku(game_idx)

    def end_kyoku_with_log(self, game_idx: int, events_json: str) -> None:
        self._continuation.end_kyoku_with_log(game_idx, events_json)

    def end_game(self, game_idx: int, scores: list[int]) -> None:
        self._continuation.end_game(game_idx, scores)

    def _inside_replay_cut(self, events: Sequence[dict[str, Any]]) -> bool:
        if not self._scripted_policy:
            return False
        assert self._replay_until is not None
        return current_opportunity(events) <= self._replay_until

    def auto_reaction(
        self,
        state: Any,
        menu: list[actions.MenuItem],
        events: list[dict[str, Any]],
        game_index: int,
    ) -> dict[str, Any] | None:
        if not self._scripted_policy:
            return self._continuation.auto_reaction(
                state, menu, events, game_index
            )

        labels = tuple(str(item["label"]) for item in menu)
        if self._script and self._script[0].menu == labels:
            return None
        if not self._inside_replay_cut(events):
            return self._continuation.auto_reaction(
                state, menu, events, game_index
            )

        # Live-seat journals omit reactions automatically declined by the harness.
        kinds = {str(item.get("kind")) for item in menu}
        if (
            menu
            and not bool(state.last_cans.can_discard)
            and kinds <= _DECLINABLE_REACTIONS
        ):
            pass_item = next(
                (item for item in menu if item.get("kind") == "none"), None
            )
            if pass_item is not None:
                return pass_item["event"]
        return None

    def decide(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[actions.MenuItem],
        game_index: int = 0,
    ) -> dict[str, Any]:
        if not self._scripted_policy:
            return self._continuation.decide(
                player_id, state, events, menu, game_index=game_index
            )

        current = current_opportunity(events)
        if not self._script:
            if current <= self._replay_until:
                raise ReplayDiverged(
                    f"{self.name}: unrecorded model decision before replay cut "
                    f"at {current}"
                )
            return self._continuation.decide(
                player_id, state, events, menu, game_index=game_index
            )

        expected = self._script[0]
        if current != expected.opportunity_key:
            raise ReplayDiverged(
                f"{self.name}: decision {expected.sequence} recorded at "
                f"{expected.opportunity_key}, arena reached {current}"
            )
        labels = tuple(str(item["label"]) for item in menu)
        if expected.player_id != player_id:
            raise ReplayDiverged(
                f"{self.name}: recorded player {expected.player_id}, "
                f"arena requested {player_id}"
            )
        if expected.menu != labels:
            raise ReplayDiverged(
                f"{self.name}: decision {expected.sequence} recorded "
                f"{expected.menu!r}, arena offered {labels!r}"
            )
        self._script.popleft()
        return menu[expected.choice]["event"]

    def assert_consumed(self) -> None:
        if self._script:
            next_row = self._script[0]
            raise ReplayDiverged(
                f"{self.name}: replay ended before decision {next_row.sequence}"
            )
