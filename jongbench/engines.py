from __future__ import annotations

import copy
import json
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import actions, prompts, providers


DecisionSink = list[dict[str, Any]] | Callable[[dict[str, Any]], None]


@dataclass
class _Conversation:
    """One seat's running transcript for one kyoku."""

    kyoku: tuple[Any, ...] | None
    messages: list[dict[str, Any]] = field(default_factory=list)
    seen_events: int = 0
    calls_enabled: bool = True


# Reaction kinds a declined-calls seat may skip. `hora` is deliberately absent: the furo
# toggle must never cost a seat a win, and neither may it touch its own turn.
_DECLINABLE = {"none", "chi", "pon", "daiminkan"}


def _kyoku_id(events: list[dict[str, Any]]) -> tuple[Any, ...] | None:
    for event in events:
        if event.get("type") == "start_kyoku":
            return (event.get("bakaze"), event.get("kyoku"), event.get("honba"))
    return None


class GameAborted(RuntimeError):
    pass


def sanitize_events(events: list[dict[str, Any]], player_id: int) -> list[dict[str, Any]]:
    sanitized = copy.deepcopy(events)
    for event in sanitized:
        if event.get("type") == "start_kyoku":
            tehais = event.get("tehais")
            if isinstance(tehais, list):
                for seat, hand in enumerate(tehais):
                    if seat != player_id:
                        tehais[seat] = ["?"] * 13
        elif event.get("type") == "tsumo" and event.get("actor") != player_id:
            if "pai" in event:
                event["pai"] = "?"
    return sanitized


class BaseEngine(ABC):
    # A human needs to see even the only legal decision so the web and terminal
    # UIs can keep the choice in the player's hands. Automated engines can
    # safely skip an otherwise pointless round trip.
    auto_choose_single_menu = True

    engine_type = "mjai-log"

    def __init__(
        self,
        name: str,
        spectator: Any | None = None,
        concurrency: int = 4,
    ) -> None:
        self.name = name
        self.spectator = spectator
        self.concurrency = max(1, int(concurrency))
        self.player_ids: list[int] | None = None
        self.cancel_event: threading.Event | None = None

    def set_player_ids(self, player_ids: list[int]) -> None:
        self.player_ids = [int(player_id) for player_id in player_ids]

    def react_batch(self, game_states: list[Any]) -> list[str]:
        if self.player_ids is None:
            raise RuntimeError("player_ids not set")
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise GameAborted("game aborted")

        reactions: list[dict[str, Any] | None] = [None] * len(game_states)
        jobs: list[
            tuple[int, int, int, Any, list[dict[str, Any]], list[actions.MenuItem]]
        ] = []

        for index, game_state in enumerate(game_states):
            game_index = int(game_state.game_index)
            player_id = self.player_ids[game_index]
            raw_events = json.loads(game_state.events_json)
            if self.spectator is not None:
                self.spectator.publish(game_index, player_id, raw_events)

            events = sanitize_events(raw_events, player_id)
            menu = actions.build_menu(game_state.state)
            auto = self.auto_reaction(game_state.state, menu, events, game_index)
            if not menu:
                reactions[index] = {"type": "none"}
            elif len(menu) == 1 and self.auto_choose_single_menu:
                reactions[index] = menu[0]["event"]
            elif auto is not None:
                reactions[index] = auto
            else:
                jobs.append((index, game_index, player_id, game_state.state, events, menu))

        if jobs:
            if self.concurrency == 1 or len(jobs) == 1:
                for index, game_index, player_id, state, events, menu in jobs:
                    reactions[index] = self.decide(
                        player_id, state, events, menu, game_index=game_index
                    )
            else:
                with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                    futures = [
                        executor.submit(
                            self.decide, player_id, state, events, menu, game_index=game_index
                        )
                        for _, game_index, player_id, state, events, menu in jobs
                    ]
                    for (index, *_), future in zip(jobs, futures, strict=True):
                        reactions[index] = future.result()

        if self.cancel_event is not None and self.cancel_event.is_set():
            raise GameAborted("game aborted")

        encoded: list[str] = []
        for reaction in reactions:
            if reaction is None:
                raise RuntimeError("missing reaction")
            encoded.append(json.dumps(reaction, separators=(",", ":")))
        return encoded

    def auto_reaction(
        self,
        state: Any,
        menu: list[actions.MenuItem],
        events: list[dict[str, Any]],
        game_index: int,
    ) -> dict[str, Any] | None:
        """An action to play without consulting the engine, or None to decide normally."""
        return None

    def start_game(self, game_idx: int) -> None:
        pass

    def end_kyoku(self, game_idx: int) -> None:
        pass

    def end_kyoku_with_log(self, game_idx: int, events_json: str) -> None:
        if self.spectator is not None:
            events = json.loads(events_json)
            self.spectator.publish(game_idx, self.player_ids[game_idx], events)
        self.end_kyoku(game_idx)

    def end_game(self, game_idx: int, scores: list[int]) -> None:
        pass

    @abstractmethod
    def decide(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[actions.MenuItem],
        game_index: int = 0,
    ) -> dict[str, Any]:
        raise NotImplementedError


class RandomEngine(BaseEngine):
    def __init__(
        self,
        name: str,
        seed: int = 0,
        spectator: Any | None = None,
        concurrency: int = 4,
        **_: Any,
    ) -> None:
        del concurrency
        super().__init__(name, spectator=spectator, concurrency=1)
        self._random = random.Random(seed)
        self._random_lock = threading.Lock()

    def decide(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[actions.MenuItem],
        game_index: int = 0,
    ) -> dict[str, Any]:
        del player_id, state, events
        for item in menu:
            if item.get("kind") == "hora":
                return item["event"]

        weights = [6 if item.get("kind") == "none" else 1 for item in menu]
        with self._random_lock:
            return self._random.choices(menu, weights=weights, k=1)[0]["event"]


class LLMEngine(BaseEngine):
    def __init__(
        self,
        name: str,
        spec_str: str,
        api_key: str | None = None,
        decision_log: DecisionSink | None = None,
        temperature: float = 0.6,
        max_tokens: int = 1200,
        spectator: Any | None = None,
        concurrency: int = 4,
        state_hints: bool = True,
        reasoning: str | None = None,
        conversational: bool = True,
    ) -> None:
        super().__init__(name, spectator=spectator, concurrency=concurrency)
        self.spec = providers.parse_spec(spec_str)
        self.reasoning = reasoning
        self.provider = providers.make_provider(
            self.spec, api_key=api_key, reasoning=reasoning
        )
        self.decision_log: DecisionSink = [] if decision_log is None else decision_log
        self.temperature = temperature
        self.max_tokens = max_tokens
        if reasoning not in {None, "off"}:
            floor = 32000 if reasoning in {"xhigh", "max"} else 16000
            self.max_tokens = max(self.max_tokens, floor)
        self.state_hints = bool(state_hints)
        self.totals = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "fallbacks": 0,
            "retries": 0,
            "calls_declined": 0,
            "call_policy_changes": 0,
        }
        self.conversational = bool(conversational)
        self._log_lock = threading.Lock()
        self._conversations: dict[int, _Conversation] = {}
        self._conversation_lock = threading.Lock()

    def decide(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[actions.MenuItem],
        game_index: int = 0,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        labels = [str(item["label"]) for item in menu]
        prompt_events = _prompt_safe_events(events)
        history, opening = self._open_turn(game_index, prompt_events)
        own_turn = bool(state.last_cans.can_discard)
        calls_enabled = history.calls_enabled if own_turn else None
        if opening:
            prompt = prompts.build_user_prompt(
                player_id,
                state,
                prompt_events,
                menu,
                state_hints=self.state_hints,
                calls_enabled=calls_enabled,
            )
        else:
            prompt = prompts.build_followup_prompt(
                player_id,
                state,
                prompt_events[history.seen_events :],
                menu,
                state_hints=self.state_hints,
                calls_enabled=calls_enabled,
            )
        raw_response = ""
        raw_reasoning = ""
        served_by: str | None = None
        retries = 0
        calls = 0
        usage_total = _empty_usage()
        fallback: str | None = None
        choice: int | None = None

        def complete_once(user_prompt: str, *, correcting: bool = False) -> providers.Completion:
            nonlocal calls, raw_response, raw_reasoning, served_by
            calls += 1
            turns = list(history.messages)
            if correcting:
                turns += [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": raw_response or "(no reply)"},
                ]
            # The breakpoint rides the newest user turn: everything above it is a stable
            # append-only prefix, so each turn reads what the previous turn wrote.
            turns.append({"role": "user", "content": providers.cacheable(user_prompt)})
            result = self.provider.complete(
                [
                    {"role": "system", "content": providers.cacheable(prompts.SYSTEM)},
                    *turns,
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            raw_response = result.text
            raw_reasoning = result.reasoning
            served_by = result.served_by or served_by
            _add_usage(usage_total, result.usage)
            return result

        try:
            complete_once(prompt)
        except Exception:
            retries = 1
            try:
                complete_once(prompt)
                choice = prompts.extract_choice(raw_response, len(menu))
            except ValueError as exc:
                fallback = f"invalid_choice:{exc}"
            except Exception as exc:
                fallback = f"provider_error:{type(exc).__name__}"
        else:
            try:
                choice = prompts.extract_choice(raw_response, len(menu))
            except ValueError as exc:
                retries = 1
                retry_prompt = (
                    f"Your previous reply was invalid: {exc}. "
                    'Reply again with exactly: {"choice": N}'
                )
                try:
                    complete_once(retry_prompt, correcting=True)
                    choice = prompts.extract_choice(raw_response, len(menu))
                except ValueError as retry_exc:
                    fallback = f"invalid_choice:{retry_exc}"
                except Exception as retry_exc:
                    fallback = f"provider_error:{type(retry_exc).__name__}"

        if choice is None:
            choice = _fallback_choice(menu)
            if fallback is None:
                fallback = "invalid_choice"

        requested = prompts.extract_call_policy(raw_response) if own_turn else None
        if requested is not None and requested != history.calls_enabled:
            history.calls_enabled = requested
            with self._log_lock:
                self.totals["call_policy_changes"] += 1

        # The transcript records what was actually played, fallbacks included, so the
        # seat's next turn reasons from the real board rather than an answer it never gave.
        self._close_turn(history, prompt, choice, len(prompt_events))

        latency_ms = (time.perf_counter() - started) * 1000.0
        record = {
            "ts": time.time(),
            "player_id": player_id,
            "kyoku_events_len": len(events),
            "menu": labels,
            "choice": choice,
            "choice_label": labels[choice],
            "prompt_version": 3,
            "state_hints": self.state_hints,
            "calls_enabled": history.calls_enabled,
            "fallback": fallback,
            "raw_response": raw_response[:4000],
            "raw_reasoning": raw_reasoning[:16000],
            "served_by": served_by,
            "usage": usage_total,
            "latency_ms": latency_ms,
            "retries": retries,
        }
        self._record_decision(record, calls, usage_total, fallback is not None, retries)
        return menu[choice]["event"]

    def auto_reaction(
        self,
        state: Any,
        menu: list[actions.MenuItem],
        events: list[dict[str, Any]],
        game_index: int,
    ) -> dict[str, Any] | None:
        """Pass on a call the seat has already said it does not want. This only fires on
        a pure chi/pon/open-kan reaction: a menu offering a win, or anything on the
        seat's own turn, is always put to the model."""
        if not menu or bool(state.last_cans.can_discard):
            return None
        kinds = {str(item.get("kind")) for item in menu}
        if not kinds <= _DECLINABLE:
            return None
        with self._conversation_lock:
            history = self._conversations.get(game_index)
            if history is None or history.kyoku != _kyoku_id(events):
                return None
            if history.calls_enabled:
                return None
        pass_item = next((item for item in menu if item.get("kind") == "none"), None)
        if pass_item is None:
            return None
        with self._log_lock:
            self.totals["calls_declined"] += 1
        return pass_item["event"]

    def _open_turn(
        self, game_index: int, events: list[dict[str, Any]]
    ) -> tuple[_Conversation, bool]:
        """This seat's live conversation for the current kyoku, and whether it is new.
        A kyoku is the natural span: the board resets, so carrying the previous hand's
        transcript would cost tokens and invite the model to reason from a dead board."""
        kyoku = _kyoku_id(events)
        if not self.conversational:
            return _Conversation(kyoku=kyoku), True
        with self._conversation_lock:
            history = self._conversations.get(game_index)
            if history is None or history.kyoku != kyoku:
                history = _Conversation(kyoku=kyoku)
                self._conversations[game_index] = history
            return history, not history.messages

    def _close_turn(
        self, history: _Conversation, prompt: str, choice: int, seen_events: int
    ) -> None:
        if not self.conversational:
            return
        with self._conversation_lock:
            history.messages.append({"role": "user", "content": prompt})
            history.messages.append(
                {"role": "assistant", "content": json.dumps({"choice": choice})}
            )
            history.seen_events = seen_events

    def _record_decision(
        self,
        record: dict[str, Any],
        calls: int,
        usage: dict[str, int],
        did_fallback: bool,
        retries: int,
    ) -> None:
        with self._log_lock:
            self.totals["calls"] += calls
            for key in _USAGE_KEYS:
                self.totals[key] += usage.get(key, 0)
            self.totals["fallbacks"] += int(did_fallback)
            self.totals["retries"] += retries
            if callable(self.decision_log):
                self.decision_log(record)
            else:
                self.decision_log.append(record)


def make_engine(name: str, spec_str: str, **kwargs: Any) -> BaseEngine:
    spec = providers.parse_spec(spec_str)
    if spec.provider == "human":
        human_io = kwargs.get("human_io")
        if human_io is None:
            raise ValueError("human seat requires human_io")
        return HumanEngine(name, human_io, spectator=kwargs.get("spectator"))
    if spec.provider == "random":
        return RandomEngine(name, **kwargs)
    return LLMEngine(name, spec_str, **kwargs)


_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)


def _empty_usage() -> dict[str, int]:
    return dict.fromkeys(_USAGE_KEYS, 0)


def _add_usage(total: dict[str, int], usage: dict[str, Any] | None) -> None:
    for key in _USAGE_KEYS:
        total[key] += int((usage or {}).get(key, 0) or 0)


def _fallback_choice(menu: list[actions.MenuItem]) -> int:
    for idx, item in enumerate(menu):
        event = item.get("event", {})
        if (
            item.get("kind") == "discard"
            and event.get("type") == "dahai"
            and event.get("tsumogiri")
        ):
            return idx
    for idx, item in enumerate(menu):
        if item.get("kind") == "none" or item.get("event", {}).get("type") == "none":
            return idx
    return 0


def _prompt_safe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_events = copy.deepcopy(events)
    for event in safe_events:
        if event.get("type") == "tsumo" and event.get("pai") == "?":
            event.pop("pai", None)
    return safe_events


class HumanIO(ABC):
    @abstractmethod
    def ask(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError


class TerminalHumanIO(HumanIO):
    def ask(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[dict[str, Any]],
    ) -> dict[str, Any]:
        print("-" * 80)
        print(prompts.render_state(player_id, state, events))
        print("")
        print(prompts.render_menu(menu))
        while True:
            raw = input("your choice> ")
            try:
                choice = int(raw)
            except ValueError:
                print("enter a number")
                continue
            if 0 <= choice < len(menu):
                return menu[choice]["event"]
            print(f"choose 0-{len(menu) - 1}")


class HumanEngine(BaseEngine):
    auto_choose_single_menu = False

    def __init__(
        self,
        name: str,
        io: HumanIO,
        spectator: Any | None = None,
    ) -> None:
        super().__init__(name, spectator=spectator, concurrency=1)
        self.io = io

    def decide(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[actions.MenuItem],
        game_index: int = 0,
    ) -> dict[str, Any]:
        return self.io.ask(player_id, state, events, menu)
