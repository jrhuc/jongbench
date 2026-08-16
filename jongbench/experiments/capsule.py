"""Content-addressed deterministic replay capsules."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from .identity import ControlPolicyIdentity
from .schema import (
    _boolean,
    _integer,
    _number,
    _object_list,
    _optional_string,
    _sha256,
    _string,
    _string_tuple,
    content_id,
)

REPLAY_CAPSULE_FORMAT = "jongbench.replay-capsule"
REPLAY_CAPSULE_SCHEMA = 1
RULESET = "libriichi.tenhou-hanchan-v1"
SEATS = ("seat0", "seat1", "seat2", "seat3")


@dataclass(frozen=True, slots=True)
class ScriptedDecision:
    """One globally ordered model-mediated arena decision."""

    sequence: int
    seat: str
    player_id: int
    kyoku: int
    honba: int
    junme: int
    tiles_left: int
    event_count: int
    menu: tuple[str, ...]
    choice: int
    choice_label: str
    fallback: str | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.seat not in SEATS:
            raise ValueError(f"unknown seat {self.seat!r}")
        if self.player_id not in range(4):
            raise ValueError("player_id must be in [0, 3]")
        if min(
            self.kyoku, self.honba, self.junme, self.tiles_left, self.event_count
        ) < 0:
            raise ValueError("decision coordinates must be non-negative")
        if len(self.menu) < 2:
            raise ValueError("a scripted decision needs at least two legal options")
        if len(set(self.menu)) != len(self.menu):
            raise ValueError("decision menu contains duplicate labels")
        if self.choice not in range(len(self.menu)):
            raise ValueError("choice is out of range for the decision menu")
        if self.choice_label != self.menu[self.choice]:
            raise ValueError("choice_label does not match menu[choice]")

    @property
    def opportunity_key(self) -> tuple[int, int, int]:
        """Stable key shared by simultaneous reactions to one event prefix."""
        return (self.kyoku, self.honba, self.event_count)

    @classmethod
    def from_journal_row(cls, sequence: int, row: dict[str, Any]) -> Self:
        menu = _string_tuple(row.get("menu"), "decision.menu")
        choice = _integer(row.get("choice"), "decision.choice")
        if choice not in range(len(menu)):
            raise ValueError("decision.choice is out of range")
        recorded_label = row.get("choice_label")
        if recorded_label is not None and recorded_label != menu[choice]:
            raise ValueError("decision.choice_label does not match menu[choice]")
        fallback = row.get("fallback")
        if fallback is not None and not isinstance(fallback, str):
            raise TypeError("decision.fallback must be a string or null")
        return cls(
            sequence=sequence,
            seat=_string(row.get("seat"), "decision.seat"),
            player_id=_integer(row.get("player_id"), "decision.player_id"),
            kyoku=_integer(row.get("kyoku"), "decision.kyoku"),
            honba=_integer(row.get("honba"), "decision.honba"),
            junme=_integer(row.get("junme"), "decision.junme"),
            tiles_left=_integer(row.get("tiles_left"), "decision.tiles_left"),
            event_count=_integer(
                row.get("kyoku_events_len"), "decision.kyoku_events_len"
            ),
            menu=menu,
            choice=choice,
            choice_label=menu[choice],
            fallback=fallback,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls.from_journal_row(
            _integer(value.get("sequence"), "decision.sequence"), value
        )

    def with_choice(self, choice: int) -> Self:
        if choice not in range(len(self.menu)):
            raise ValueError("forced choice is out of range")
        return replace(
            self, choice=choice, choice_label=self.menu[choice], fallback=None
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "seat": self.seat,
            "player_id": self.player_id,
            "kyoku": self.kyoku,
            "honba": self.honba,
            "junme": self.junme,
            "tiles_left": self.tiles_left,
            "kyoku_events_len": self.event_count,
            "menu": list(self.menu),
            "choice": self.choice,
            "choice_label": self.choice_label,
            "fallback": self.fallback,
        }


@dataclass(frozen=True, slots=True)
class ReplayCapsule:
    """The immutable information needed to reproduce any recorded decision.

    Decisions remain in global journal order. The intervention cut includes the entire
    target opportunity, so reactions made to the same event prefix remain factual
    co-treatments rather than being changed because the arena asked them later.
    """

    seed: tuple[int, int]
    rotation: int
    table: tuple[str, str, str, str]
    models: tuple[tuple[str, str], ...]
    evaluated_seat: str | None
    profile: str
    control_checkpoint_sha256: str | None
    control_use_policy: bool
    control_boltzmann_epsilon: float
    control_boltzmann_temp: float
    final_scores: tuple[tuple[str, int], ...]
    final_placements: tuple[tuple[str, int], ...]
    decisions: tuple[ScriptedDecision, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        self._validate_identity_fields()
        self._validate_outcomes()
        self._validate_decisions()

    def _validate_identity_fields(self) -> None:
        if len(self.seed) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.seed
        ):
            raise ValueError("seed must be two non-negative integers")
        if self.rotation not in range(4):
            raise ValueError("rotation must be in [0, 3]")
        if len(set(self.table)) != 4 or set(self.table) != set(SEATS):
            raise ValueError("table must contain each logical seat exactly once")
        model_names = tuple(name for name, _ in self.models)
        if set(model_names) != set(SEATS) or len(model_names) != 4:
            raise ValueError("models must name each logical seat exactly once")
        if self.evaluated_seat is not None and self.evaluated_seat not in SEATS:
            raise ValueError("evaluated_seat must be a known seat or null")
        if not self.profile:
            raise ValueError("profile must be non-empty")
        if self.control_checkpoint_sha256 is not None:
            _sha256(self.control_checkpoint_sha256, "control_checkpoint_sha256")
        epsilon = float(self.control_boltzmann_epsilon)
        temperature = float(self.control_boltzmann_temp)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("control_boltzmann_epsilon must be finite and in [0, 1]")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("control_boltzmann_temp must be finite and positive")
        if self.source is not None and (
            not isinstance(self.source, str) or not self.source
        ):
            raise ValueError("source must be a non-empty string or null")

    def _validate_outcomes(self) -> None:
        score_names = tuple(name for name, _ in self.final_scores)
        placement_names = tuple(name for name, _ in self.final_placements)
        if set(score_names) != set(SEATS) or len(score_names) != 4:
            raise ValueError("final_scores must name each logical seat")
        if set(placement_names) != set(SEATS) or len(placement_names) != 4:
            raise ValueError("final_placements must name each logical seat")
        if sorted(value for _, value in self.final_placements) != [1, 2, 3, 4]:
            raise ValueError("final placements must be exactly 1, 2, 3, 4")

    def _validate_decisions(self) -> None:
        if tuple(decision.sequence for decision in self.decisions) != tuple(
            range(len(self.decisions))
        ):
            raise ValueError("decision sequence must be contiguous and zero-based")
        positions = {seat: index for index, seat in enumerate(self.table)}
        seen_opportunities: set[tuple[int, int, int]] = set()
        previous: tuple[int, int, int] | None = None
        for decision in self.decisions:
            if decision.player_id != positions[decision.seat]:
                raise ValueError(
                    f"decision {decision.sequence} player_id does not match table"
                )
            key = decision.opportunity_key
            if key != previous and key in seen_opportunities:
                raise ValueError("decision opportunity rows must be contiguous")
            seen_opportunities.add(key)
            previous = key

    @property
    def capsule_id(self) -> str:
        return content_id("replay-capsule-v1", self._payload_dict())

    @property
    def wall_id(self) -> str:
        """Random deal identity, independent of policy and prompt profile."""
        return content_id(
            "wall-v1", {"ruleset": RULESET, "seed": list(self.seed)}
        )

    def control_identity_for(
        self, checkpoint_sha256: str
    ) -> ControlPolicyIdentity:
        return ControlPolicyIdentity.create(
            checkpoint_sha256=checkpoint_sha256,
            use_policy=self.control_use_policy,
            boltzmann_epsilon=self.control_boltzmann_epsilon,
            boltzmann_temp=self.control_boltzmann_temp,
        )

    def control_policy_id_for(self, checkpoint_sha256: str) -> str:
        return self.control_identity_for(checkpoint_sha256).policy_id

    @classmethod
    def from_episode(cls, episode_dir: str | Path) -> Self:
        from .episode import load_episode_capsule

        return load_episode_capsule(episode_dir)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if value.get("record_type") != "replay_capsule":
            raise ValueError("not a replay capsule record")
        if value.get("format") != REPLAY_CAPSULE_FORMAT:
            raise ValueError("unsupported replay capsule format")
        if value.get("schema_version") != REPLAY_CAPSULE_SCHEMA:
            raise ValueError("unsupported replay capsule schema")
        if value.get("ruleset") != RULESET:
            raise ValueError("unsupported replay capsule ruleset")
        seed_raw = value.get("seed")
        if not isinstance(seed_raw, list) or len(seed_raw) != 2:
            raise TypeError("seed must contain two integers")
        table_raw = _string_tuple(value.get("table"), "table", length=4)
        models_raw = _object_list(value.get("models"), "models")
        scores_raw = _object_list(value.get("final_scores"), "final_scores")
        placements_raw = _object_list(
            value.get("final_placements"), "final_placements"
        )
        decisions_raw = _object_list(value.get("decisions"), "decisions")
        capsule = cls(
            seed=(
                _integer(seed_raw[0], "seed[0]"),
                _integer(seed_raw[1], "seed[1]"),
            ),
            rotation=_integer(value.get("rotation"), "rotation"),
            table=(table_raw[0], table_raw[1], table_raw[2], table_raw[3]),
            models=tuple(
                (
                    _string(item.get("seat"), "models.seat"),
                    _string(item.get("model"), "models.model"),
                )
                for item in models_raw
            ),
            evaluated_seat=_optional_string(
                value.get("evaluated_seat"), "evaluated_seat"
            ),
            profile=_string(value.get("profile"), "profile"),
            control_checkpoint_sha256=_sha256(
                value.get("control_checkpoint_sha256"),
                "control_checkpoint_sha256",
                optional=True,
            ),
            control_use_policy=_boolean(
                value.get("control_use_policy", False), "control_use_policy"
            ),
            control_boltzmann_epsilon=_number(
                value.get("control_boltzmann_epsilon", 0.0),
                "control_boltzmann_epsilon",
            ),
            control_boltzmann_temp=_number(
                value.get("control_boltzmann_temp", 1.0),
                "control_boltzmann_temp",
            ),
            final_scores=tuple(
                (
                    _string(item.get("seat"), "final_scores.seat"),
                    _integer(item.get("score"), "final_scores.score"),
                )
                for item in scores_raw
            ),
            final_placements=tuple(
                (
                    _string(item.get("seat"), "final_placements.seat"),
                    _integer(item.get("placement"), "final_placements.placement"),
                )
                for item in placements_raw
            ),
            decisions=tuple(
                ScriptedDecision.from_dict(item) for item in decisions_raw
            ),
            source=_optional_string(value.get("source"), "source"),
        )
        record_id = value.get("id")
        if record_id is not None and record_id != capsule.capsule_id:
            raise ValueError("replay capsule id does not match canonical content")
        return capsule

    def decision(self, sequence: int) -> ScriptedDecision:
        if sequence not in range(len(self.decisions)):
            raise IndexError(f"decision sequence {sequence} is out of range")
        return self.decisions[sequence]

    def opportunity_id(self, sequence: int) -> str:
        target = self.decision(sequence)
        return content_id(
            "decision-opportunity-v1",
            {
                "capsule_id": self.capsule_id,
                "kyoku": target.kyoku,
                "honba": target.honba,
                "event_count": target.event_count,
            },
        )

    def decision_id(self, sequence: int) -> str:
        target = self.decision(sequence)
        return content_id(
            "decision-v1",
            {
                "opportunity_id": self.opportunity_id(sequence),
                "sequence": sequence,
                "seat": target.seat,
                "menu": list(target.menu),
            },
        )

    def opportunity_end(self, sequence: int) -> int:
        target = self.decision(sequence)
        end = sequence
        while (
            end + 1 < len(self.decisions)
            and self.decisions[end + 1].opportunity_key == target.opportunity_key
        ):
            end += 1
        return end

    def script_for(
        self, target_sequence: int, forced_choice: int
    ) -> tuple[ScriptedDecision, ...]:
        target = self.decision(target_sequence)
        rows = list(self.decisions[: self.opportunity_end(target_sequence) + 1])
        rows[target_sequence] = target.with_choice(forced_choice)
        return tuple(rows)

    def scripts_by_seat(
        self, target_sequence: int, forced_choice: int
    ) -> dict[str, tuple[ScriptedDecision, ...]]:
        grouped: dict[str, list[ScriptedDecision]] = {seat: [] for seat in SEATS}
        for decision in self.script_for(target_sequence, forced_choice):
            grouped[decision.seat].append(decision)
        return {seat: tuple(rows) for seat, rows in grouped.items()}

    def score(self, seat: str) -> int:
        return dict(self.final_scores)[seat]

    def placement(self, seat: str) -> int:
        return dict(self.final_placements)[seat]

    def model(self, seat: str) -> str:
        return dict(self.models)[seat]

    def table_position(self, seat: str) -> int:
        return self.table.index(seat)

    def assert_single_policy_replacement(self, seat: str) -> None:
        """Require exactly one live policy seat and three Mortal controls."""
        if seat not in SEATS:
            raise ValueError(f"unknown evaluated seat {seat!r}")
        live = [name for name, model in self.models if model != "mortal"]
        if live != [seat]:
            raise ValueError(
                "matched-control outcome requires exactly one live policy seat "
                f"and three mortal controls; found live seats {live!r}"
            )

    def _payload_dict(self) -> dict[str, object]:
        return {
            "record_type": "replay_capsule",
            "format": REPLAY_CAPSULE_FORMAT,
            "schema_version": REPLAY_CAPSULE_SCHEMA,
            "ruleset": RULESET,
            "seed": list(self.seed),
            "rotation": self.rotation,
            "table": list(self.table),
            "models": [
                {"seat": seat, "model": model} for seat, model in self.models
            ],
            "evaluated_seat": self.evaluated_seat,
            "profile": self.profile,
            "control_checkpoint_sha256": self.control_checkpoint_sha256,
            "control_use_policy": self.control_use_policy,
            "control_boltzmann_epsilon": self.control_boltzmann_epsilon,
            "control_boltzmann_temp": self.control_boltzmann_temp,
            "final_scores": [
                {"seat": seat, "score": score} for seat, score in self.final_scores
            ],
            "final_placements": [
                {"seat": seat, "placement": placement}
                for seat, placement in self.final_placements
            ],
            "decisions": [decision.as_dict() for decision in self.decisions],
        }

    def as_dict(self, *, include_source: bool = True) -> dict[str, object]:
        record = {**self._payload_dict(), "id": self.capsule_id}
        if include_source:
            record["source"] = self.source
        return record

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def read(cls, path: str | Path) -> Self:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("replay capsule file must contain a JSON object")
        return cls.from_dict(value)
