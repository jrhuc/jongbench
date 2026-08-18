"""Graded single-decision positions extracted from a finished game.

A hanchan costs ~1,000 model calls to play but yields several hundred separately
gradeable decisions. Replaying those decisions as standalone tasks measures the same
thing the full-game rating measures - agreement with Mortal - at one call each, with
byte-identical prompts across models, which a live game can never give you.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, Required, TextIO, TypedDict, cast

import jongbench  # noqa: F401  (sets up the libriichi import path)
import libriichi

from . import actions, engines, evaluate, prompts

BANK_SCHEMA_VERSION = 2
BANK_FORMAT = "jongbench.riichi-decision-bank"
REWARD_NAME = "mortal_q_advantage"
REWARD_NORMALIZATION = "per_position_min_max"


class DecisionInfo(TypedDict):
    seat: int
    kyoku: int
    honba: int
    junme: int
    tiles_left: int
    shanten: int
    at_furiten: bool


class BankRow(TypedDict):
    record_type: Literal["position"]
    schema_version: Literal[2]
    id: str
    name: str
    prompt: str
    prompt_without_state_hints: str
    system_prompt: str
    menu: list[str]
    rewards: list[float]
    best_index: int
    info: DecisionInfo


class GeneratorProvenance(TypedDict):
    name: str
    version: str


class RewardContract(TypedDict):
    name: str
    normalization: str
    range: list[float]


class ReviewerProvenance(TypedDict):
    name: str
    checkpoint: str
    checkpoint_sha256: str
    temperature: float


class SourceArtifact(TypedDict):
    name: str
    sha256: str


class SourceProvenance(TypedDict, total=False):
    kind: Required[str]
    description: str
    seed: int
    games: int
    artifacts: list[SourceArtifact]


class BankManifest(TypedDict):
    record_type: Literal["manifest"]
    schema_version: Literal[2]
    bank_format: str
    generator: GeneratorProvenance
    reward: RewardContract
    reviewer: ReviewerProvenance
    source: SourceProvenance


_ROW_FIELDS = frozenset(BankRow.__required_keys__)
_INFO_FIELDS = frozenset(DecisionInfo.__required_keys__)
_MANIFEST_FIELDS = frozenset(BankManifest.__required_keys__)
_ID_FIELDS = (
    "prompt",
    "prompt_without_state_hints",
    "system_prompt",
    "menu",
    "rewards",
    "best_index",
    "info",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def position_id(row: Mapping[str, object]) -> str:
    """Stable content identity for the board as rendered and graded.

    Names and file order deliberately do not participate. Both prompt variants and the
    system prompt do: changing anything a model sees creates a new decision identity.
    """
    identity = {key: row[key] for key in _ID_FIELDS}
    return "sha256:" + hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


def _object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, object], fields: frozenset[str], where: str
) -> None:
    missing = fields - value.keys()
    if missing:
        raise ValueError(
            f"{where} is missing required field(s): {', '.join(sorted(missing))}"
        )
    extra = value.keys() - fields
    if extra:
        raise ValueError(f"{where} has unknown field(s): {', '.join(sorted(extra))}")


def _nonempty_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{where} must be a non-empty string")
    return value


def _integer(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{where} must be an integer")
    return value


def validate_bank_row(value: object, row_index: int | None = None) -> BankRow:
    where = "bank row" if row_index is None else f"bank row {row_index}"
    row = _object(value, where)
    _exact_fields(row, _ROW_FIELDS, where)
    _nonempty_string(row["record_type"], f"{where}.record_type")
    if row["record_type"] != "position":
        raise ValueError(f"{where} has invalid record_type")
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != BANK_SCHEMA_VERSION
    ):
        raise ValueError(f"{where} has unsupported schema_version")
    for key in ("id", "name", "prompt", "prompt_without_state_hints", "system_prompt"):
        _nonempty_string(row[key], f"{where}.{key}")

    menu = row["menu"]
    if not isinstance(menu, list):
        raise TypeError(f"{where}.menu must be a list")
    if len(menu) < 2:
        raise ValueError(f"{where}.menu must contain at least two options")
    for index, label in enumerate(menu):
        _nonempty_string(label, f"{where}.menu[{index}]")
    if len(set(menu)) != len(menu):
        raise ValueError(f"{where}.menu must not contain duplicate options")

    rewards = row["rewards"]
    if not isinstance(rewards, list):
        raise TypeError(f"{where}.rewards must be a list")
    if len(rewards) != len(menu):
        raise ValueError(f"{where} has mismatched menu and rewards lengths")
    for index, reward in enumerate(rewards):
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise TypeError(f"{where}.rewards[{index}] must be a number")
        if not math.isfinite(float(reward)):
            raise ValueError(f"{where}.rewards[{index}] must be finite")
        if not 0.0 <= float(reward) <= 1.0:
            raise ValueError(f"{where}.rewards[{index}] must be in [0, 1]")

    best_index = _integer(row["best_index"], f"{where}.best_index")
    if not 0 <= best_index < len(menu):
        raise ValueError(f"{where}.best_index is out of range")
    if float(rewards[best_index]) != max(map(float, rewards)):
        raise ValueError(f"{where}.best_index must select an argmax reward")

    info = _object(row["info"], f"{where}.info")
    _exact_fields(info, _INFO_FIELDS, f"{where}.info")
    seat = _integer(info["seat"], f"{where}.info.seat")
    if not 0 <= seat <= 3:
        raise ValueError(f"{where}.info.seat must be in [0, 3]")
    for key in ("kyoku", "honba", "junme"):
        if _integer(info[key], f"{where}.info.{key}") < 0:
            raise ValueError(f"{where}.info.{key} must be non-negative")
    tiles_left = _integer(info["tiles_left"], f"{where}.info.tiles_left")
    if not 0 <= tiles_left <= 70:
        raise ValueError(f"{where}.info.tiles_left must be in [0, 70]")
    shanten = _integer(info["shanten"], f"{where}.info.shanten")
    if not -1 <= shanten <= 13:
        raise ValueError(f"{where}.info.shanten must be in [-1, 13]")
    if not isinstance(info["at_furiten"], bool):
        raise TypeError(f"{where}.info.at_furiten must be a boolean")

    expected_id = position_id(row)
    if row["id"] != expected_id:
        raise ValueError(f"{where}.id does not match its canonical content hash")
    return cast(BankRow, row)


def validate_bank_manifest(value: object) -> BankManifest:
    where = "bank manifest"
    manifest = _object(value, where)
    _exact_fields(manifest, _MANIFEST_FIELDS, where)
    _nonempty_string(manifest["record_type"], f"{where}.record_type")
    if manifest["record_type"] != "manifest":
        raise ValueError(f"{where} has invalid record_type")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != BANK_SCHEMA_VERSION
    ):
        raise ValueError(f"{where} has unsupported schema_version")
    _nonempty_string(manifest["bank_format"], f"{where}.bank_format")
    if manifest["bank_format"] != BANK_FORMAT:
        raise ValueError(f"{where} has unsupported bank_format")

    generator = _object(manifest["generator"], f"{where}.generator")
    _exact_fields(
        generator,
        frozenset(GeneratorProvenance.__required_keys__),
        f"{where}.generator",
    )
    _nonempty_string(generator["name"], f"{where}.generator.name")
    _nonempty_string(generator["version"], f"{where}.generator.version")

    reward = _object(manifest["reward"], f"{where}.reward")
    _exact_fields(
        reward, frozenset(RewardContract.__required_keys__), f"{where}.reward"
    )
    _nonempty_string(reward["name"], f"{where}.reward.name")
    _nonempty_string(reward["normalization"], f"{where}.reward.normalization")
    if reward["name"] != REWARD_NAME or reward["normalization"] != REWARD_NORMALIZATION:
        raise ValueError(f"{where} has unsupported reward semantics")
    reward_range = reward["range"]
    if not isinstance(reward_range, list) or len(reward_range) != 2:
        raise TypeError(f"{where}.reward.range must be a two-number list")
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in reward_range
    ):
        raise TypeError(f"{where}.reward.range must contain numbers")
    if not all(math.isfinite(float(item)) for item in reward_range):
        raise ValueError(f"{where}.reward.range must be finite")
    if list(map(float, reward_range)) != [0.0, 1.0]:
        raise ValueError(f"{where}.reward.range must be [0.0, 1.0]")

    reviewer = _object(manifest["reviewer"], f"{where}.reviewer")
    _exact_fields(
        reviewer, frozenset(ReviewerProvenance.__required_keys__), f"{where}.reviewer"
    )
    _nonempty_string(reviewer["name"], f"{where}.reviewer.name")
    _nonempty_string(reviewer["checkpoint"], f"{where}.reviewer.checkpoint")
    checkpoint_sha256 = _nonempty_string(
        reviewer["checkpoint_sha256"], f"{where}.reviewer.checkpoint_sha256"
    )
    if _SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise ValueError(f"{where}.reviewer.checkpoint_sha256 must be a SHA-256 digest")
    temperature = reviewer["temperature"]
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError(f"{where}.reviewer.temperature must be a number")
    if not math.isfinite(float(temperature)) or float(temperature) < 0:
        raise ValueError(
            f"{where}.reviewer.temperature must be finite and non-negative"
        )

    source = _object(manifest["source"], f"{where}.source")
    _nonempty_string(source.get("kind"), f"{where}.source.kind")
    if "description" in source:
        _nonempty_string(source["description"], f"{where}.source.description")
    for key in ("seed", "games"):
        if key in source and _integer(source[key], f"{where}.source.{key}") < 0:
            raise ValueError(f"{where}.source.{key} must be non-negative")
    if "artifacts" in source:
        artifacts = source["artifacts"]
        if not isinstance(artifacts, list):
            raise TypeError(f"{where}.source.artifacts must be a list")
        for index, artifact_value in enumerate(artifacts):
            artifact_where = f"{where}.source.artifacts[{index}]"
            artifact = _object(artifact_value, artifact_where)
            _exact_fields(
                artifact, frozenset(SourceArtifact.__required_keys__), artifact_where
            )
            _nonempty_string(artifact["name"], f"{artifact_where}.name")
            digest = _nonempty_string(artifact["sha256"], f"{artifact_where}.sha256")
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{artifact_where}.sha256 must be a SHA-256 digest")
    # Extra source fields are deliberately permitted: provenance is generator-specific,
    # while `kind` remains the stable required discriminator.
    _canonical_json(source)
    return cast(BankManifest, manifest)


def bank_manifest(
    *,
    reviewer_checkpoint: str,
    reviewer_checkpoint_sha256: str,
    source: SourceProvenance,
    reviewer_temperature: float = 0.1,
    generator_version: str | None = None,
) -> BankManifest:
    """Build the required first JSONL record for a schema-v2 bank."""
    if generator_version is None:
        try:
            generator_version = version("jongbench")
        except PackageNotFoundError:
            generator_version = "0+unknown"
    # Detach the caller-owned mapping and prove that it is JSON-safe.
    source_copy = json.loads(_canonical_json(source))
    manifest: BankManifest = {
        "record_type": "manifest",
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_format": BANK_FORMAT,
        "generator": {"name": "jongbench", "version": generator_version},
        "reward": {
            "name": REWARD_NAME,
            "normalization": REWARD_NORMALIZATION,
            "range": [0.0, 1.0],
        },
        "reviewer": {
            "name": "Mortal",
            "checkpoint": reviewer_checkpoint,
            "checkpoint_sha256": reviewer_checkpoint_sha256,
            "temperature": reviewer_temperature,
        },
        "source": cast(SourceProvenance, source_copy),
    }
    return validate_bank_manifest(manifest)


def dump_bank(handle: TextIO, manifest: BankManifest, rows: Iterable[BankRow]) -> int:
    """Write one validated manifest followed by validated, uniquely identified rows."""
    checked_manifest = validate_bank_manifest(manifest)
    checked_rows: list[BankRow] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(rows):
        row = validate_bank_row(value, index)
        if row["id"] in seen_ids:
            raise ValueError(f"bank row {index}.id duplicates an earlier row")
        seen_ids.add(row["id"])
        checked_rows.append(row)
    if not checked_rows:
        raise ValueError("bank must contain at least one position row")
    handle.write(_canonical_json(checked_manifest) + "\n")
    for row in checked_rows:
        handle.write(_canonical_json(row) + "\n")
    return len(checked_rows)


@dataclass
class Position:
    """One decision, with Mortal's opinion of every legal answer.

    `rewards[i]` is the normalised Q-advantage of `menu[i]`: 1.0 for Mortal's
    own choice, 0.0 for its worst, linear between. That is the per-decision term of the
    mjai-reviewer
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

    def to_task_dict(self) -> BankRow:
        row: dict[str, Any] = {
            "record_type": "position",
            "schema_version": BANK_SCHEMA_VERSION,
            "prompt": self.prompt(state_hints=True),
            "prompt_without_state_hints": self.prompt(state_hints=False),
            "system_prompt": prompts.SYSTEM,
            "menu": list(self.menu),
            "rewards": list(self.rewards),
            "best_index": self.best_index,
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
